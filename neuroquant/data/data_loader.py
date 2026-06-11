"""
NeuroQuant v2.0 - Generic Data Loading Module

Provides a unified interface for loading ANY dataset
(torchvision built-ins, ImageFolder, or synthetic)
and returning standard DataLoaders for train/val/test/calibration.

No hardcoded crop sizes or architecture assumptions.
All spatial dimensions are driven by config.input_shape.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split

from neuroquant.config import QuantizationConfig

logger = logging.getLogger("neuroquant")

# Optional torchvision. We prefer ``torchvision.transforms.v2`` because
# it is the only API that natively synchronises image transforms with
# bounding boxes (``tv_tensors.BoundingBoxes``) and segmentation masks
# (``tv_tensors.Mask``). The legacy ``torchvision.transforms`` is kept
# only as a fallback for old wheels — the project's pinned
# torchvision==0.22.1 ships v2 unconditionally.
HAS_TORCHVISION = False
HAS_TRANSFORMS_V2 = False
T: Any = None
try:
    import torchvision
    HAS_TORCHVISION = True
    try:
        from torchvision.transforms import v2 as T  # type: ignore[attr-defined]
        HAS_TRANSFORMS_V2 = True
    except ImportError:  # pragma: no cover — torchvision < 0.16
        import torchvision.transforms as T  # type: ignore[no-redef]
        HAS_TRANSFORMS_V2 = False
except ImportError:
    HAS_TORCHVISION = False
    HAS_TRANSFORMS_V2 = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detection collate function (module-level so it pickles cleanly across
# DataLoader workers on platforms that use spawn instead of fork).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _seed_worker(worker_id: int) -> None:
    """Seed numpy + python RNG per DataLoader worker for reproducibility.

    The torch RNG is forked deterministically by the DataLoader from its
    ``generator``, but augmentation that draws from ``numpy.random`` /
    ``random`` (not torch) would otherwise get an unseeded stream in each
    worker — making multi-worker runs non-reproducible despite
    ``set_seed``. Windows forces ``num_workers=0`` so this only matters on
    POSIX, but wiring it makes the strict-determinism contract hold there
    too (L8).
    """
    import random as _random

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def detection_collate_fn(batch: List[Any]) -> Tuple[Tuple[Any, ...], ...]:
    """Collate function for object-detection batches.

    Torchvision detection models consume ``(List[Tensor], List[Dict])``
    — one tensor per image, one target-dict per image — because each
    image can carry a different number of bounding boxes. The default
    ``torch.utils.data.default_collate`` tries to stack everything into
    a single tensor and crashes on the ragged box list.

    This collate keeps the batch as a tuple of tuples
    ``(images_tuple, targets_tuple)``. Downstream the user can hand
    ``images_tuple`` to the model and forward ``targets_tuple`` to a
    detection-aware loss / evaluator without further transformation.
    """
    return tuple(zip(*batch))


class _HFDictDataset(Dataset):
    """Adapter so a HuggingFace ``Dataset`` plays nicely with PyTorch's
    ``DataLoader``.

    HF datasets in ``set_format("torch")`` mode return a dict per row,
    which the default ``torch.utils.data.default_collate`` happily
    stacks key-by-key into a single dict-of-tensors per batch — exactly
    what the NLP loss bridge expects. We wrap the HF dataset so
    ``__len__`` and ``__getitem__`` come from PyTorch's protocol
    rather than ``datasets.Dataset``'s slightly different one.
    """

    def __init__(self, hf_dataset: Any) -> None:
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._ds[int(idx)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-dataset normalisation statistics (correct constants, not assumptions)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET_STATS: Dict[str, Dict[str, Tuple[float, ...]]] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2023, 0.1994, 0.2010),
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std":  (0.229, 0.224, 0.225),
    },
}


class GenericDatasetLoader:
    """
    Generic dataset loader that supports:
    - TorchVision built-in datasets (CIFAR-10, CIFAR-100, etc.)
    - ImageFolder-based custom datasets
    - User-supplied Dataset classes via dataset.class
    - Synthetic random datasets (for testing without real data)

    All spatial dimensions are driven by config.input_shape.
    Returns standardized DataLoaders for all pipeline phases.
    """

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self.dataset_name = config.dataset_name.lower()
        self.dataset_path = config.dataset_path
        self.dataset_class = config.dataset_class
        self.batch_size = config.batch_size
        self.num_workers = self._resolve_num_workers(config.num_workers)
        self.input_shape = config.input_shape  # (C, H, W)
        self.split_seed = int(config.hyperparams.seed)

        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        # Held-out NSGA-search slice with eval-time transforms so search
        # fitness measurements are not corrupted by random augmentation
        # and the public ``test`` set is never touched during search.
        self._search_dataset: Optional[Dataset] = None
        # Eval-transform view of the training split used for calibration
        # (PTQ/GPTQ/AWQ scale search, Fisher sensitivity, AdaRound
        # reconstruction). Built by the loaders whose training transform
        # augments (torchvision, ImageFolder) so calibration always sees
        # inference-time preprocessing, never RandomCrop/flip/jitter (C3).
        # Stays ``None`` for loaders whose training data is already
        # eval-transformed (custom Dataset class, synthetic, HuggingFace),
        # where ``get_calibration_loader`` falls back to ``_train_dataset``.
        self._calib_dataset: Optional[Dataset] = None

        self._load()

    def _load(self) -> None:
        """Load the dataset based on config."""
        # HuggingFace ``datasets`` integration is gated on the ``[nlp]``
        # extras (``transformers`` + ``datasets``). Activated when the
        # dataset name starts with ``hf:`` (e.g. ``hf:imdb``) so it
        # never collides with torchvision's namespace.
        if self.dataset_name.startswith("hf:") or getattr(self.config, "task", "") == "nlp":
            self._load_huggingface_dataset()
            return
        if self.dataset_class:
            self._load_custom_dataset_class()
        elif self.dataset_name == "cifar10":
            self._load_torchvision_dataset("CIFAR10", "cifar10")
        elif self.dataset_name == "cifar100":
            self._load_torchvision_dataset("CIFAR100", "cifar100")
        elif self.dataset_name == "imagefolder":
            self._load_imagefolder()
        elif self.dataset_name == "synthetic":
            self._load_synthetic()
        else:
            # Try to load as a torchvision dataset by name
            self._load_torchvision_dataset(self.dataset_name, self.dataset_name)

    @staticmethod
    def _resolve_num_workers(num_workers: int) -> int:
        """Use configured workers except on Windows, where forkless loading is safer."""
        if sys.platform == "win32":
            return 0
        return max(0, int(num_workers))

    def _split_generator(self, offset: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(self.split_seed + int(offset))

    def _spatial_hw(self) -> Tuple[int, int]:
        spatial_h = self.input_shape[1] if len(self.input_shape) >= 3 else self.input_shape[-1]
        spatial_w = self.input_shape[2] if len(self.input_shape) >= 3 else self.input_shape[-1]
        return spatial_h, spatial_w

    def _build_v2_transform(
        self,
        *,
        train: bool,
        stats_key: str = "imagenet",
        cifar_native: bool = False,
    ) -> Any:
        """Task-aware torchvision-v2 transform pipeline.

        Returns a ``v2.Compose`` whose steps differ only by what the
        current ``config.task`` needs:

          * ``classification`` — spatial resize / random-crop, optional
            random horizontal flip on train, ``ToImage``, ``ToDtype``
            (float32, scaled), ``Normalize``.
          * ``detection`` — same spatial transforms, but v2 dispatches
            them onto ``tv_tensors.BoundingBoxes`` targets so boxes are
            resized / flipped in lockstep with the image. ``Normalize``
            is type-aware: it only acts on the ``Image`` input,
            leaving boxes/labels untouched. The dataset is expected to
            yield targets in the torchvision-v2 contract (wrap with
            ``tv_tensors.wrap_dataset_for_transforms_v2`` if needed —
            this loader does not mutate the user's dataset).
          * ``segmentation`` — same spatial transforms; v2's
            ``Resize`` kernel automatically falls back to
            ``InterpolationMode.NEAREST`` whenever the input is a
            ``tv_tensors.Mask``, so per-pixel class indices are never
            blurred into floating-point values. Images keep the
            ``BILINEAR`` interpolation. Again, the dataset is expected
            to wrap mask targets as ``tv_tensors.Mask``.

        Falls back to the legacy v1 pipeline when v2 is unavailable
        (torchvision < 0.16). The v1 path cannot synchronise targets,
        so detection / segmentation users on old torchvision wheels
        will see a single-warning notice and degraded behaviour.

        Args:
            train: ``True`` enables augmentation (flip / random crop).
            stats_key: Normalisation stats key into ``DATASET_STATS``.
            cifar_native: ``True`` only for CIFAR-class datasets at
                their native 32×32 resolution — enables the canonical
                ``RandomCrop(32, padding=4)`` augmentation that
                materially improves CIFAR accuracy. Ignored for any
                task other than ``classification``.
        """
        if not HAS_TORCHVISION:
            return None

        h, w = self._spatial_hw()
        stats = DATASET_STATS.get(stats_key, DATASET_STATS["imagenet"])
        mean, std = list(stats["mean"]), list(stats["std"])
        task = getattr(self.config, "task", "classification")

        # ── v1 legacy path (torchvision < 0.16) ──
        if not HAS_TRANSFORMS_V2:
            if task != "classification":
                logger.warning(
                    "torchvision.transforms.v2 unavailable; detection and "
                    "segmentation targets WILL NOT be transformed in sync "
                    "with the image. Upgrade torchvision to >= 0.16."
                )
            legacy_steps: List[Any] = [T.Resize((h, w))]
            if train:
                legacy_steps.append(T.RandomHorizontalFlip())
            legacy_steps.append(T.ToTensor())
            legacy_steps.append(T.Normalize(mean, std))
            return T.Compose(legacy_steps)

        # ── v2 path (preferred) ──
        steps: List[Any] = []

        # 1. Spatial transforms — task-aware. For classification on a
        #    CIFAR-native input we keep the canonical RandomCrop+Flip
        #    pair; everything else uses Resize+optional-flip so batches
        #    collate to a uniform spatial size.
        if task == "classification" and cifar_native and train:
            steps.append(T.RandomCrop(h, padding=max(1, h // 8)))
            steps.append(T.RandomHorizontalFlip(p=0.5))
        else:
            # ``Resize`` accepts ``(h, w)``. The interpolation for image
            # inputs stays at BILINEAR (the constructor default); for
            # ``tv_tensors.Mask`` v2's kernel dispatch silently
            # overrides it to NEAREST so class indices stay integer.
            steps.append(T.Resize((h, w)))
            if train:
                steps.append(T.RandomHorizontalFlip(p=0.5))

        # 2. Type conversion. ``ToImage`` upgrades PIL / ndarray inputs
        #    into ``tv_tensors.Image`` so every subsequent transform
        #    can dispatch on the canonical tv_tensor types.
        #    ``ToDtype(float32, scale=True)`` rescales uint8 [0, 255] →
        #    float32 [0, 1] — the contract Normalize expects.
        steps.append(T.ToImage())
        steps.append(T.ToDtype(torch.float32, scale=True))

        # 3. Normalize is type-aware in v2: it only acts on Image
        #    tensors, leaving any BoundingBoxes / Mask targets in the
        #    sample untouched. No need to special-case detection /
        #    segmentation here.
        steps.append(T.Normalize(mean=mean, std=std))

        return T.Compose(steps)

    def _eval_transform(self):
        """Legacy alias: build an eval-time (non-augmenting) pipeline.

        Kept for callers (e.g. ``_load_custom_dataset_class``) that
        request a single transform. Routes through
        :meth:`_build_v2_transform` so the task-awareness applies.
        """
        return self._build_v2_transform(train=False)

    def _collate_fn(self) -> Optional[Any]:
        """Return the right ``collate_fn`` for the active task.

        Detection batches have ragged target lists (one dict per
        image with variable box count), so the default
        ``torch.utils.data.default_collate`` crashes. We use the
        module-level :func:`detection_collate_fn` instead. Classification
        and segmentation batches stack normally — they return ``None``
        so the DataLoader keeps its default collate.
        """
        if getattr(self.config, "task", "classification") == "detection":
            return detection_collate_fn
        return None

    def _resolve_split_dir(
        self,
        configured: Optional[str],
        default: Optional[Path],
    ) -> Optional[Path]:
        if configured:
            p = Path(configured)
            return p if p.is_absolute() else Path(self.dataset_path) / p
        return default

    def _split_train_search_val(
        self,
        full_train: Dataset,
        *,
        generator_offset: int = 0,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        n = len(full_train)
        n_val = max(n // 10, 1)
        n_search = max(n // 10, 1)
        n_train = n - n_val - n_search
        if n_train < 1:
            raise ValueError(
                "Dataset is too small to split into train/search/val; "
                f"got {n} samples."
            )
        return random_split(
            full_train, [n_train, n_search, n_val],
            generator=self._split_generator(generator_offset),
        )

    @staticmethod
    def _callable_accepts_arg(obj: Any, name: str) -> bool:
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            return True
        return (
            name in params
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        )

    def _instantiate_dataset_class(
        self,
        cls: Any,
        *,
        split: Optional[str] = None,
        train: Optional[bool] = None,
        transform: Any = None,
    ) -> Dataset:
        kwargs: Dict[str, Any] = {}
        for root_key in ("root", "data_dir", "path"):
            if self._callable_accepts_arg(cls, root_key):
                kwargs[root_key] = str(self.dataset_path)
                break
        if split is not None and self._callable_accepts_arg(cls, "split"):
            kwargs["split"] = split
        if train is not None and self._callable_accepts_arg(cls, "train"):
            kwargs["train"] = train
        if transform is not None and self._callable_accepts_arg(cls, "transform"):
            kwargs["transform"] = transform
        if self._callable_accepts_arg(cls, "download"):
            kwargs["download"] = False

        try:
            ds = cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Could not instantiate dataset class '{self.dataset_class}' "
                f"with supported arguments {sorted(kwargs)}. Use a constructor "
                "accepting root/data_dir/path, split or train, and optional transform."
            ) from exc
        if not isinstance(ds, Dataset):
            raise TypeError(
                f"dataset.class '{self.dataset_class}' returned {type(ds)!r}, "
                "expected torch.utils.data.Dataset."
            )
        return ds

    def _load_custom_dataset_class(self) -> None:
        """Load a user-supplied Dataset class from dataset.class."""
        parts = str(self.dataset_class).rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                "dataset.class must be fully qualified, e.g. "
                "'my_pkg.my_data.MyDataset'."
            )
        module_name, class_name = parts
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        transform = self._eval_transform()

        supports_split = self._callable_accepts_arg(cls, "split")
        supports_train = self._callable_accepts_arg(cls, "train")

        if supports_split:
            full_train = self._instantiate_dataset_class(
                cls, split="train", transform=transform,
            )
            try:
                self._test_dataset = self._instantiate_dataset_class(
                    cls, split="test", transform=transform,
                )
            except Exception:
                try:
                    self._test_dataset = self._instantiate_dataset_class(
                        cls, split="val", transform=transform,
                    )
                except Exception:
                    self._test_dataset = None
        elif supports_train:
            full_train = self._instantiate_dataset_class(
                cls, train=True, transform=transform,
            )
            try:
                self._test_dataset = self._instantiate_dataset_class(
                    cls, train=False, transform=transform,
                )
            except Exception:
                self._test_dataset = None
        else:
            full = self._instantiate_dataset_class(cls, transform=transform)
            n = len(full)
            n_test = max(n // 10, 1)
            n_val = max(n // 10, 1)
            n_search = max(n // 10, 1)
            n_train = n - n_test - n_val - n_search
            if n_train < 1:
                raise ValueError(
                    "Custom dataset is too small to split into "
                    f"train/search/val/test; got {n} samples."
                )
            (
                self._train_dataset,
                self._search_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = random_split(
                full, [n_train, n_search, n_val, n_test],
                generator=self._split_generator(),
            )
            logger.info(
                "Loaded custom dataset %s: train=%d, search=%d, val=%d, test=%d",
                self.dataset_class, n_train, n_search, n_val, n_test,
            )
            return

        (
            self._train_dataset,
            self._search_dataset,
            self._val_dataset,
        ) = self._split_train_search_val(full_train)
        if self._test_dataset is None:
            self._test_dataset = self._val_dataset
            logger.warning(
                "Custom dataset %s did not expose a test split; using val split.",
                self.dataset_class,
            )
        logger.info(
            "Loaded custom dataset %s: train=%d, search=%d, val=%d, test=%d",
            self.dataset_class,
            len(self._train_dataset), len(self._search_dataset),
            len(self._val_dataset), len(self._test_dataset),
        )

    # ------------------------------------------------------------------
    # TorchVision datasets (generic loader)
    # ------------------------------------------------------------------

    def _load_torchvision_dataset(self, tv_class_name: str, stats_key: str) -> None:
        """
        Load any torchvision dataset by class name.

        Uses config.input_shape to determine crop/resize size.
        Falls back to ImageNet normalisation if stats not known.
        """
        if not HAS_TORCHVISION:
            raise ImportError(f"torchvision required for {tv_class_name}")

        # Determine spatial size from config
        spatial_h = self.input_shape[1] if len(self.input_shape) >= 3 else self.input_shape[-1]
        spatial_w = self.input_shape[2] if len(self.input_shape) >= 3 else self.input_shape[-1]

        # The CIFAR-canonical RandomCrop+Flip augmentation is only safe
        # when the dataset is being used at its native 32×32 resolution
        # (so the crop window is meaningful). For up-/down-sized inputs
        # we fall back to plain Resize+Flip — the builder handles both
        # cases via the ``cifar_native`` flag.
        native_sizes = {"cifar10": 32, "cifar100": 32}
        native = native_sizes.get(stats_key)
        cifar_native = bool(native and native == spatial_h)

        train_transform = self._build_v2_transform(
            train=True, stats_key=stats_key, cifar_native=cifar_native,
        )
        test_transform = self._build_v2_transform(
            train=False, stats_key=stats_key, cifar_native=cifar_native,
        )

        # Resolve the dataset class from torchvision
        ds_cls = getattr(torchvision.datasets, tv_class_name, None)
        if ds_cls is None:
            raise ValueError(
                f"torchvision.datasets.{tv_class_name} not found. "
                f"Available: {[n for n in dir(torchvision.datasets) if n[0].isupper()][:15]}..."
            )

        data_dir = str(self.dataset_path)
        # Keep train-time augmentation ONLY on train split. Validation uses
        # deterministic eval transforms to avoid stochastic metric drift.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align.*",
                category=Warning,
            )
            full_train_aug = ds_cls(
                root=data_dir, train=True, download=True,
                transform=train_transform,
            )
            full_train_eval = ds_cls(
                root=data_dir, train=True, download=True,
                transform=test_transform,
            )
            self._test_dataset = ds_cls(
                root=data_dir, train=False, download=True,
                transform=test_transform,
            )

        # Split train into train (80%) / search (10%) / val (10%) with
        # shared indices and the configured seed so splits are reproducible
        # across runs. ``search`` is the NSGA-II fitness slice — it must
        # NOT be ``val`` (used for QAT early-stop) or ``test`` (used for
        # the public headline number). All three slices share the
        # eval-time transform so deterministic metrics flow through them;
        # only ``train`` keeps the augmenting transform.
        n = len(full_train_aug)
        n_val = max(n // 10, 1)
        n_search = max(n // 10, 1)
        n_train = n - n_val - n_search
        indices = torch.randperm(n, generator=self._split_generator()).tolist()
        train_idx = indices[:n_train]
        search_idx = indices[n_train:n_train + n_search]
        val_idx = indices[n_train + n_search:]

        self._train_dataset = Subset(full_train_aug, train_idx)
        self._search_dataset = Subset(full_train_eval, search_idx)
        self._val_dataset = Subset(full_train_eval, val_idx)
        # Calibration draws the SAME training samples but through the
        # deterministic eval transform (C3): the underlying dataset has a
        # fixed sample order, so ``train_idx`` selects identical images to
        # ``full_train_aug`` minus the random augmentation.
        self._calib_dataset = Subset(full_train_eval, train_idx)

        logger.info(
            "Loaded %s: train=%d, search=%d, val=%d, test=%d (spatial=%dx%d)",
            tv_class_name, n_train, n_search, n_val,
            len(self._test_dataset), spatial_h, spatial_w,
        )

    # ------------------------------------------------------------------
    # ImageFolder
    # ------------------------------------------------------------------

    def _load_imagefolder(self) -> None:
        """Load custom dataset from ImageFolder directory structure."""
        if not HAS_TORCHVISION:
            raise ImportError("torchvision required for ImageFolder")

        spatial_h, spatial_w = self._spatial_hw()

        # Separate train (augmenting) vs eval (deterministic) transforms.
        # ImageFolder is used most often for classification on
        # ImageNet-style data, but the builder is task-aware so a
        # detection / segmentation user pointing at a v2-compatible
        # dataset still gets synchronised box / mask transforms.
        train_transform = self._build_v2_transform(
            train=True, stats_key="imagenet",
        )
        eval_transform = self._build_v2_transform(
            train=False, stats_key="imagenet",
        )

        data_dir = Path(self.dataset_path)
        train_dir = self._resolve_split_dir(
            self.config.dataset_train_dir, data_dir / "train",
        )
        val_dir = self._resolve_split_dir(self.config.dataset_val_dir, None)
        test_dir = self._resolve_split_dir(
            self.config.dataset_test_dir, data_dir / "test",
        )

        if train_dir and train_dir.exists():
            full_train = torchvision.datasets.ImageFolder(
                str(train_dir), transform=train_transform
            )
            if val_dir and val_dir.exists():
                self._val_dataset = torchvision.datasets.ImageFolder(
                    str(val_dir), transform=eval_transform,
                )
                n = len(full_train)
                n_search = max(n // 10, 1)
                n_train = n - n_search
                if n_train < 1:
                    raise ValueError(
                        "ImageFolder train split is too small to reserve "
                        f"a search split; got {n} samples."
                    )
                self._train_dataset, self._search_dataset = random_split(
                    full_train, [n_train, n_search],
                    generator=self._split_generator(),
                )
            else:
                (
                    self._train_dataset,
                    self._search_dataset,
                    self._val_dataset,
                ) = self._split_train_search_val(full_train)
        else:
            raise FileNotFoundError(f"ImageFolder train dir not found: {train_dir}")

        # Eval-transform calibration view aligned to the train split (C3):
        # re-open the same train directory with the non-augmenting
        # transform, then mirror whatever Subset indices the train split
        # ended up with so calibration sees inference-time preprocessing.
        full_train_eval = torchvision.datasets.ImageFolder(
            str(train_dir), transform=eval_transform,
        )
        if isinstance(self._train_dataset, Subset):
            self._calib_dataset = Subset(
                full_train_eval, self._train_dataset.indices,
            )
        else:
            self._calib_dataset = full_train_eval

        if test_dir and test_dir.exists():
            self._test_dataset = torchvision.datasets.ImageFolder(
                str(test_dir), transform=eval_transform
            )
        else:
            # Use val split as test
            self._test_dataset = self._val_dataset
            logger.warning("No test dir found at %s, using val split.", test_dir)

        logger.info(
            "Loaded ImageFolder: train=%s, val=%s, test=%s (spatial=%dx%d)",
            train_dir, val_dir or "<split-from-train>",
            test_dir if test_dir and test_dir.exists() else "<val>",
            spatial_h, spatial_w,
        )

    # ------------------------------------------------------------------
    # Synthetic dataset (for testing without real data)
    # ------------------------------------------------------------------

    def _load_synthetic(self) -> None:
        """
        Generate a synthetic random dataset for testing.

        Uses config.input_shape and config.num_classes.
        """
        n_train = 500
        n_test = 100
        c, h, w = self.input_shape
        num_classes = self.config.num_classes
        data_gen = self._split_generator()

        train_images = torch.randn(n_train, c, h, w, generator=data_gen)
        train_labels = torch.randint(0, num_classes, (n_train,), generator=data_gen)
        test_images = torch.randn(n_test, c, h, w, generator=data_gen)
        test_labels = torch.randint(0, num_classes, (n_test,), generator=data_gen)

        full_train = TensorDataset(train_images, train_labels)
        n_val = max(n_train // 10, 1)
        n_search = max(n_train // 10, 1)
        n_tr = n_train - n_val - n_search
        (
            self._train_dataset,
            self._search_dataset,
            self._val_dataset,
        ) = random_split(
            full_train, [n_tr, n_search, n_val],
            generator=self._split_generator(offset=1),
        )
        self._test_dataset = TensorDataset(test_images, test_labels)

        logger.info(
            "Loaded synthetic dataset: train=%d, search=%d, val=%d, test=%d "
            "(shape=%s, classes=%d)",
            n_tr, n_search, n_val, n_test, self.input_shape, num_classes,
        )

    # ------------------------------------------------------------------
    # HuggingFace dataset loader (optional [nlp] extras)
    # ------------------------------------------------------------------

    def _load_huggingface_dataset(self) -> None:
        """Load a tokenised HuggingFace dataset for NLP-task pipelines.

        Activated when ``dataset_name`` starts with ``hf:`` (e.g.
        ``hf:imdb`` or ``hf:glue/sst2``) or when ``task == "nlp"``.
        Requires the optional ``[nlp]`` extras:

        ```
        pip install neuroquant[nlp]
        ```

        Behaviour:
          * Strips the ``hf:`` prefix and splits on ``/`` into
            ``(dataset, subset)``. ``hf:imdb`` → ``load_dataset("imdb")``;
            ``hf:glue/sst2`` → ``load_dataset("glue", "sst2")``.
          * Picks the tokenizer from ``config.model_name`` so the
            framework follows the same single-source-of-truth convention
            it uses for CV (model and dataset can be swapped together).
          * Tokenises ``text`` / ``sentence`` (auto-detected) into
            ``input_ids`` + ``attention_mask`` and stores the original
            ``label`` field as ``labels``.
          * Wraps each split in a ``_HFDictDataset`` that yields the
            tokenizer-output dict so the NLP loss bridge in
            ``cli.py:_build_task_loss_fn`` can splat it into the model
            unchanged.

        The ImportError is propagated with an actionable hint instead
        of swallowed — the user *asked* for NLP, so silently downgrading
        to a synthetic dataset would mask the missing dep.
        """
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
            from transformers import AutoTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "HuggingFace datasets / transformers not installed. "
                "Run `pip install neuroquant[nlp]` to add them, or set "
                "task back to 'classification' and use a torchvision "
                "dataset."
            ) from exc

        spec = self.dataset_name.removeprefix("hf:") or self.dataset_name
        if "/" in spec:
            ds_name, subset = spec.split("/", 1)
            raw = load_dataset(ds_name, subset)
        else:
            raw = load_dataset(spec)

        # Resolve the tokenizer from ``model.name``. Defaults to
        # bert-base-uncased so the loader works for ``hf:imdb`` even
        # when the user hasn't picked a model — they can override via
        # ``config.model_name``.
        model_name = getattr(self.config, "model_name", None) or "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect the text column — HF datasets are inconsistent:
        # imdb uses ``text``, sst2 uses ``sentence``, etc.
        sample = raw[next(iter(raw))][0]
        text_col = next(
            (k for k in ("text", "sentence", "premise", "question") if k in sample),
            None,
        )
        if text_col is None:
            raise ValueError(
                f"Could not auto-detect text column in HF dataset "
                f"'{spec}'. Available columns: {list(sample.keys())}. "
                "Pre-tokenise manually and pass via dataset_class instead."
            )

        max_len = int(getattr(self.config.hyperparams, "nlp_max_seq_len", 128))

        def _tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
            enc = tokenizer(
                example[text_col],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors=None,
            )
            return enc

        tokenised = raw.map(_tokenize, batched=True)
        tokenised = tokenised.rename_column("label", "labels") \
            if "label" in tokenised[next(iter(tokenised))].column_names \
            else tokenised
        tokenised.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        # Pick canonical splits, falling back to the first available
        # name if a dataset uses something custom.
        split_keys = list(tokenised.keys())
        train_key = "train" if "train" in split_keys else split_keys[0]
        val_key = next((k for k in ("validation", "valid", "dev") if k in split_keys), None)
        test_key = "test" if "test" in split_keys else None

        self._train_dataset = _HFDictDataset(tokenised[train_key])
        if val_key:
            self._val_dataset = _HFDictDataset(tokenised[val_key])
        if test_key:
            self._test_dataset = _HFDictDataset(tokenised[test_key])
        else:
            # No test split — reuse val so the rest of the pipeline
            # has something to evaluate on.
            self._test_dataset = self._val_dataset or self._train_dataset
        self._search_dataset = self._val_dataset

        logger.info(
            "Loaded HuggingFace dataset '%s' (tokenizer=%s, max_len=%d). "
            "Splits: train=%d, val=%s, test=%s",
            spec, model_name, max_len,
            len(self._train_dataset),
            len(self._val_dataset) if self._val_dataset else "n/a",
            len(self._test_dataset) if self._test_dataset else "n/a",
        )

    # ------------------------------------------------------------------
    # DataLoader getters
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """Return training DataLoader.

        Uses :meth:`_collate_fn` so detection batches (ragged
        per-image target dicts) collate via :func:`detection_collate_fn`
        instead of the default stacker that would crash on the
        variable box count.
        """
        # Seed the shuffle RNG so epoch ordering is reproducible, and seed
        # each worker's numpy/random stream so any non-torch augmentation
        # is too — part of the strict-determinism contract ``set_seed``
        # advertises (L8). ``worker_init_fn`` is only meaningful when
        # workers are actually forked.
        gen = torch.Generator()
        gen.manual_seed(self.split_seed)
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._collate_fn(),
            generator=gen,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
        )

    def get_val_loader(self) -> DataLoader:
        """Return validation DataLoader."""
        ds = self._val_dataset or self._test_dataset
        return DataLoader(
            ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate_fn(),
        )

    def get_test_loader(self) -> DataLoader:
        """Return test DataLoader.

        The test split is the *only* slice used for the public headline
        number reported in the summary table. Nothing in the pipeline
        (training, NSGA fitness, QAT early-stop) ever reads from it
        before the final evaluation pass, so its accuracy is an unbiased
        estimate of deployment-time performance.
        """
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate_fn(),
        )

    def get_search_loader(self) -> DataLoader:
        """Return the NSGA-search DataLoader.

        Held-out 10% slice of the original training set, with eval-time
        (non-augmenting) transforms. NSGA-II reads only from this loader
        when scoring candidate quantization configs; ``val_loader`` is
        reserved for QAT early-stopping and ``test_loader`` for the
        final report. This separation prevents the val/test contamination
        that the previous single-loader design introduced.

        Falls back to ``val`` then ``test`` for older datasets that did
        not pre-build a search slice (only relevant on legacy resumes).
        """
        ds = self._search_dataset or self._val_dataset or self._test_dataset
        return DataLoader(
            ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate_fn(),
        )

    def get_class_names(self) -> Optional[List[str]]:
        """Return the dataset's class-name list if it is exposed.

        torchvision datasets (CIFAR10, ImageFolder, etc.) attach a
        ``.classes`` attribute. Synthetic and unknown datasets return
        ``None`` so the caller can fall back to numeric indices.
        """
        for ds in (
            self._test_dataset, self._val_dataset,
            self._search_dataset, self._train_dataset,
        ):
            if ds is None:
                continue
            base = ds.dataset if isinstance(ds, Subset) else ds
            classes = getattr(base, "classes", None)
            if classes:
                return list(classes)
        return None

    def get_calibration_loader(self, num_batches: int = 20) -> DataLoader:
        """Return a calibration DataLoader (subset of training data).

        Honours the task-aware ``collate_fn`` so detection calibration
        receives ``(images_tuple, targets_tuple)`` batches, matching
        the contract every quantizer expects from the rest of the
        pipeline.
        """
        # Prefer the eval-transform calibration view when the loader built
        # one (torchvision / ImageFolder). Falls back to the training
        # dataset for loaders whose training data is already
        # eval-transformed (custom Dataset class / synthetic / HuggingFace),
        # so calibration never runs on randomly-augmented images (C3).
        calib_source = (
            self._calib_dataset if self._calib_dataset is not None
            else self._train_dataset
        )
        n_samples = min(num_batches * self.batch_size, len(calib_source))
        subset = Subset(calib_source, list(range(n_samples)))
        return DataLoader(
            subset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate_fn(),
        )

    def get_sample_images(
        self, num_samples: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sample images and labels for XAI visualization."""
        ds = self._test_dataset
        images, labels = [], []
        for i in range(min(num_samples, len(ds))):
            img, lbl = ds[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

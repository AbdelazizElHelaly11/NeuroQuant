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

        self._load()

    def _load(self) -> None:
        """Load the dataset based on config."""
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
    # DataLoader getters
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """Return training DataLoader.

        Uses :meth:`_collate_fn` so detection batches (ragged
        per-image target dicts) collate via :func:`detection_collate_fn`
        instead of the default stacker that would crash on the
        variable box count.
        """
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._collate_fn(),
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
        n_samples = min(num_batches * self.batch_size, len(self._train_dataset))
        subset = Subset(self._train_dataset, list(range(n_samples)))
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

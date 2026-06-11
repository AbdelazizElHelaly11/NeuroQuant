"""
NeuroQuant — Pascal VOC 2012 semantic-segmentation dataset adapter.

NeuroQuant's generic loaders cannot drive ``torchvision.datasets.VOCSegmentation``
directly for two reasons:

  1. VOCSegmentation's constructor takes ``image_set="train"/"val"`` — not
     the ``train=`` / ``split=`` kwargs the generic loader probes for.
  2. Segmentation needs the **mask resized in lockstep with the image**
     (image → bilinear, mask → nearest, so class indices stay integer).
     The generic loader only passes an image ``transform=``, which would
     leave the mask un-resized and break collation / the loss.

This adapter solves both. It reads the standard VOCdevkit layout
directly (so it does not depend on torchvision's download/extract logic
inside a compute job) and returns ``(image[3,S,S] float, mask[S,S] long)``
pairs with the boundary/ignore label 255 preserved.

Wire it via ``dataset.class`` in the YAML::

    dataset:
      class: "neuroquant.data.voc_segmentation.VOCSegmentationDataset"
      path: "./data"        # dir that contains VOCdevkit/VOC2012/...

The NeuroQuant loader probes the ``split`` kwarg, so it instantiates the
adapter as ``split="train"`` (the 1464-image train set, which NeuroQuant
further splits 80/10/10 into train/search/val) and ``split="test"`` (which
this adapter maps to the 1449-image VOC **val** set — the standard mIOU
benchmark). The image ``transform`` the loader passes is accepted and
ignored: the adapter always applies its own synchronized resize +
ImageNet normalization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("neuroquant")

# ImageNet statistics — torchvision's pretrained DeepLabV3 / FCN / LRASPP
# segmentation weights were trained with these, so the FP32 baseline is
# only meaningful when inputs are normalized the same way.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Pascal VOC boundary / "void" pixels carry label 255 and are excluded
# from both the loss and the mIOU.
VOC_IGNORE_INDEX = 255
VOC_NUM_CLASSES = 21  # 20 object categories + background


def _resolve_voc_root(root: str, year: str) -> Path:
    """Locate the ``VOC<year>`` directory under a few common layouts.

    Accepts a path that is the VOC2012 dir itself, a dir containing
    ``VOCdevkit/VOC<year>``, or a dir containing ``VOC<year>``.
    """
    base = Path(root).expanduser()
    candidates = [
        base / "VOCdevkit" / f"VOC{year}",
        base / f"VOC{year}",
        base,
    ]
    for c in candidates:
        if (c / "ImageSets" / "Segmentation").is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find a Pascal VOC{year} segmentation layout under "
        f"'{root}'. Expected one of: "
        + ", ".join(str(c) for c in candidates)
        + ". Download it on the login node with "
        "`python -m neuroquant.data.voc_segmentation --download --root <dir>` "
        "or torchvision's VOCSegmentation(download=True)."
    )


class VOCSegmentationDataset(Dataset):
    """Pascal VOC 2012 segmentation, returning fixed-size tensor pairs.

    Args:
        root: Directory containing the VOCdevkit layout (see
            :func:`_resolve_voc_root` for the accepted shapes).
        split: ``"train"`` → VOC ``train`` (1464 imgs); ``"val"``/``"test"``
            → VOC ``val`` (1449 imgs); ``"trainval"`` → both. NeuroQuant's
            loader passes ``"train"`` and ``"test"``.
        transform: Accepted for loader compatibility and **ignored** — the
            adapter always applies its own synchronized resize + normalize.
        download: When ``True`` and the layout is missing, fall back to
            ``torchvision.datasets.VOCSegmentation(download=True)`` to fetch
            it, then re-resolve. NeuroQuant forces this to ``False`` inside
            jobs, so download on the login node up front.
        year: VOC year (only ``"2012"`` is standard for segmentation).
        image_size: Square side the image and mask are resized to (513 is
            the AdaRound / DeepLabV3 convention).
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Any = None,
        download: bool = False,
        *,
        year: str = "2012",
        image_size: int = 513,
    ) -> None:
        del transform  # accepted for loader compatibility, intentionally unused
        self.root = root
        self.year = str(year)
        self.image_size = int(image_size)

        image_set = {
            "train": "train",
            "trainval": "trainval",
            "val": "val",
            "valid": "val",
            "test": "val",   # VOC2012 has no public seg test set → use val
        }.get(str(split).lower(), str(split).lower())
        self.image_set = image_set

        try:
            voc_dir = _resolve_voc_root(root, self.year)
        except FileNotFoundError:
            if download:
                voc_dir = self._download_then_resolve(root)
            else:
                raise

        self._jpeg_dir = voc_dir / "JPEGImages"
        self._mask_dir = voc_dir / "SegmentationClass"
        split_file = voc_dir / "ImageSets" / "Segmentation" / f"{image_set}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"VOC split list not found: {split_file}. "
                f"Available: {[p.name for p in (voc_dir / 'ImageSets' / 'Segmentation').glob('*.txt')]}"
            )

        self._ids: List[str] = [
            line.strip() for line in split_file.read_text().splitlines()
            if line.strip()
        ]
        if not self._ids:
            raise RuntimeError(f"VOC split '{image_set}' is empty ({split_file}).")

        self._mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

        logger.info(
            "VOCSegmentation: split=%s (image_set=%s), %d images, size=%dx%d, root=%s",
            split, image_set, len(self._ids), self.image_size, self.image_size,
            voc_dir,
        )

    @staticmethod
    def _download_then_resolve(root: str) -> Path:
        """Fetch VOC2012 via torchvision, then resolve the layout."""
        try:
            import torchvision  # noqa: WPS433 — optional, only on the download path
        except ImportError as exc:  # pragma: no cover
            raise FileNotFoundError(
                "VOC not present and torchvision is unavailable to download it."
            ) from exc
        logger.info("Downloading Pascal VOC 2012 to %s (one-time, ~2 GB) ...", root)
        torchvision.datasets.VOCSegmentation(
            root=str(root), year="2012", image_set="train", download=True,
        )
        return _resolve_voc_root(root, "2012")

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self._ids[int(idx)]
        img = Image.open(self._jpeg_dir / f"{img_id}.jpg").convert("RGB")
        mask = Image.open(self._mask_dir / f"{img_id}.png")  # P-mode palette

        s = self.image_size
        # Image → bilinear; mask → nearest so the 0..20 / 255 class indices
        # are never blended into fractional values.
        img = img.resize((s, s), Image.BILINEAR)
        mask = mask.resize((s, s), Image.NEAREST)

        img_t = torch.from_numpy(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )
        img_t = (img_t - self._mean) / self._std

        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.int64))  # [H, W]

        return img_t, mask_t

    # torchvision datasets expose ``.classes``; provide it so any caption /
    # class-name lookup downstream has something sensible.
    classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
        "tvmonitor",
    ]


def _cli() -> None:
    """``python -m neuroquant.data.voc_segmentation --download --root ./data``.

    Convenience entry point for the HPC login node: pre-fetch VOC 2012 so
    the (download-disabled) compute job can read it straight off disk.
    """
    import argparse

    ap = argparse.ArgumentParser(description="Download / verify Pascal VOC 2012.")
    ap.add_argument("--root", default="./data", help="Destination root dir.")
    ap.add_argument("--download", action="store_true", help="Fetch if missing.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = VOCSegmentationDataset(
        root=args.root, split="val", download=args.download,
    )
    img, mask = ds[0]
    print(
        f"OK — VOC val ready: {len(ds)} images, "
        f"image{tuple(img.shape)} mask{tuple(mask.shape)} "
        f"(unique labels in sample: {sorted(set(mask.unique().tolist()))[:8]}...)"
    )


if __name__ == "__main__":
    _cli()

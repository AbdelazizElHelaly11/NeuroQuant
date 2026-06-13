"""
NeuroQuant — COCO object-detection dataset adapter (dependency-free).

NeuroQuant's generic loaders cannot drive ``torchvision.datasets.CocoDetection``
directly for three reasons:

  1. ``CocoDetection`` requires ``pycocotools`` to parse the annotation
     JSON — an extra native dependency that is awkward to install inside
     a locked-down HPC job.
  2. Its constructor takes ``root`` + ``annFile`` (two positional paths),
     not the ``split=`` / ``train=`` kwargs the NeuroQuant loader probes.
  3. It yields ``(PIL.Image, List[ann_dict])`` in raw COCO bbox format
     (``[x, y, w, h]``, category ids with gaps) — not the
     ``(Tensor[3,S,S], {"boxes": xyxy, "labels": ...})`` contract every
     torchvision detection model + the NeuroQuant mAP metric expect.

This adapter solves all three. It parses ``instances_*.json`` with the
standard-library ``json`` module (no pycocotools), reads images straight
off disk, and returns fixed-size tensor pairs ready for
``torchvision.models.detection`` models.

**Important — image range.** torchvision detection models wrap their own
``GeneralizedRCNNTransform`` which normalises inputs with ImageNet
mean/std internally. The model therefore expects **un-normalised images
in ``[0, 1]``**. This adapter returns exactly that (a plain
``float / 255``), so the model must do the normalisation — double
normalising here would silently destroy mAP.

**Split partitioning / no data leak.** COCO ``train2017`` is ~18 GB, far
too large for a time-boxed job, so we evaluate on a subset of the 5k-image
``val2017`` set. To keep the NeuroQuant ``train`` / ``search`` / ``val``
slices strictly disjoint from the headline ``test`` slice, the adapter
deterministically partitions the sorted image-id list: the first
``train_frac`` ids form the ``"train"`` split (which NeuroQuant further
sub-splits 80/10/10 into train/search/val) and the remainder forms the
``"test"`` / ``"val"`` split. The two never overlap, so the public mAP is
measured on images no phase ever calibrated, searched, or fine-tuned on.

Wire it via ``dataset.class`` in the YAML::

    dataset:
      class: "neuroquant.data.coco_detection.CocoDetectionDataset"
      path: "./data/coco"        # dir that contains val2017/ + annotations/

Download the ~1 GB subset on the login node first::

    python -m neuroquant.data.coco_detection --download --root ./data/coco
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("neuroquant")

# COCO 2017 category id → name. Ids are sparse (1..90 with gaps); the 80
# real classes are listed here. torchvision's COCO-pretrained detectors
# emit labels in this exact id space (0 = background), so ground-truth
# ``category_id`` is used directly as the label — no remapping needed.
_COCO_ID_TO_NAME: Dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}


def _build_class_list(num_slots: int = 91) -> List[str]:
    """Return a ``label_index -> name`` list (sparse ids filled in)."""
    names = [f"id{i}" for i in range(num_slots)]
    names[0] = "background"
    for cid, name in _COCO_ID_TO_NAME.items():
        if cid < num_slots:
            names[cid] = name
    return names


class CocoDetectionDataset(Dataset):
    """COCO-2017 detection, returning fixed-size ``[0, 1]`` tensor pairs.

    Args:
        root: Directory containing ``val2017/`` (or ``<split>2017/``) and
            ``annotations/instances_<split>2017.json``. A few common
            layouts are probed (see :meth:`_resolve_layout`).
        split: ``"train"`` → the first ``train_frac`` of the sorted,
            capped image-id list; ``"val"`` / ``"test"`` → the remainder.
            NeuroQuant's loader instantiates the adapter twice, with
            ``"train"`` and ``"test"``; the two id sets are disjoint.
        transform: Accepted for loader compatibility and **ignored** — the
            adapter always applies its own resize and returns ``[0, 1]``
            images (the detection model normalises internally).
        download: When ``True`` and the layout is missing, raise an
            actionable error pointing at the module's ``--download`` CLI
            (COCO has no torchvision auto-download). NeuroQuant forces
            this to ``False`` inside jobs anyway.
        year: COCO year directory suffix (``"2017"``).
        coco_split: Which on-disk COCO split to read (``"val"`` keeps the
            footprint at ~1 GB; ``"train"`` is ~18 GB).
        image_size: Square side images are resized to (320 matches the
            ``fasterrcnn_mobilenet_v3_large_320_fpn`` min/max size).
        max_images: Cap the number of images used (keeps a run inside the
            time budget). ``None`` uses every annotated image.
        train_frac: Fraction of the capped image list assigned to the
            ``"train"`` split; the rest is the disjoint ``"test"`` split.
    """

    def __init__(
        self,
        root: str = "./data/coco",
        split: str = "train",
        transform: Any = None,
        download: bool = False,
        *,
        year: str = "2017",
        coco_split: str = "val",
        image_size: int = 320,
        max_images: Optional[int] = 800,
        train_frac: float = 0.8,
    ) -> None:
        del transform  # accepted for loader compatibility, intentionally unused
        self.root = root
        self.year = str(year)
        self.coco_split = coco_split
        self.image_size = int(image_size)
        self.split = str(split).lower()

        try:
            self._img_dir, self._ann_file = self._resolve_layout(
                root, self.coco_split, self.year,
            )
        except FileNotFoundError:
            if download:
                raise FileNotFoundError(
                    "COCO not found and there is no torchvision auto-download "
                    "for it. Fetch the ~1 GB subset on the login node first:\n"
                    "  python -m neuroquant.data.coco_detection --download "
                    f"--root {root}\n"
                )
            raise

        # Parse the annotation JSON once with the stdlib (no pycocotools).
        with open(self._ann_file, "r") as f:
            coco = json.load(f)

        id_to_file: Dict[int, str] = {
            int(im["id"]): im["file_name"] for im in coco["images"]
        }
        # image_id -> list of (xyxy box, category_id). ``iscrowd`` regions
        # are excluded — they are not scored as individual instances.
        anns_by_img: Dict[int, List[Tuple[List[float], int]]] = {}
        for ann in coco.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            box = [float(x), float(y), float(x + w), float(y + h)]
            anns_by_img.setdefault(int(ann["image_id"]), []).append(
                (box, int(ann["category_id"]))
            )

        # Only keep images that actually have at least one annotation so the
        # mAP denominator (GT count) is non-trivial for every sampled image.
        annotated_ids = sorted(i for i in id_to_file if anns_by_img.get(i))
        if max_images is not None:
            annotated_ids = annotated_ids[: int(max_images)]
        if not annotated_ids:
            raise RuntimeError(
                f"No annotated COCO images found under '{self._img_dir}'."
            )

        # Deterministic disjoint partition: first ``train_frac`` → train,
        # remainder → test/val. Keeps the headline test slice unseen.
        n_train = max(1, int(len(annotated_ids) * float(train_frac)))
        if self.split in ("test", "val", "valid"):
            self._ids = annotated_ids[n_train:] or annotated_ids[-max(1, len(annotated_ids) // 5):]
        else:
            self._ids = annotated_ids[:n_train]

        self._anns = anns_by_img
        self._files = id_to_file

        logger.info(
            "CocoDetection: split=%s, %d images (of %d annotated), size=%dx%d, "
            "src=%s",
            self.split, len(self._ids), len(annotated_ids),
            self.image_size, self.image_size, self._img_dir,
        )

    # ------------------------------------------------------------------
    # Layout resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_layout(root: str, coco_split: str, year: str) -> Tuple[Path, Path]:
        """Locate ``<split><year>/`` images and ``instances_*`` JSON."""
        base = Path(root).expanduser()
        img_name = f"{coco_split}{year}"
        ann_name = f"instances_{coco_split}{year}.json"
        img_candidates = [base / img_name, base / "images" / img_name, base]
        ann_candidates = [
            base / "annotations" / ann_name,
            base / ann_name,
        ]
        img_dir = next((c for c in img_candidates if c.is_dir()), None)
        ann_file = next((c for c in ann_candidates if c.is_file()), None)
        if img_dir is None or ann_file is None:
            raise FileNotFoundError(
                f"Could not find COCO {img_name} layout under '{root}'. "
                f"Expected images at one of {[str(c) for c in img_candidates]} "
                f"and annotations at one of {[str(c) for c in ann_candidates]}."
            )
        return img_dir, ann_file

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self._ids[int(idx)]
        path = self._img_dir / self._files[img_id]
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size

        s = self.image_size
        img = img.resize((s, s), Image.BILINEAR)
        # [0, 1] CHW float — NO normalisation (the detector does it).
        img_t = torch.from_numpy(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        sx = s / max(orig_w, 1)
        sy = s / max(orig_h, 1)
        boxes: List[List[float]] = []
        labels: List[int] = []
        for (x1, y1, x2, y2), cat_id in self._anns.get(img_id, []):
            nx1, nx2 = x1 * sx, x2 * sx
            ny1, ny2 = y1 * sy, y2 * sy
            # Clamp to the resized canvas and drop degenerate boxes.
            nx1, nx2 = max(0.0, min(nx1, s)), max(0.0, min(nx2, s))
            ny1, ny2 = max(0.0, min(ny1, s)), max(0.0, min(ny2, s))
            if nx2 - nx1 < 1.0 or ny2 - ny1 < 1.0:
                continue
            boxes.append([nx1, ny1, nx2, ny2])
            labels.append(int(cat_id))

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([int(img_id)]),
        }
        return img_t, target

    # torchvision datasets expose ``.classes``; provide a label-indexed
    # list so XAI caption lookups have sensible names.
    classes = _build_class_list(91)


def _download_coco_subset(root: str, max_images: int = 800) -> None:
    """Download COCO val2017 images + instances annotations to ``root``.

    Pulls the official annotation zip (~240 MB) and only the image files
    referenced by the first ``max_images`` annotated ids (~a few hundred
    MB), so the footprint stays well under 1 GB. Run on the login node.
    """
    import io
    import urllib.request
    import zipfile

    base = Path(root).expanduser()
    ann_dir = base / "annotations"
    img_dir = base / "val2017"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    ann_json = ann_dir / "instances_val2017.json"
    if not ann_json.is_file():
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        logger.info("Downloading COCO annotations (~240 MB) from %s ...", ann_url)
        with urllib.request.urlopen(ann_url) as resp:  # noqa: S310
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("instances_val2017.json"):
                    with zf.open(member) as src, open(ann_json, "wb") as dst:
                        dst.write(src.read())
        logger.info("Saved %s", ann_json)

    with open(ann_json, "r") as f:
        coco = json.load(f)
    id_to_file = {int(im["id"]): im["file_name"] for im in coco["images"]}
    have_ann = {int(a["image_id"]) for a in coco.get("annotations", [])
                if int(a.get("iscrowd", 0)) == 0}
    wanted = sorted(have_ann)[: int(max_images)]

    base_url = "http://images.cocodataset.org/val2017/"
    for n, img_id in enumerate(wanted, 1):
        fname = id_to_file[img_id]
        dst = img_dir / fname
        if dst.is_file():
            continue
        urllib.request.urlretrieve(base_url + fname, dst)  # noqa: S310
        if n % 50 == 0:
            logger.info("  downloaded %d/%d images ...", n, len(wanted))
    logger.info("COCO subset ready: %d images in %s", len(wanted), img_dir)


def _cli() -> None:
    """``python -m neuroquant.data.coco_detection --download --root ./data/coco``."""
    import argparse

    ap = argparse.ArgumentParser(description="Download / verify a COCO val2017 subset.")
    ap.add_argument("--root", default="./data/coco", help="Destination root dir.")
    ap.add_argument("--download", action="store_true", help="Fetch if missing.")
    ap.add_argument("--max-images", type=int, default=800,
                    help="Number of annotated images to download.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.download:
        _download_coco_subset(args.root, max_images=args.max_images)

    ds = CocoDetectionDataset(
        root=args.root, split="test", max_images=args.max_images,
    )
    img, target = ds[0]
    print(
        f"OK — COCO ready: {len(ds)} test images, "
        f"image{tuple(img.shape)} (range [{img.min():.2f},{img.max():.2f}]), "
        f"first target: {target['boxes'].shape[0]} boxes, "
        f"labels={target['labels'][:6].tolist()}"
    )


if __name__ == "__main__":
    _cli()

"""Focused genericity regression tests that avoid downloads."""

from __future__ import annotations

import sys

import torch
from torch.utils.data import Dataset


class CustomToyDataset(Dataset):
    """Small split-aware dataset used by dataset.class tests."""

    classes = ["zero", "one", "two"]

    def __init__(self, root: str = "./data", split: str = "train", transform=None):
        del root, transform
        n = 50 if split == "train" else 12
        gen = torch.Generator().manual_seed(100 if split == "train" else 200)
        self.images = torch.randn(n, 3, 16, 16, generator=gen)
        self.labels = torch.randint(0, len(self.classes), (n,), generator=gen)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def test_custom_dataset_class_path_loads_standard_loaders():
    from config import QuantizationConfig
    from data.data_loader import GenericDatasetLoader

    cfg = QuantizationConfig()
    cfg.dataset_name = "custom"
    cfg.dataset_class = "test_genericity_config.CustomToyDataset"
    cfg.input_shape = (3, 16, 16)
    cfg.num_classes = 3
    cfg.batch_size = 4
    cfg.num_workers = 0

    loader = GenericDatasetLoader(cfg)

    assert len(loader.get_train_loader().dataset) == 40
    assert len(loader.get_search_loader().dataset) == 5
    assert len(loader.get_val_loader().dataset) == 5
    assert len(loader.get_test_loader().dataset) == 12
    assert loader.get_class_names() == CustomToyDataset.classes


def test_synthetic_split_seed_comes_from_config():
    from config import QuantizationConfig
    from data.data_loader import GenericDatasetLoader

    cfg_a = QuantizationConfig()
    cfg_a.dataset_name = "synthetic"
    cfg_a.hyperparams.seed = 1
    cfg_b = QuantizationConfig()
    cfg_b.dataset_name = "synthetic"
    cfg_b.hyperparams.seed = 2

    first = GenericDatasetLoader(cfg_a)._train_dataset.indices[:10]
    second = GenericDatasetLoader(cfg_b)._train_dataset.indices[:10]

    assert first != second


def test_loader_respects_num_workers_off_windows(monkeypatch):
    from config import QuantizationConfig
    import data.data_loader as data_loader

    monkeypatch.setattr(data_loader.sys, "platform", "linux")
    cfg = QuantizationConfig()
    cfg.dataset_name = "synthetic"
    cfg.num_workers = 2

    loader = data_loader.GenericDatasetLoader(cfg)

    assert loader.num_workers == 2


def test_cli_seed_and_device_defaults_do_not_override_config(monkeypatch):
    from main import parse_args

    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = parse_args()

    assert args.seed is None
    assert args.device is None


def test_dataset_config_roundtrip_preserves_generic_fields(tmp_path):
    from config import QuantizationConfig

    cfg = QuantizationConfig()
    cfg.dataset_class = "test_genericity_config.CustomToyDataset"
    cfg.dataset_train_dir = "images/train"
    cfg.dataset_val_dir = "images/val"
    cfg.dataset_test_dir = "images/test"

    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = QuantizationConfig.from_yaml(path)

    assert loaded.dataset_class == cfg.dataset_class
    assert loaded.dataset_train_dir == cfg.dataset_train_dir
    assert loaded.dataset_val_dir == cfg.dataset_val_dir
    assert loaded.dataset_test_dir == cfg.dataset_test_dir

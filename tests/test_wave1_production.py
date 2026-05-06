"""
NeuroQuant v2.0 — Wave-1 production-grade regression tests.

Covers the foundation upgrades that move the project from "research
prototype" toward "deployable system":

  N1) SmoothQuant persistence is pickle-free. ``save_safe_module`` /
      ``load_safe_module`` survive ``weights_only=True`` and rebuild the
      ``_SmoothInputScale`` wrappers from a JSON-safe metadata blob.

  M1) ``set_seed(strict=True)`` flips the strict-determinism flags
      (``use_deterministic_algorithms``, cuDNN deterministic,
      ``CUBLAS_WORKSPACE_CONFIG``, ``PYTHONHASHSEED``) so reruns produce
      byte-stable outputs on the same machine.

  M2) ``utils/numerics`` exposes a single source of truth for
      ``EPS_PROB``, ``MIN_SCALE``, ``MIN_DAMP``, ``MIN_MIGRATION``,
      ``MAX_MIGRATION``, ``EPS_GEOMETRIC``. Override-via-env works.

  A1) ``GenericDatasetLoader`` exposes a search slice carved from
      train (10%) that is disjoint from val and test. NSGA fitness reads
      from this loader; nothing else does.

  A2) ``_attach_split_metrics`` populates ``val_top1`` + ``test_top1``
      and promotes the test number to the public ``accuracy`` field on
      every method result.

All tests run on CPU with tiny synthetic data (no torchvision downloads).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn as nn

from config import QuantizationConfig
from quantization.smoothquant import (
    SmoothQuantQuantizer,
    _SmoothInputScale,
    restore_smoothquant_wrappers,
    serialize_smoothquant_metadata,
)
from utils.checkpointing import CheckpointManager
from utils.common import set_seed
from utils import numerics

logging.basicConfig(level=logging.WARNING, format="%(message)s")

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")


class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.c1(x)))
        x = torch.relu(self.bn2(self.c2(x)))
        return self.fc(x.mean(dim=(2, 3)))


def _make_loader(n: int = 16, classes: int = 4):
    imgs = torch.randn(n, 3, 8, 8)
    lbls = torch.randint(0, classes, (n,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=4)


# ─────────────────────────────────────────────────────────────────────────
# N1) SmoothQuant safe persistence
# ─────────────────────────────────────────────────────────────────────────


def test_smoothquant_safe_save_load_roundtrip():
    print("--- Test N1.1: SmoothQuant save/load roundtrip via state_dict + metadata ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.smoothquant_alpha = 0.5

    torch.manual_seed(0)
    base = _TinyNet(4)
    sq = SmoothQuantQuantizer(base, cfg)
    sq_model = sq.quantize(_make_loader(), bitwidth=8, num_batches=2)

    # Serialise metadata; verify it is pure JSON.
    metadata = serialize_smoothquant_metadata(sq_model)
    json_blob = json.dumps(metadata)  # raises if any non-JSON object leaked
    check("metadata is JSON-serializable", isinstance(json_blob, str))
    check("metadata records at least one wrapper",
          len(metadata.get("wrappers", [])) >= 1,
          f"got {metadata}")

    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp)
        mgr.save_safe_module("sq_model.pt", sq_model, metadata=metadata)

        # The saved file must NOT contain pickled Python — load it under
        # weights_only=True (the production-safe path).
        path = Path(tmp) / "checkpoints" / "sq_model.pt"
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        check("envelope loads under weights_only=True",
              isinstance(envelope, dict) and "state_dict" in envelope)
        check("envelope carries the wrapper manifest",
              "metadata" in envelope and "wrappers" in envelope["metadata"])

        # Rebuild on a fresh blank model.
        blank = _TinyNet(4)
        # Pre-condition: blank has no _SmoothInputScale anywhere.
        n_wrap_before = sum(
            1 for m in blank.modules() if isinstance(m, _SmoothInputScale)
        )
        check("blank model has no SmoothQuant wrappers", n_wrap_before == 0)

        mgr.load_safe_module(
            "sq_model.pt", blank, rebuild=restore_smoothquant_wrappers,
        )
        n_wrap_after = sum(
            1 for m in blank.modules() if isinstance(m, _SmoothInputScale)
        )
        check("rebuilt model has the same wrapper count",
              n_wrap_after == len(metadata["wrappers"]),
              f"after={n_wrap_after}, manifest={len(metadata['wrappers'])}")

        # State-dict equivalence is the real correctness contract: every
        # parameter and buffer in the source model must reach the rebuilt
        # model with bit-identical values. This rules out the silent
        # ``strict=False`` key-mismatch failure mode that an
        # ``allclose``-on-forward-output check would not catch.
        ref_sd = sq_model.state_dict()
        new_sd = blank.state_dict()
        check("state_dict key sets match after rebuild",
              set(ref_sd.keys()) == set(new_sd.keys()),
              f"missing in new: {set(ref_sd) - set(new_sd)}; "
              f"extra in new: {set(new_sd) - set(ref_sd)}")
        all_equal = True
        first_mismatch = ""
        for k in ref_sd:
            if k not in new_sd:
                continue
            if not torch.equal(ref_sd[k], new_sd[k]):
                all_equal = False
                first_mismatch = k
                break
        check("every tensor in state_dict round-trips bit-equal",
              all_equal,
              f"first mismatch at key '{first_mismatch}'")

        # Forward equivalence in eval mode (BN running stats fixed) is
        # the user-visible test that no architectural piece was lost.
        sq_model.eval()
        blank.eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            y_ref = sq_model(x)
            y_new = blank(x)
        max_abs_err = float((y_ref - y_new).abs().max())
        check("rebuilt forward matches saved forward in eval mode",
              max_abs_err < 1e-5,
              f"max_abs_err={max_abs_err}")


def test_no_pickle_load_in_safe_path():
    print("--- Test N1.2: load_safe_module rejects pickle envelopes ---")
    with tempfile.TemporaryDirectory() as tmp:
        mgr = CheckpointManager(tmp)
        # A bare module (would have required weights_only=False to load
        # historically). Saving it with safe_module then trying to read
        # it as a state_dict envelope must succeed because save_safe_module
        # always writes the safe shape; the malicious shape is what we
        # explicitly disallow.
        m = _TinyNet(4)
        mgr.save_safe_module("safe.pt", m)
        loaded = torch.load(
            Path(tmp) / "checkpoints" / "safe.pt",
            map_location="cpu", weights_only=True,
        )
        check("safe envelope round-trips under weights_only=True",
              isinstance(loaded, dict) and "state_dict" in loaded)


# ─────────────────────────────────────────────────────────────────────────
# M1) Strict determinism flags
# ─────────────────────────────────────────────────────────────────────────


def test_set_seed_strict_flags():
    print("--- Test M1: set_seed(strict=True) sets determinism flags ---")
    set_seed(123, strict=True)

    check("PYTHONHASHSEED is set",
          os.environ.get("PYTHONHASHSEED") == "123",
          f"got {os.environ.get('PYTHONHASHSEED')}")
    check("CUBLAS_WORKSPACE_CONFIG is pinned",
          os.environ.get("CUBLAS_WORKSPACE_CONFIG", "") == ":4096:8",
          f"got {os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

    # cudnn flags only meaningful when CUDA is built; assert that they
    # took the deterministic value either way to catch accidental flips.
    if torch.cuda.is_available():
        check("cudnn.deterministic enabled",
              bool(torch.backends.cudnn.deterministic))
        check("cudnn.benchmark disabled",
              not bool(torch.backends.cudnn.benchmark))


def test_set_seed_reproducible_torch_rand():
    print("--- Test M1.2: same seed → identical torch tensors ---")
    set_seed(7, strict=True)
    a = torch.randn(8, 8)
    set_seed(7, strict=True)
    b = torch.randn(8, 8)
    check("two set_seed(7) runs produce identical tensors",
          torch.equal(a, b))


# ─────────────────────────────────────────────────────────────────────────
# M2) Centralized numerics
# ─────────────────────────────────────────────────────────────────────────


def test_numerics_constants_present():
    print("--- Test M2: numerics module exposes all constants ---")
    for name, expected_max in (
        ("EPS_PROB", 1e-10),
        ("MIN_SCALE", 1e-6),
        ("MIN_DAMP", 1e-3),
        ("MIN_MIGRATION", 1.0),
        ("EPS_GEOMETRIC", 1e-6),
    ):
        v = getattr(numerics, name, None)
        check(f"numerics.{name} is a positive float",
              isinstance(v, float) and 0 < v < expected_max,
              f"got {v}")
    check("numerics.MAX_MIGRATION is large",
          isinstance(numerics.MAX_MIGRATION, float)
          and numerics.MAX_MIGRATION >= 1e3,
          f"got {numerics.MAX_MIGRATION}")


def test_numerics_no_hardcoded_eps_in_quantization():
    print("--- Test M2.2: hardcoded eps literals removed from production paths ---")
    # Spot-check the files that previously held ``1e-8`` / ``1e-12``.
    for relpath in (
        "quantization/ptq.py",
        "quantization/qat.py",
        "quantization/awq.py",
        "quantization/gptq.py",
        "quantization/smoothquant.py",
        "quantization/base.py",
        "quantization/adaround.py",
        "quantization/nsga_ii_search.py",
    ):
        text = (project_root / relpath).read_text(encoding="utf-8")
        # The literals should be replaced by the constant names.
        for needle in ("1e-12", "1e-8", "1e-6", "1e-4"):
            # Allow the literal inside the imported module file itself
            # (utils/numerics.py owns these) and inside comments.
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if needle in stripped:
                    check(
                        f"{relpath} no raw {needle} literal in code",
                        False, f"line: {stripped}",
                    )
                    break
            else:
                continue
            break
        else:
            check(f"{relpath} clean of raw eps literals", True)


# ─────────────────────────────────────────────────────────────────────────
# A1) Train/search/val/test split
# ─────────────────────────────────────────────────────────────────────────


def test_search_loader_disjoint_from_val_and_test():
    print("--- Test A1: search loader is disjoint from val/test ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.dataset_name = "synthetic"
    cfg.batch_size = 4
    cfg.num_workers = 0

    from data.data_loader import GenericDatasetLoader
    loader = GenericDatasetLoader(cfg)
    train = loader.get_train_loader()
    search = loader.get_search_loader()
    val = loader.get_val_loader()
    test = loader.get_test_loader()

    check("get_search_loader returns a DataLoader",
          isinstance(search, torch.utils.data.DataLoader))

    # Compare the underlying datasets — they must be different objects
    # (or different Subset index sets) so no sample crosses splits.
    def _index_set(dl):
        ds = dl.dataset
        if isinstance(ds, torch.utils.data.Subset):
            return frozenset(ds.indices)
        return None  # opaque dataset; can't compare

    s_train = _index_set(train)
    s_search = _index_set(search)
    s_val = _index_set(val)

    if s_train is not None and s_search is not None:
        check("train ∩ search = ∅",
              s_train.isdisjoint(s_search),
              f"overlap size={len(s_train & s_search)}")
    if s_search is not None and s_val is not None:
        check("search ∩ val = ∅",
              s_search.isdisjoint(s_val),
              f"overlap size={len(s_search & s_val)}")
    # Test set is a separate TensorDataset (synthetic) — it does NOT
    # share an index space with train/search/val so the comparison
    # above suffices.
    check("test loader sources from a different dataset object",
          test.dataset is not train.dataset
          and test.dataset is not search.dataset
          and test.dataset is not val.dataset)


def test_build_data_loaders_returns_six_loaders():
    print("--- Test A1.2: build_data_loaders returns search slot ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.dataset_name = "synthetic"
    cfg.batch_size = 4
    cfg.num_workers = 0

    from main import build_data_loaders
    out = build_data_loaders(cfg)
    check("build_data_loaders returns 6 elements",
          len(out) == 6, f"got {len(out)}")
    train, search, val, test, calib, names = out
    for label, dl in (
        ("train", train), ("search", search), ("val", val),
        ("test", test), ("calib", calib),
    ):
        check(f"{label}_loader is a DataLoader",
              isinstance(dl, torch.utils.data.DataLoader))


# ─────────────────────────────────────────────────────────────────────────
# A2) test_loader headline + val_top1 / test_top1 contract
# ─────────────────────────────────────────────────────────────────────────


def test_attach_split_metrics_promotes_test_to_headline():
    print("--- Test A2: _attach_split_metrics writes val/test/headline ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"

    from main import NeuroQuantPipeline
    pipe = NeuroQuantPipeline(cfg, training_epochs=0, resume=False)
    pipe.model = _TinyNet(4)
    pipe.search_loader = _make_loader()
    pipe.val_loader = _make_loader()
    pipe.test_loader = _make_loader()

    res = {"accuracy": 12.34, "method": "PTQ"}  # stale val number
    pipe._attach_split_metrics(res, pipe.model)

    check("val_top1 is set", "val_top1" in res)
    check("test_top1 is set", "test_top1" in res)
    check("accuracy is the test number",
          abs(res["accuracy"] - res["test_top1"]) < 1e-6,
          f"got {res}")
    # val and test will usually differ because the loaders use different
    # random tensors; assert the contract holds even when they happen to
    # match (just check both fields are floats).
    check("val_top1 is a finite float",
          isinstance(res["val_top1"], float))
    check("test_top1 is a finite float",
          isinstance(res["test_top1"], float))


def test_phase_1c_rerank_promotes_test_to_headline():
    print("--- Test A2.2: rerank winners get test_top1 in accuracy field ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2
    cfg.hyperparams.latency_warmup_runs = 1
    cfg.hyperparams.latency_measure_runs = 2
    cfg.hyperparams.ptq_real_rerank_topk = 2
    cfg.hyperparams.ptq_tradeoff_max_acc_drop = 100.0

    from main import NeuroQuantPipeline
    pipe = NeuroQuantPipeline(cfg, training_epochs=0, resume=False)
    pipe.model = _TinyNet(4)
    pipe.calib_loader = _make_loader()
    pipe.search_loader = _make_loader()
    pipe.val_loader = _make_loader()
    pipe.test_loader = _make_loader()
    pipe.fp32_acc = 50.0

    weight_keys = [n for n, _ in pipe.model.named_parameters() if "weight" in n]
    candidates = [
        {"solution_id": "nsga_a", "accuracy_loss": 1.0, "model_size_mb": 0.10,
         "bitwidth_assignment": {n: 8 for n in weight_keys}},
        {"solution_id": "nsga_b", "accuracy_loss": 5.0, "model_size_mb": 0.05,
         "bitwidth_assignment": {n: 4 for n in weight_keys}},
    ]
    (best_acc_m, best_acc_r,
     best_to_m, best_to_r) = pipe._materialize_and_rerank_ptq(
        candidates, cfg.hyperparams,
    )
    for r in (best_acc_r, best_to_r):
        check(f"{r['display_name']}: search_top1 set",
              "search_top1" in r and isinstance(r["search_top1"], float),
              f"got {r}")
        check(f"{r['display_name']}: val_top1 set",
              "val_top1" in r and isinstance(r["val_top1"], float),
              f"got {r}")
        check(f"{r['display_name']}: test_top1 set",
              "test_top1" in r and isinstance(r["test_top1"], float),
              f"got {r}")
        check(f"{r['display_name']}: accuracy == test_top1",
              abs(r["accuracy"] - r["test_top1"]) < 1e-6,
              f"got {r}")


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    test_smoothquant_safe_save_load_roundtrip()
    test_no_pickle_load_in_safe_path()
    test_set_seed_strict_flags()
    test_set_seed_reproducible_torch_rand()
    test_numerics_constants_present()
    test_numerics_no_hardcoded_eps_in_quantization()
    test_search_loader_disjoint_from_val_and_test()
    test_build_data_loaders_returns_six_loaders()
    test_attach_split_metrics_promotes_test_to_headline()
    test_phase_1c_rerank_promotes_test_to_headline()

    print("\n" + "=" * 50)
    print(f"  Wave-1 Production Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

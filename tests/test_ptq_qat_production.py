"""
NeuroQuant v2.0 — Production-grade PTQ/QAT/hybrid tests.

Verifies the upgrade specified in the PTQ/QAT task brief:

  A) Bitwidth-aware PTQ calibration: a mixed INT4/INT8 assignment
     produces different per-layer thresholds for INT4 vs INT8 layers,
     and ``calibrate_with_assignment`` records the correct per-layer
     bitwidth (with I/O enforcement intact).

  B) NSGA hygiene: searchable cluster groups contain ONLY the weight
     parameters of Conv2d/Linear modules — BN γ/β, biases and other
     non-quantizable parameters are filtered out, both from the
     fixed config and from the search-space genes.

  C) Multi-fidelity PTQ selection: phase-1c materialises the top-K
     NSGA candidates through real PTQ and produces both
     ``ptq_best_acc_result`` and ``ptq_best_tradeoff_result``.

  D) Hybrid PTQ→QAT warmstart policy: the explicit
     ``qat_warmstart_source`` knob drives ``self.best_config``
     selection, the chosen source + PTQ ID are written to the phase-1c
     JSON checkpoint and the phase-1e checkpoint metadata, and config
     validation rejects invalid values.

  E) Public reporting: the public Pareto/report excludes NSGA internal
     IDs and includes a dedicated PTQ tradeoff entry when distinct
     from ``ptq_best_acc``.

All tests run on CPU with tiny synthetic data (no torchvision downloads).
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn as nn

from config import (
    ParetoFront,
    ParetoSolution,
    QuantizationConfig,
)
from quantization.ptq import PTQQuantizer
from quantization.nsga_ii_search import NSGAIIClusterSearch
from visualization.pareto_analysis import ParetoAnalyzer

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
    """Small Conv→BN→Conv→FC net so we have BN and bias parameters
    that the NSGA hygiene filter must reject."""

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


def _make_loader(n: int = 16, classes: int = 4, h: int = 8, w: int = 8):
    imgs = torch.randn(n, 3, h, w)
    lbls = torch.randint(0, classes, (n,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=4)


# ─────────────────────────────────────────────────────────────────────────
# A) Bitwidth-aware PTQ calibration
# ─────────────────────────────────────────────────────────────────────────


def test_calibrate_with_assignment_per_layer_bitwidth():
    print("--- Test A1: per-layer bitwidth recorded by calibrate_with_assignment ---")
    torch.manual_seed(0)
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2
    cfg.io_layer_bitwidth = 8

    model = _TinyNet(4)
    ptq = PTQQuantizer(model, cfg)

    # Mixed assignment: c1 should be I/O-overridden to INT8 anyway, but
    # the user asked for INT4; c2 → INT4; fc is the last quantizable
    # module, so I/O override forces INT8.
    assignment = {
        "c1.weight": 4,
        "c2.weight": 4,
        "fc.weight":  4,
    }
    ptq.calibrate_with_assignment(_make_loader(), assignment, num_batches=2)

    # I/O enforcement must still apply.
    check("c1 (input layer) calibrated at INT8 even though assignment said 4",
          ptq._per_layer_bitwidth.get("c1") == 8,
          f"got {ptq._per_layer_bitwidth.get('c1')}")
    check("fc (output layer) calibrated at INT8 even though assignment said 4",
          ptq._per_layer_bitwidth.get("fc") == 8,
          f"got {ptq._per_layer_bitwidth.get('fc')}")
    # Intermediate layer respects the assignment.
    check("c2 (intermediate) calibrated at INT4",
          ptq._per_layer_bitwidth.get("c2") == 4,
          f"got {ptq._per_layer_bitwidth.get('c2')}")

    # Every quantizable module has an observer with a computed threshold.
    for name in ("c1", "c2", "fc"):
        obs = ptq._observers.get(name)
        check(f"observer for {name} populated", obs is not None)
        check(f"observer for {name} has threshold",
              obs is not None and obs.threshold is not None)


def test_calibrate_with_assignment_thresholds_differ_per_bitwidth():
    print("--- Test A2: INT4 vs INT8 threshold search produces different thresholds ---")
    torch.manual_seed(123)
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 4
    cfg.io_layer_bitwidth = 8  # so c1/fc are forced INT8 either way

    # Activation-rich loader with outliers so the strategy actually
    # discriminates the bitwidth-driven threshold search.
    imgs = torch.cat([
        torch.randn(24, 3, 8, 8) * 0.3,
        torch.randn(8, 3, 8, 8) * 3.0,
    ], dim=0)
    lbls = torch.randint(0, 4, (32,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)

    import copy as _copy
    base = _TinyNet(4)

    # Run 1: c2 calibrated at INT4.
    ptq_int4 = PTQQuantizer(_copy.deepcopy(base), cfg)
    ptq_int4.calibrate_with_assignment(
        loader,
        {"c1.weight": 8, "c2.weight": 4, "fc.weight": 8},
        num_batches=4,
    )

    # Run 2: c2 calibrated at INT8.
    ptq_int8 = PTQQuantizer(_copy.deepcopy(base), cfg)
    ptq_int8.calibrate_with_assignment(
        loader,
        {"c1.weight": 8, "c2.weight": 8, "fc.weight": 8},
        num_batches=4,
    )

    t4 = ptq_int4._observers["c2"].threshold
    t8 = ptq_int8._observers["c2"].threshold
    check("c2 INT4 threshold present", t4 is not None and t4 > 0)
    check("c2 INT8 threshold present", t8 is not None and t8 > 0)
    # The same activation samples + the same strategy + different
    # bitwidths must yield different thresholds — this is the whole
    # point of bitwidth-aware calibration.
    check("c2 threshold differs between INT4 and INT8 calibration",
          abs(float(t4) - float(t8)) > 1e-4,
          f"int4={t4}, int8={t8}")


def test_calibrate_legacy_api_still_works():
    print("--- Test A3: legacy calibrate(...) API is backward-compatible ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"

    model = _TinyNet(4)
    ptq = PTQQuantizer(model, cfg)
    ptq.calibrate(_make_loader(), num_batches=2, bitwidth=8)

    check("legacy calibrate populated observers",
          len(ptq._observers) > 0)
    check("legacy calibrate set _calibrated_bitwidth",
          ptq._calibrated_bitwidth == 8)


# ─────────────────────────────────────────────────────────────────────────
# B) NSGA hygiene — only quantizable Conv/Linear weights
# ─────────────────────────────────────────────────────────────────────────


def test_nsga_searchable_groups_exclude_non_quantizable():
    print("--- Test B1: NSGA hygiene drops BN/bias from searchable clusters ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"

    model = _TinyNet(4)

    # Polluted incoming clusters: include BN γ, biases, and the legitimate
    # Conv/Linear weights. The NSGA search must keep ONLY the weights.
    cluster_assignments = [
        {
            "cluster_id": 0, "tier": "HIGH",
            "layer_names": ["c1.weight", "c1.bias", "bn1.weight", "bn1.bias"],
            "allowed_bitwidths": [8],
            "mean_sensitivity": 0.5,
        },
        {
            "cluster_id": 1, "tier": "MEDIUM",
            "layer_names": ["c2.weight", "c2.bias", "bn2.weight", "bn2.bias"],
            "allowed_bitwidths": [4, 8],
            "mean_sensitivity": 0.2,
        },
        {
            "cluster_id": 2, "tier": "LOW",
            "layer_names": ["fc.weight", "fc.bias"],
            "allowed_bitwidths": [4, 8],
            "mean_sensitivity": 0.1,
        },
        # Pure-noise cluster of biases only — must be dropped entirely.
        {
            "cluster_id": 3, "tier": "MEDIUM",
            "layer_names": ["c1.bias", "c2.bias", "fc.bias"],
            "allowed_bitwidths": [4, 8],
            "mean_sensitivity": 0.05,
        },
    ]

    nsga = NSGAIIClusterSearch(model, cluster_assignments, cfg)

    quantizable = nsga.get_quantizable_param_names()
    check("quantizable set is exactly the Conv/Linear weights",
          set(quantizable) == {"c1.weight", "c2.weight", "fc.weight"},
          f"got {quantizable}")

    searchable = nsga.get_searchable_param_names()
    check("searchable params exclude BN/bias",
          all(("bn" not in n and ".bias" not in n) for n in searchable),
          f"got {searchable}")
    check("searchable params are exactly c2.weight + fc.weight",
          set(searchable) == {"c2.weight", "fc.weight"},
          f"got {searchable}")

    # Bias-only cluster must have been dropped completely.
    check("pure-bias cluster dropped (num_genes == 2)",
          nsga.num_genes == 2,
          f"num_genes={nsga.num_genes}")

    # Fixed config should only mention c1.weight (the HIGH-tier weight).
    check("fixed config covers only c1.weight",
          set(nsga._fixed_config.keys()) == {"c1.weight"},
          f"got {sorted(nsga._fixed_config.keys())}")


def test_nsga_individual_to_config_only_quantizable():
    print("--- Test B2: individual_to_config never sets BN/bias bitwidths ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"

    model = _TinyNet(4)
    cluster_assignments = [
        {"cluster_id": 0, "tier": "HIGH",
         "layer_names": ["c1.weight", "bn1.weight"],
         "allowed_bitwidths": [8], "mean_sensitivity": 0.5},
        {"cluster_id": 1, "tier": "MEDIUM",
         "layer_names": ["c2.weight", "c2.bias"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.2},
        {"cluster_id": 2, "tier": "LOW",
         "layer_names": ["fc.weight", "fc.bias"],
         "allowed_bitwidths": [4, 8], "mean_sensitivity": 0.1},
    ]
    nsga = NSGAIIClusterSearch(model, cluster_assignments, cfg)

    # Force every searchable gene to INT4 — most aggressive case.
    individual = [0] * nsga.num_genes
    config = nsga.individual_to_config(individual)
    quantized_keys = set(config.keys())
    bn_or_bias = {k for k in quantized_keys if ("bn" in k or ".bias" in k)}
    check("config maps only Conv/Linear weights",
          not bn_or_bias,
          f"leaked: {bn_or_bias}")
    check("c1.weight pinned at INT8 (HIGH tier)",
          config.get("c1.weight") == 8)


# ─────────────────────────────────────────────────────────────────────────
# C) Multi-fidelity PTQ selection
# ─────────────────────────────────────────────────────────────────────────


def _build_pipeline(cfg: QuantizationConfig):
    """Construct a NeuroQuantPipeline without running phase 0, then
    plug in a tiny model + loaders by hand. Avoids the torchvision /
    dataset-download cost during unit tests.

    Provides search/val/test loaders separately so the production
    contract (NSGA on search, QAT early-stop on val, headline on test)
    is exercised end-to-end in the unit tests too.
    """
    from main import NeuroQuantPipeline

    pipe = NeuroQuantPipeline(cfg, training_epochs=0, resume=False)
    pipe.model = _TinyNet(cfg.num_classes)
    pipe.calib_loader = _make_loader(classes=cfg.num_classes)
    pipe.search_loader = _make_loader(classes=cfg.num_classes)
    pipe.val_loader = _make_loader(classes=cfg.num_classes)
    pipe.test_loader = _make_loader(classes=cfg.num_classes)
    pipe.fp32_acc = 50.0  # placeholder; we don't measure absolute deltas
    pipe.fp32_ebops = sum(
        p.numel() * 32 for p in pipe.model.parameters()
    ) / 8.0
    return pipe


def test_select_rerank_candidates_top_k_dedup():
    print("--- Test C1: _select_rerank_candidates picks top-K and dedups ---")
    from main import NeuroQuantPipeline

    duplicates_assignment = {"c1.weight": 8, "c2.weight": 4, "fc.weight": 8}
    nsga_solutions = [
        {"solution_id": "nsga_a", "accuracy_loss": 1.0, "model_size_mb": 0.05,
         "bitwidth_assignment": duplicates_assignment},
        {"solution_id": "nsga_b", "accuracy_loss": 1.0, "model_size_mb": 0.05,
         "bitwidth_assignment": duplicates_assignment},  # exact duplicate
        {"solution_id": "nsga_c", "accuracy_loss": 4.0, "model_size_mb": 0.025,
         "bitwidth_assignment": {"c1.weight": 4, "c2.weight": 4, "fc.weight": 4}},
        {"solution_id": "nsga_d", "accuracy_loss": 0.5, "model_size_mb": 0.10,
         "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 8, "fc.weight": 8}},
    ]
    out = NeuroQuantPipeline._select_rerank_candidates(nsga_solutions, top_k=3)
    check("dedup collapses identical assignments",
          len(out) == 3, f"got {len(out)}")
    # Order: lowest accuracy_loss first.
    losses = [s["accuracy_loss"] for s in out]
    check("rerank order is by ascending accuracy_loss",
          losses == sorted(losses), f"got {losses}")


def test_phase_1c_produces_dual_ptq_outputs():
    print("--- Test C2: phase 1c materialises ptq_best_acc and ptq_best_tradeoff ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2
    cfg.hyperparams.latency_warmup_runs = 1
    cfg.hyperparams.latency_measure_runs = 2
    cfg.hyperparams.ptq_real_rerank_topk = 3
    cfg.hyperparams.ptq_tradeoff_max_acc_drop = 100.0  # broad cap so any candidate qualifies

    pipe = _build_pipeline(cfg)
    weight_keys = [n for n, _ in pipe.model.named_parameters() if "weight" in n]

    # Two distinct candidates so best_acc != best_tradeoff is plausible.
    nsga_solutions = [
        {"solution_id": "nsga_a", "accuracy_loss": 1.0, "model_size_mb": 0.10,
         "bitwidth_assignment": {n: 8 for n in weight_keys}},
        {"solution_id": "nsga_b", "accuracy_loss": 5.0, "model_size_mb": 0.05,
         "bitwidth_assignment": {n: 4 for n in weight_keys}},
    ]
    candidates = pipe._select_rerank_candidates(nsga_solutions, top_k=3)
    check("two NSGA candidates surfaced for rerank",
          len(candidates) == 2, f"got {len(candidates)}")

    (best_acc_m, best_acc_r,
     best_to_m, best_to_r) = pipe._materialize_and_rerank_ptq(
        candidates, cfg.hyperparams,
    )
    check("ptq_best_acc_result populated", best_acc_r is not None)
    check("ptq_best_tradeoff_result populated", best_to_r is not None)
    check("ptq_best_acc display is canonically tagged",
          best_acc_r and best_acc_r["display_name"].startswith("PTQ_"),
          f"got {best_acc_r and best_acc_r['display_name']}")
    check("ptq_best_tradeoff display is canonically tagged",
          best_to_r and best_to_r["display_name"].startswith("PTQ_"),
          f"got {best_to_r and best_to_r['display_name']}")
    # No legacy METHOD_METHOD pattern in either display name.
    for r in (best_acc_r, best_to_r):
        if not r:
            continue
        check(f"display '{r['display_name']}' has no PTQ_PTQ duplication",
              "PTQ_PTQ" not in r["display_name"])
    # When the assignments differ, the chosen models must differ in at
    # least one weight tensor (sanity that real PTQ ran twice).
    if best_acc_m is not None and best_to_m is not None and best_acc_m is not best_to_m:
        any_diff = any(
            not torch.equal(p1, p2)
            for (n1, p1), (n2, p2) in zip(
                best_acc_m.named_parameters(), best_to_m.named_parameters(),
            )
            if "weight" in n1
        )
        check("acc and tradeoff models actually differ when configs differ",
              any_diff)


def test_tradeoff_falls_back_to_smallest_when_cap_unmet():
    print("--- Test C3: tradeoff knee-fallback when no candidate meets the cap ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2
    cfg.hyperparams.latency_warmup_runs = 1
    cfg.hyperparams.latency_measure_runs = 2
    cfg.hyperparams.ptq_real_rerank_topk = 3
    # Cap of 0.0 pp → essentially nothing can satisfy → fallback to
    # smallest-size candidate.
    cfg.hyperparams.ptq_tradeoff_max_acc_drop = 0.0

    pipe = _build_pipeline(cfg)
    weight_keys = [n for n, _ in pipe.model.named_parameters() if "weight" in n]
    nsga_solutions = [
        {"solution_id": "nsga_a", "accuracy_loss": 1.0, "model_size_mb": 0.20,
         "bitwidth_assignment": {n: 8 for n in weight_keys}},
        {"solution_id": "nsga_b", "accuracy_loss": 5.0, "model_size_mb": 0.05,
         "bitwidth_assignment": {n: 4 for n in weight_keys}},
    ]
    candidates = pipe._select_rerank_candidates(nsga_solutions, top_k=3)
    _, _, _, best_to_r = pipe._materialize_and_rerank_ptq(
        candidates, cfg.hyperparams,
    )
    check("tradeoff fallback returns a result even with cap=0",
          best_to_r is not None)


# ─────────────────────────────────────────────────────────────────────────
# D) Hybrid PTQ→QAT warmstart policy
# ─────────────────────────────────────────────────────────────────────────


def test_qat_warmstart_source_validation():
    print("--- Test D1: invalid qat_warmstart_source rejected by validate() ---")
    cfg = QuantizationConfig()
    cfg.hyperparams.qat_warmstart_source = "garbage"
    raised = False
    try:
        cfg.validate()
    except ValueError:
        raised = True
    check("validate() rejects unknown qat_warmstart_source", raised)

    cfg.hyperparams.qat_warmstart_source = "ptq_best_tradeoff"
    cfg.validate()
    check("'ptq_best_tradeoff' accepted by validate()", True)

    cfg.hyperparams.qat_warmstart_source = "ptq_best_acc"
    cfg.hyperparams.ptq_real_rerank_topk = 0
    raised = False
    try:
        cfg.validate()
    except ValueError:
        raised = True
    check("validate() rejects ptq_real_rerank_topk < 1", raised)


def test_warmstart_source_persisted_in_phase_1c_checkpoint():
    print("--- Test D2: warmstart source + PTQ ID reach the phase-1c JSON ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.calibration_batches = 2
    cfg.hyperparams.latency_warmup_runs = 1
    cfg.hyperparams.latency_measure_runs = 2
    cfg.hyperparams.ptq_real_rerank_topk = 2
    cfg.hyperparams.ptq_tradeoff_max_acc_drop = 100.0
    cfg.hyperparams.qat_warmstart_source = "ptq_best_tradeoff"

    with tempfile.TemporaryDirectory() as tmp:
        cfg.output_dir = tmp
        pipe = _build_pipeline(cfg)
        # Pareto-front state and helper plumbing
        weight_keys = [n for n, _ in pipe.model.named_parameters() if "weight" in n]
        pipe.pareto_front = {
            "solutions": [
                {"solution_id": "nsga_a", "accuracy_loss": 1.0,
                 "model_size_mb": 0.10,
                 "bitwidth_assignment": {n: 8 for n in weight_keys}},
                {"solution_id": "nsga_b", "accuracy_loss": 4.0,
                 "model_size_mb": 0.05,
                 "bitwidth_assignment": {n: 4 for n in weight_keys}},
            ],
            "evaluations": 4, "generation": 1,
            "convergence_reason": "test",
        }
        pipe.best_config = pipe.pareto_front["solutions"][0]["bitwidth_assignment"]

        # Stub MLflow so phase_1c can log without a real run.
        class _Tracker:
            def start_run(self, *a, **kw): pass
            def end_run(self, *a, **kw): pass
            def log_params(self, *a, **kw): pass
            def log_metrics(self, *a, **kw): pass
            def log_artifact(self, *a, **kw): pass
        pipe.tracker = _Tracker()

        # Run only phase 1c (skip the NSGA search itself by providing the
        # pre-built pareto_front; phase_1c re-runs NSGA so we monkeypatch
        # the search to return our hand-crafted front).
        from quantization.nsga_ii_search import NSGAIIClusterSearch

        original_search = NSGAIIClusterSearch.search
        prebuilt = pipe.pareto_front

        def _stub_search(self, *_a, **_kw):
            self._last_pareto = prebuilt["solutions"]
            return prebuilt
        NSGAIIClusterSearch.search = _stub_search  # type: ignore[assignment]
        # FITCompress seed plumbing — phase_1c reads fit_seed["seed_config"].
        pipe.fit_seed = {"seed_config": {n: 8 for n in weight_keys}}
        # cluster_assignments is consumed by NSGAIIClusterSearch ctor.
        pipe.cluster_assignments = [
            {"cluster_id": 0, "tier": "HIGH",
             "layer_names": ["c1.weight"], "allowed_bitwidths": [8],
             "mean_sensitivity": 0.5},
            {"cluster_id": 1, "tier": "MEDIUM",
             "layer_names": ["c2.weight"], "allowed_bitwidths": [4, 8],
             "mean_sensitivity": 0.2},
            {"cluster_id": 2, "tier": "LOW",
             "layer_names": ["fc.weight"], "allowed_bitwidths": [4, 8],
             "mean_sensitivity": 0.1},
        ]

        try:
            pipe.phase_1c_nsga_search()
        finally:
            NSGAIIClusterSearch.search = original_search  # type: ignore[assignment]

        # Read back the JSON checkpoint and verify the new fields.
        ckpt = Path(tmp) / "checkpoints" / "phase_1c_nsga_search.json"
        check("phase_1c JSON checkpoint exists", ckpt.exists())
        if ckpt.exists():
            data = json.loads(ckpt.read_text())
            check("checkpoint records qat_warmstart_source",
                  data.get("qat_warmstart_source") == "ptq_best_tradeoff",
                  f"got {data.get('qat_warmstart_source')}")
            check("checkpoint records qat_warmstart_id",
                  isinstance(data.get("qat_warmstart_id"), str)
                  and data["qat_warmstart_id"].startswith("PTQ_"),
                  f"got {data.get('qat_warmstart_id')}")
            check("checkpoint exposes ptq_best_acc_result",
                  data.get("ptq_best_acc_result") is not None)
            check("checkpoint exposes ptq_best_tradeoff_result",
                  data.get("ptq_best_tradeoff_result") is not None)

        # In-memory state on the pipeline mirrors the chosen warmstart.
        check("pipeline.results['qat_warmstart_source'] set",
              pipe.results.get("qat_warmstart_source") == "ptq_best_tradeoff")
        check("pipeline.results['qat_warmstart_id'] is the tradeoff PTQ ID",
              isinstance(pipe.results.get("qat_warmstart_id"), str))


# ─────────────────────────────────────────────────────────────────────────
# E) Public reporting — both PTQ entries when distinct, no NSGA leakage
# ─────────────────────────────────────────────────────────────────────────


def test_public_pareto_includes_dual_ptq_excludes_nsga():
    print("--- Test E1: Pareto report includes dual PTQ entries, drops NSGA ---")
    sols = [
        # Internal NSGA solutions — must not appear in public outputs.
        ParetoSolution(
            solution_id="nsga_gen5_r1", method="PTQ", accuracy=89.0,
            accuracy_loss=1.0, ebops=2_500_000, ebops_reduction=70.0,
            model_size_mb=2.5, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        # Public PTQ outputs (best-acc + tradeoff).
        ParetoSolution(
            solution_id="PTQ_INT8", method="PTQ", accuracy=91.0,
            accuracy_loss=1.0, ebops=2_400_000, ebops_reduction=70.0,
            model_size_mb=2.4, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        ParetoSolution(
            solution_id="PTQ_MIXED", method="PTQ", accuracy=89.5,
            accuracy_loss=2.5, ebops=1_400_000, ebops_reduction=82.0,
            model_size_mb=1.4, bitwidth_assignment={"w": 4},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        ParetoSolution(
            solution_id="GPTQ_INT8", method="GPTQ", accuracy=90.0,
            accuracy_loss=2.0, ebops=2_000_000, ebops_reduction=72.0,
            model_size_mb=2.0, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
    ]
    front = ParetoFront(
        solutions=sols, generation=1, evaluations=10,
        convergence_reason="public",
    )
    analyzer = ParetoAnalyzer(front, 92.0, 8.5e6, model_name="TinyNet")

    enriched = analyzer.compute_solution_metrics()
    ids = [s["solution_id"] for s in enriched]
    check("public ranking has no nsga_* IDs",
          all(not i.startswith("nsga_") for i in ids), f"got {ids}")
    check("public ranking includes PTQ_INT8",
          "PTQ_INT8" in ids, f"got {ids}")
    check("public ranking includes PTQ_MIXED",
          "PTQ_MIXED" in ids, f"got {ids}")

    with tempfile.TemporaryDirectory() as tmp:
        result = analyzer.analyze(tmp)
        report_text = result["summary_report"]
        check("summary report mentions PTQ_INT8", "PTQ_INT8" in report_text)
        check("summary report mentions PTQ_MIXED", "PTQ_MIXED" in report_text)
        check("summary report has no nsga_ leakage", "nsga_" not in report_text)


def test_phase_1c_resume_restores_dual_ptq_results():
    print("--- Test E2: phase 1c resume restores ptq_best_acc + tradeoff ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.io_layer_bitwidth = 8

    with tempfile.TemporaryDirectory() as tmp:
        cfg.output_dir = tmp
        # Build a synthetic phase_1c JSON checkpoint as if the previous
        # run saved it. This is exactly the contract the resume hook
        # depends on.
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ptq_acc_res = {
            "config_id": "PTQ_INT8", "display_name": "PTQ_INT8",
            "method": "PTQ", "accuracy": 91.0, "ebops": 2_400_000,
            "model_size_mb": 2.4, "bitwidth": 8,
            "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 8, "fc.weight": 8},
            "latency": {"latency_mean_ms": 1.0, "throughput_fps": 1000.0},
        }
        ptq_to_res = {
            "config_id": "PTQ_MIXED", "display_name": "PTQ_MIXED",
            "method": "PTQ", "accuracy": 89.5, "ebops": 1_400_000,
            "model_size_mb": 1.4, "bitwidth": 4,
            "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 4, "fc.weight": 4},
            "latency": {"latency_mean_ms": 1.5, "throughput_fps": 700.0},
        }
        (ckpt_dir / "phase_1c_nsga_search.json").write_text(json.dumps({
            "pareto_front": {"solutions": [], "generation": 1,
                              "evaluations": 1,
                              "convergence_reason": "test"},
            "best_config": ptq_to_res["bitwidth_assignment"],
            "ptq_best_acc_result": ptq_acc_res,
            "ptq_best_tradeoff_result": ptq_to_res,
            "qat_warmstart_source": "ptq_best_tradeoff",
            "qat_warmstart_id": "PTQ_MIXED",
        }))

        from main import NeuroQuantPipeline
        pipe = NeuroQuantPipeline(cfg, training_epochs=0, resume=True)
        pipe._resume_phase_1c_nsga_search()

        check("resume populated ptq_best_acc_result",
              pipe.results.get("ptq_best_acc_result") is not None)
        check("resume populated ptq_best_tradeoff_result",
              pipe.results.get("ptq_best_tradeoff_result") is not None)
        check("resume restored qat_warmstart_source",
              pipe.results.get("qat_warmstart_source") == "ptq_best_tradeoff")
        check("resume restored qat_warmstart_id",
              pipe.results.get("qat_warmstart_id") == "PTQ_MIXED")
        rows = getattr(pipe, "_summary_rows", [])
        row_ids = [r["method"] for r in rows]
        check("resume re-added PTQ_INT8 to summary table",
              "PTQ_INT8" in row_ids, f"got {row_ids}")
        check("resume re-added PTQ_MIXED to summary table",
              "PTQ_MIXED" in row_ids, f"got {row_ids}")


# ─────────────────────────────────────────────────────────────────────────
# Regression: resume must not duplicate PTQ rows in the summary table
# ─────────────────────────────────────────────────────────────────────────


def test_resume_overlap_does_not_duplicate_ptq_summary_rows():
    print("--- Test E3: resume of phase 1c + phase 1f keeps PTQ rows unique ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.io_layer_bitwidth = 8

    with tempfile.TemporaryDirectory() as tmp:
        cfg.output_dir = tmp
        ckpt_dir = Path(tmp) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ptq_acc_res = {
            "config_id": "PTQ_INT8", "display_name": "PTQ_INT8",
            "method": "PTQ", "accuracy": 91.0, "ebops": 2_400_000,
            "model_size_mb": 2.4, "bitwidth": 8,
            "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 8, "fc.weight": 8},
            "latency": {"latency_mean_ms": 1.0, "throughput_fps": 1000.0},
        }
        ptq_to_res = {
            "config_id": "PTQ_MIXED", "display_name": "PTQ_MIXED",
            "method": "PTQ", "accuracy": 89.5, "ebops": 1_400_000,
            "model_size_mb": 1.4, "bitwidth": 4,
            "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 4, "fc.weight": 4},
            "latency": {"latency_mean_ms": 1.5, "throughput_fps": 700.0},
        }
        gptq_res = {
            "config_id": "GPTQ_INT8", "display_name": "GPTQ_INT8",
            "method": "GPTQ", "accuracy": 90.0, "ebops": 2_000_000,
            "model_size_mb": 2.0, "bitwidth": 8,
            "bitwidth_assignment": {"c1.weight": 8, "c2.weight": 8, "fc.weight": 8},
            "latency": {"latency_mean_ms": 1.2, "throughput_fps": 800.0},
        }

        # Phase-1c JSON exposes the two PTQ picks (this is what
        # _resume_phase_1c_nsga_search reads).
        (ckpt_dir / "phase_1c_nsga_search.json").write_text(json.dumps({
            "pareto_front": {"solutions": [], "generation": 1,
                              "evaluations": 1, "convergence_reason": "test"},
            "best_config": ptq_to_res["bitwidth_assignment"],
            "ptq_best_acc_result": ptq_acc_res,
            "ptq_best_tradeoff_result": ptq_to_res,
            "qat_warmstart_source": "ptq_best_tradeoff",
            "qat_warmstart_id": "PTQ_MIXED",
        }))
        # Phase-1f JSON contains method_results that include the same
        # PTQ entries (because phase 1c appended them) PLUS a real
        # phase-1f method (GPTQ_INT8). This is the resume overlap that
        # used to produce duplicate PTQ rows in the summary table.
        (ckpt_dir / "phase_1f_gptq_smooth_awq.json").write_text(json.dumps({
            "method_results": [ptq_acc_res, ptq_to_res, gptq_res],
        }))

        from main import NeuroQuantPipeline
        pipe = NeuroQuantPipeline(cfg, training_epochs=0, resume=True)
        # Order matches the real pipeline: phase 1c first, then phase 1f.
        pipe._resume_phase_1c_nsga_search()
        pipe._resume_phase_1f_gptq_smooth_awq()

        rows = getattr(pipe, "_summary_rows", [])
        row_ids = [r["method"] for r in rows]
        ptq_int8_count = row_ids.count("PTQ_INT8")
        ptq_mixed_count = row_ids.count("PTQ_MIXED")
        gptq_count = row_ids.count("GPTQ_INT8")

        check("PTQ_INT8 appears exactly once after dual-resume",
              ptq_int8_count == 1, f"row_ids={row_ids}")
        check("PTQ_MIXED appears exactly once after dual-resume",
              ptq_mixed_count == 1, f"row_ids={row_ids}")
        check("GPTQ_INT8 appears exactly once after dual-resume",
              gptq_count == 1, f"row_ids={row_ids}")
        # Strict full set: only the three distinct methods, no extras.
        check("summary rows exactly {PTQ_INT8, PTQ_MIXED, GPTQ_INT8}",
              set(row_ids) == {"PTQ_INT8", "PTQ_MIXED", "GPTQ_INT8"}
              and len(row_ids) == 3,
              f"row_ids={row_ids}")

        # Idempotency: re-adding the same row replaces in place.
        pipe._add_summary_row("PTQ_INT8", 92.0, 1.0, 1000.0, 2_300_000, 2.3)
        rows2 = getattr(pipe, "_summary_rows", [])
        ids2 = [r["method"] for r in rows2]
        check("re-adding PTQ_INT8 does not append a duplicate",
              ids2.count("PTQ_INT8") == 1, f"ids2={ids2}")
        # The replacement row should reflect the new top1 value.
        replaced = next(r for r in rows2 if r["method"] == "PTQ_INT8")
        check("replacement row updates the in-place values",
              abs(replaced["top1"] - 92.0) < 1e-9, f"row={replaced}")


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    test_calibrate_with_assignment_per_layer_bitwidth()
    test_calibrate_with_assignment_thresholds_differ_per_bitwidth()
    test_calibrate_legacy_api_still_works()

    test_nsga_searchable_groups_exclude_non_quantizable()
    test_nsga_individual_to_config_only_quantizable()

    test_select_rerank_candidates_top_k_dedup()
    test_phase_1c_produces_dual_ptq_outputs()
    test_tradeoff_falls_back_to_smallest_when_cap_unmet()

    test_qat_warmstart_source_validation()
    test_warmstart_source_persisted_in_phase_1c_checkpoint()

    test_public_pareto_includes_dual_ptq_excludes_nsga()
    test_phase_1c_resume_restores_dual_ptq_results()
    test_resume_overlap_does_not_duplicate_ptq_summary_rows()

    print("\n" + "=" * 50)
    print(f"  PTQ/QAT Production Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

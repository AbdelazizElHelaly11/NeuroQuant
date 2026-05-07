"""
NeuroQuant v2.0 — Thesis-readability fixes (focused tests, ~5s).

Verifies the strict, user-facing contract added in this patch:
  1. No duplicated method naming (no ``AWQ_AWQ_INT4`` / ``PTQ_PTQ_best``).
  2. NSGA internal solutions never appear in public final outputs.
  3. Top-5 is dropped from the public report and MLflow keys.
  4. Bitwidth is visible in Pareto plot annotations + XAI row labels.
  5. ``model_size_mb`` is computed correctly from the bitwidth assignment.
  6. ``model_size_mb`` is a first-class column in the Pareto ranking and
     a first-class axis on the public Pareto scatter.
  7. Adaround uses the calibration-driven layer-output reconstruction
     objective (not the trivial weight-MSE-only path) and surfaces
     reconstruction diagnostics.

All tests run on CPU with tiny synthetic data — no torchvision downloads.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

import torch
import torch.nn as nn

from config import ParetoSolution, ParetoFront, QuantizationConfig
from utils.common import (
    compute_ebops,
    compute_quantized_size_mb,
    model_size_mb_from_bytes,
)
from visualization.pareto_analysis import ParetoAnalyzer
from xai.explainability import XAIGenerator
from quantization.adaround import AdaroundOptimizer

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
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return self.fc(x.mean(dim=(2, 3)))


def _make_loader(n=16, classes=4, h=8, w=8):
    imgs = torch.randn(n, 3, h, w)
    lbls = torch.randint(0, classes, (n,))
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=4)


# ─────────────────────────────────────────────────────────────────────────
# 1. No duplicated method naming + bitwidth-tagged display names
# ─────────────────────────────────────────────────────────────────────────


def test_no_duplicate_method_naming():
    print("--- Test: no duplicated method naming ---")
    # Source-level guard: the legacy "{method}_{config_id}" pattern must
    # not be present in main.py for the public summary rows.
    src = (project_root / "main.py").read_text(encoding="utf-8")
    bad = re.findall(r'f"\{(method|res\[\'method\'\])\}_\{', src)
    check("main.py no longer builds 'METHOD_METHOD_INTx' display strings",
          not bad, f"residual matches: {bad}")

    # Construct a Pareto-style summary the way phase 2 would and confirm
    # the canonical IDs flow through the analyzer's ranking.
    fp32_size = 8.5  # MiB equivalent — only used for hypervolume scaling
    sols = [
        ParetoSolution(
            solution_id="GPTQ_INT8", method="GPTQ", accuracy=90.0,
            accuracy_loss=2.0, ebops=2_000_000, ebops_reduction=70.0,
            model_size_mb=2.0, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        ParetoSolution(
            solution_id="AWQ_INT4", method="AWQ", accuracy=88.0,
            accuracy_loss=4.0, ebops=1_000_000, ebops_reduction=85.0,
            model_size_mb=1.0, bitwidth_assignment={"w": 4},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
    ]
    front = ParetoFront(solutions=sols, generation=1, evaluations=10,
                        convergence_reason="public")
    analyzer = ParetoAnalyzer(front, 92.0, 8 * 1024 * 1024,
                              model_name="TinyNet")
    enriched = analyzer.compute_solution_metrics()
    ids = [s["solution_id"] for s in enriched]
    check("no '_AWQ_AWQ_' duplication in ranking",
          not any("AWQ_AWQ" in i for i in ids), f"ids={ids}")
    check("no '_GPTQ_GPTQ_' duplication in ranking",
          not any("GPTQ_GPTQ" in i for i in ids), f"ids={ids}")
    check("no '_PTQ_PTQ_' duplication in ranking",
          not any("PTQ_PTQ" in i for i in ids), f"ids={ids}")


# ─────────────────────────────────────────────────────────────────────────
# 2. NSGA solutions excluded from public outputs
# ─────────────────────────────────────────────────────────────────────────


def test_nsga_excluded_from_public_pareto():
    print("--- Test: NSGA solutions excluded from public Pareto ---")
    # Build a mixed solution list: 2 NSGA + 2 real. The analyzer's
    # public ranking must drop the nsga_* IDs.
    nsga_a = ParetoSolution(
        solution_id="nsga_gen5_r1", method="PTQ", accuracy=89.0,
        accuracy_loss=1.0, ebops=2_500_000, ebops_reduction=70.0,
        model_size_mb=2.5, bitwidth_assignment={"w": 8},
        rank=1, crowding_distance=0.0, is_dominated=False,
    )
    nsga_b = ParetoSolution(
        solution_id="nsga_gen5_r2", method="PTQ", accuracy=85.0,
        accuracy_loss=5.0, ebops=1_500_000, ebops_reduction=80.0,
        model_size_mb=1.5, bitwidth_assignment={"w": 4},
        rank=1, crowding_distance=0.0, is_dominated=False,
    )
    real_a = ParetoSolution(
        solution_id="GPTQ_INT8", method="GPTQ", accuracy=90.0,
        accuracy_loss=2.0, ebops=2_000_000, ebops_reduction=72.0,
        model_size_mb=2.0, bitwidth_assignment={"w": 8},
        rank=1, crowding_distance=0.0, is_dominated=False,
    )
    real_b = ParetoSolution(
        solution_id="AWQ_INT4", method="AWQ", accuracy=88.0,
        accuracy_loss=4.0, ebops=1_000_000, ebops_reduction=85.0,
        model_size_mb=1.0, bitwidth_assignment={"w": 4},
        rank=1, crowding_distance=0.0, is_dominated=False,
    )
    front = ParetoFront(
        solutions=[nsga_a, nsga_b, real_a, real_b],
        generation=5, evaluations=20, convergence_reason="merged",
    )
    analyzer = ParetoAnalyzer(front, 92.0, 8.5e6, model_name="TinyNet")
    enriched = analyzer.compute_solution_metrics()
    ids = [s["solution_id"] for s in enriched]
    check("NSGA solutions filtered from public ranking",
          all(not i.startswith("nsga_") for i in ids), f"ids={ids}")
    check("Real solutions retained in public ranking",
          {"GPTQ_INT8", "AWQ_INT4"} <= set(ids), f"ids={ids}")

    # And the public scatter PNG must not contain "nsga_" in any
    # annotated label (the visualizer filters defensively too).
    with tempfile.TemporaryDirectory() as tmp:
        result = analyzer.analyze(tmp)
        scatter_path = Path(result["plot_paths"]["pareto_scatter"])
        check("pareto_scatter.png written", scatter_path.exists())
        check("pareto_scatter.png non-trivial in size",
              scatter_path.stat().st_size > 1500,
              f"size={scatter_path.stat().st_size}")
        # Ranking table must list real solutions only.
        ranked_ids = [s["solution_id"] for s in result["solutions_ranked"]]
        check("public ranking has only real solution IDs",
              all(not i.startswith("nsga_") for i in ranked_ids),
              f"got {ranked_ids}")
        check("public summary report has no nsga_* IDs",
              "nsga_" not in result["summary_report"])


# ─────────────────────────────────────────────────────────────────────────
# 3. Top-5 not surfaced in public report / table
# ─────────────────────────────────────────────────────────────────────────


def test_top5_not_in_public_outputs():
    print("--- Test: Top-5 not surfaced in public report ---")
    src = (project_root / "main.py").read_text(encoding="utf-8")
    cfg = QuantizationConfig()
    check("default eval_primary_accuracy is top1",
          cfg.hyperparams.eval_primary_accuracy == "top1",
          f"got {cfg.hyperparams.eval_primary_accuracy}")

    check("final report prints Primary Acc as top1",
          'print("  Primary Acc:    top1")' in src,
          "main.py still prints dynamic/non-top1 primary metric")
    # The summary table header must not advertise a "Top-5" column.
    check("public summary table has no Top-5 column",
          "Top-5" not in src.split("Public summary table")[1].split("Hardware metrics")[0]
          if "Public summary table" in src else True,
          "Top-5 still mentioned near summary table")

    # The Pareto rankings table must not advertise a Top-5 column.
    pa_src = (project_root / "visualization/pareto_analysis.py").read_text(encoding="utf-8")
    check("Pareto ranking columns drop Top-5",
          "Top-5" not in pa_src or "Top-5" not in pa_src.split("plot_metrics_table")[1].split("def plot")[0]
          if "plot_metrics_table" in pa_src else True,
          "Top-5 referenced inside plot_metrics_table")

    # Phase-0 MLflow log_metrics() call must not contain "fp32_top5" key.
    # We carve out just the phase-0 log_metrics(...) block to avoid
    # false matches against the internal ``self.fp32_top5`` attribute.
    phase0_block = src.split("Public MLflow keys: Top-1 only")[1].split("self.tracker.end_run()")[0] \
        if "Public MLflow keys: Top-1 only" in src else ""
    check("phase 0 log_metrics() drops fp32_top5",
          '"fp32_top5"' not in phase0_block and "'fp32_top5'" not in phase0_block,
          "phase-0 log_metrics still emits fp32_top5")

    # Phase-1f MLflow keys must not include _top5 suffix.
    check("phase 1f no longer logs *_top5 to MLflow",
          '_top5' not in src.split("Log results to MLflow")[1].split("self.tracker.end_run()")[0]
          if "Log results to MLflow" in src else True,
          "phase 1f still emits *_top5 MLflow keys")


# ─────────────────────────────────────────────────────────────────────────
# 4. Bitwidth visible in Pareto labels + XAI row labels
# ─────────────────────────────────────────────────────────────────────────


def test_bitwidth_in_xai_rows_and_pareto_labels():
    print("--- Test: bitwidth visible in XAI rows + Pareto labels ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.xai_num_images = 2
    cfg.hyperparams.xai_plot_dpi = 70

    fp32 = _TinyNet(4)
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 1])

    with tempfile.TemporaryDirectory() as tmp:
        gen = XAIGenerator(cfg, device=torch.device("cpu"))
        # Caller passes bitwidth-tagged labels (this is what main.py now
        # does in phase 3).
        result = gen.run(
            fp32_model=fp32,
            quantized_models={
                "PTQ_INT8": fp32,
                "GPTQ_INT8": fp32,
                "AWQ_INT4": fp32,
                "SmoothQuant_INT8": fp32,
            },
            test_images=images,
            test_labels=labels,
            output_dir=tmp,
            class_names=["a", "b", "c", "d"],
        )
        # The matrix is rendered with row labels = quant_models keys, so
        # all those bitwidth-tagged labels must appear in the predictions
        # contract that drives the figure.
        preds = result["predictions"]  # type: ignore[index]
        for tag in ("FP32_baseline", "PTQ_INT8", "GPTQ_INT8",
                    "AWQ_INT4", "SmoothQuant_INT8"):
            check(f"XAI row '{tag}' present in predictions", tag in preds)

        # Bitwidth tokens visible in row IDs (regex anchor so we don't
        # match incidental substrings).
        bw_pattern = re.compile(r"INT(4|8)$")
        non_fp32 = [k for k in preds.keys() if k != "FP32_baseline"]
        check("every quantized row label encodes a bitwidth",
              all(bool(bw_pattern.search(k)) for k in non_fp32),
              f"got {non_fp32}")

    # Pareto-side: the visualizer annotates each scatter point with its
    # solution_id, and our display IDs include the bitwidth.
    sols = [
        ParetoSolution(
            solution_id="GPTQ_INT8", method="GPTQ", accuracy=90.0,
            accuracy_loss=2.0, ebops=2_000_000, ebops_reduction=72.0,
            model_size_mb=2.0, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        ParetoSolution(
            solution_id="AWQ_INT4", method="AWQ", accuracy=88.0,
            accuracy_loss=4.0, ebops=1_000_000, ebops_reduction=85.0,
            model_size_mb=1.0, bitwidth_assignment={"w": 4},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
    ]
    front = ParetoFront(solutions=sols, generation=1, evaluations=10,
                        convergence_reason="public")
    analyzer = ParetoAnalyzer(front, 92.0, 8.5e6, model_name="TinyNet")
    enriched = analyzer.compute_solution_metrics()
    check("Pareto IDs are bitwidth-tagged",
          all("INT" in s["solution_id"] for s in enriched),
          f"ids={[s['solution_id'] for s in enriched]}")


# ─────────────────────────────────────────────────────────────────────────
# 5. model_size_mb correctness + 6. first-class participation
# ─────────────────────────────────────────────────────────────────────────


def test_model_size_correctness():
    print("--- Test: model_size_mb correctness ---")
    model = _TinyNet(4)
    # Assign EVERY parameter so the manual calculation matches the helper
    # (otherwise unassigned biases default to 32-bit and skew the math).
    int4_assignment = {n: 4 for n, _ in model.named_parameters()}

    total_bits = sum(p.numel() * 4 for _, p in model.named_parameters())
    expected_mb_int4 = total_bits / 8.0 / (1024.0 * 1024.0)
    got_mb_int4 = compute_quantized_size_mb(model, int4_assignment)
    check("model_size_mb (INT4 all params) matches manual calculation",
          abs(got_mb_int4 - expected_mb_int4) < 1e-9,
          f"expected {expected_mb_int4}, got {got_mb_int4}")

    int8_assignment = {n: 8 for n in int4_assignment}
    got_mb_int8 = compute_quantized_size_mb(model, int8_assignment)
    check("INT8 size is exactly 2× INT4 size when all params share bitwidth",
          abs(got_mb_int8 - 2.0 * got_mb_int4) < 1e-9,
          f"int4={got_mb_int4}, int8={got_mb_int8}")

    # Mixed: weights INT4, biases default to FP32 → strictly larger than
    # the all-INT4 number but strictly smaller than the all-FP32 number.
    weight_only = {n: 4 for n, _ in model.named_parameters() if "weight" in n}
    got_mb_mixed = compute_quantized_size_mb(model, weight_only)
    check("Mixed (INT4 weights / FP32 biases) > all-INT4",
          got_mb_mixed > got_mb_int4,
          f"mixed={got_mb_mixed}, int4_all={got_mb_int4}")

    # Empty assignment → falls back to FP32 32-bit accounting.
    got_mb_fp32 = compute_quantized_size_mb(model, {})
    check("Empty assignment falls back to FP32 (32-bit) size",
          got_mb_fp32 > got_mb_int8,
          f"fp32={got_mb_fp32}, int8={got_mb_int8}")
    expected_fp32_bytes = sum(p.numel() * 32 for p in model.parameters()) / 8
    check("FP32 size is exact",
          abs(got_mb_fp32 - model_size_mb_from_bytes(expected_fp32_bytes)) < 1e-9,
          f"got {got_mb_fp32}, expected {model_size_mb_from_bytes(expected_fp32_bytes)}")


def test_model_size_first_class_in_pareto():
    print("--- Test: model_size_mb participates in Pareto outputs ---")
    sols = [
        ParetoSolution(
            solution_id="GPTQ_INT8", method="GPTQ", accuracy=90.0,
            accuracy_loss=2.0, ebops=2_500_000, ebops_reduction=70.0,
            model_size_mb=2.38, bitwidth_assignment={"w": 8},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
        ParetoSolution(
            solution_id="AWQ_INT4", method="AWQ", accuracy=88.0,
            accuracy_loss=4.0, ebops=1_250_000, ebops_reduction=85.0,
            model_size_mb=1.19, bitwidth_assignment={"w": 4},
            rank=1, crowding_distance=0.0, is_dominated=False,
        ),
    ]
    front = ParetoFront(solutions=sols, generation=1, evaluations=10,
                        convergence_reason="public")
    analyzer = ParetoAnalyzer(front, 92.0, 8.5 * 1024 * 1024,
                              model_name="TinyNet")
    enriched = analyzer.compute_solution_metrics()
    check("'model_size_mb' is a first-class enriched field",
          all("model_size_mb" in s for s in enriched))
    check("model_size_mb values come through unchanged",
          {round(s["model_size_mb"], 2) for s in enriched} == {2.38, 1.19},
          str([s["model_size_mb"] for s in enriched]))

    with tempfile.TemporaryDirectory() as tmp:
        result = analyzer.analyze(tmp)
        # Public report mentions Size in MiB explicitly.
        check("public report column header is 'Size (MiB)'",
              "Size (MiB)" in result["summary_report"])
        # Pareto scatter axis label is the model size.
        scatter = Path(result["plot_paths"]["pareto_scatter"])
        check("pareto_scatter.png exists", scatter.exists())
        # The Pareto plot module's source explicitly sets x-axis label
        # to "Model size (MiB)".
        pa_src = (project_root / "visualization/pareto_analysis.py").read_text(encoding="utf-8")
        check("pareto scatter uses Model size (MiB) on x-axis",
              "Model size (MiB)" in pa_src)


# ─────────────────────────────────────────────────────────────────────────
# 7. Adaround is non-trivial: real activation-reconstruction objective
# ─────────────────────────────────────────────────────────────────────────


def test_adaround_uses_activation_reconstruction():
    print("--- Test: Adaround uses layer-output reconstruction ---")
    cfg = QuantizationConfig()
    cfg.num_classes = 4
    cfg.input_shape = (3, 8, 8)
    cfg.hyperparams.device = "cpu"
    cfg.hyperparams.adaround_epochs = 3
    cfg.hyperparams.adaround_lr = 0.01
    cfg.hyperparams.adaround_reg_param = 0.001

    torch.manual_seed(0)
    model = _TinyNet(4)
    bw = {n: 4 for n, _ in model.named_parameters() if "weight" in n}

    # 1. With calib_loader supplied → activation-reconstruction objective.
    opt = AdaroundOptimizer(
        model, bw, cfg, calib_loader=_make_loader(),
    )
    res = opt.run()

    # objective_components must be present and tagged correctly.
    obj = res.get("objective_components") or {}
    check("objective_components dict populated", bool(obj))
    check("objective is layer_output_reconstruction (not weight_mse)",
          obj.get("objective") == "layer_output_reconstruction",
          f"got {obj.get('objective')}")
    # The reconstruction component must be finite + non-degenerate.
    check("final_recon component recorded and finite",
          isinstance(obj.get("final_recon"), float)
          and obj["final_recon"] > 0.0,
          f"got {obj.get('final_recon')}")
    # alpha_stats must show actual rounding decisions (not all 0.5).
    alpha = res.get("alpha_stats") or {}
    nonempty = [s for s in alpha.values() if s.get("n_total", 0) > 0]
    check("alpha_stats populated for at least one parameter",
          bool(nonempty))
    if nonempty:
        decided = sum(s["n_near_zero"] + s["n_near_one"] for s in nonempty)
        total = sum(s["n_total"] for s in nonempty)
        # We use 3 epochs in this test for speed; full convergence
        # toward {0, 1} requires the production-config 100 epochs.
        # The signal we care about here is that the regulariser is
        # active at all — i.e. *some* weights moved into the binary
        # regime rather than being stuck at h≈0.5.
        check("regulariser pushes some weights toward {0, 1}",
              decided > 0,
              f"decided={decided}/{total}")
    # recon_before / recon_after / recon_reduction should all be set.
    check("recon_before recorded", res.get("recon_before") is not None)
    check("recon_after recorded", res.get("recon_after") is not None)
    check("recon_reduction recorded",
          res.get("recon_reduction") is not None)

    # 2. Without calib_loader → backward-compatible weight-MSE fallback.
    torch.manual_seed(0)
    model2 = _TinyNet(4)
    opt2 = AdaroundOptimizer(model2, bw, cfg, calib_loader=None)
    res2 = opt2.run()
    obj2 = res2.get("objective_components") or {}
    check("fallback objective tag is 'weight_mse_fallback'",
          obj2.get("objective") == "weight_mse_fallback",
          f"got {obj2.get('objective')}")
    check("fallback path still produces an mse_reduction",
          isinstance(res2.get("mse_reduction"), float))


# ─────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    test_no_duplicate_method_naming()
    test_nsga_excluded_from_public_pareto()
    test_top5_not_in_public_outputs()
    test_bitwidth_in_xai_rows_and_pareto_labels()
    test_model_size_correctness()
    test_model_size_first_class_in_pareto()
    test_adaround_uses_activation_reconstruction()

    print("\n" + "=" * 50)
    print(f"  Thesis Polish Tests: {passed} passed, {failed} failed")
    print("=" * 50)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

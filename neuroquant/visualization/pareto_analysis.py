"""
NeuroQuant v2.0 - Pareto Analysis & Visualization (Phase 2)

Analyses the Pareto front from Phase 1c NSGA-II to identify
the best accuracy-efficiency trade-offs and generate publication-
quality visualisations.

Key metrics computed:
    1. Hypervolume (HV): area dominated by the Pareto front
    2. Spacing: uniformity of solution distribution
    3. Compression ratios per solution
    4. Knee point: best-balanced solution (max distance from extremes line)

Plots generated:
    1. Pareto scatter (accuracy vs EBops reduction)
    2. Trade-off curve (fitted frontier)
    3. Bitwidth distribution heatmap
    4. Metrics summary table

Enhancements over the spec:
    - Correct incremental hypervolume calculation (spec double-counts)
    - Knee point detection via perpendicular distance method
    - Light publication-style plots
    - JSON export for downstream analysis
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from neuroquant.config import (
    ParetoAnalysisResult,
    ParetoFront,
    ParetoSolution,
    QuantizationConfig,
)
from neuroquant.utils.numerics import EPS_GEOMETRIC

logger = logging.getLogger("neuroquant")

# Check for optional dependencies at module level
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ParetoAnalyzer — Metrics Computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ParetoAnalyzer:
    """
    Analyses a Pareto front from NSGA-II and computes quality metrics.

    Operates on ParetoSolution dicts from Phase 1c. All metrics are
    computed in normalised objective space [0, 1] for comparability.
    """

    def __init__(
        self,
        pareto_front: ParetoFront,
        fp32_accuracy: float,
        fp32_ebops: float,
        model_name: str = "Generic CNN",
    ) -> None:
        self.pareto_front = pareto_front
        self.solutions = pareto_front["solutions"]
        # Public-facing analyses must exclude NSGA internal IDs.
        self.public_solutions = [
            s for s in self.solutions
            if not str(s.get("solution_id", "")).startswith("nsga_")
        ]
        self.fp32_accuracy = fp32_accuracy
        self.fp32_ebops = fp32_ebops
        self.model_name = model_name

        # Sort solutions by accuracy (highest first = lowest loss).
        # All public solutions are included; dominated entries carry
        # ``is_dominated=True`` so downstream visualisers can highlight
        # the non-dominated front.
        self.solutions_ranked = sorted(
            self.public_solutions, key=lambda s: s.get("accuracy_loss", 0.0)
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Validate the Pareto front for data integrity.

        Returns:
            List of warning messages (empty = all good).
        """
        warnings: List[str] = []

        if not self.public_solutions:
            warnings.append("Pareto front is empty!")
            return warnings

        for i, sol in enumerate(self.public_solutions):
            # NaN checks
            if math.isnan(sol.get("accuracy", 0)):
                warnings.append(f"Solution {i}: accuracy is NaN")
            if math.isnan(sol.get("ebops", 0)):
                warnings.append(f"Solution {i}: EBops is NaN")

            # Range checks
            acc = sol.get("accuracy", 0)
            if acc < 0 or acc > 100:
                warnings.append(f"Solution {i}: accuracy {acc} out of [0,100]")

            if sol.get("ebops", 0) <= 0:
                warnings.append(f"Solution {i}: EBops <= 0")

        # Non-domination check
        for i, sol_i in enumerate(self.public_solutions):
            for j, sol_j in enumerate(self.public_solutions):
                if i == j:
                    continue
                if (
                    sol_j.get("accuracy_loss", 0) <= sol_i.get("accuracy_loss", 0)
                    and sol_j.get("ebops", 0) <= sol_i.get("ebops", 0)
                    and (
                        sol_j.get("accuracy_loss", 0) < sol_i.get("accuracy_loss", 0)
                        or sol_j.get("ebops", 0) < sol_i.get("ebops", 0)
                    )
                ):
                    warnings.append(
                        f"Solution {i} is dominated by solution {j}!"
                    )

        return warnings

    # ------------------------------------------------------------------
    # Per-Solution Metrics
    # ------------------------------------------------------------------

    def compute_solution_metrics(self) -> List[Dict[str, Any]]:
        """
        Compute enriched metrics for each Pareto solution.

        Returns:
            List of dicts with: accuracy, accuracy_loss, ebops,
            ebops_reduction, compression_ratio, int4_count, int8_count,
            int4_percent.
        """
        enriched: List[Dict[str, Any]] = []

        for sol in self.solutions_ranked:
            sid = str(sol.get("solution_id", "unknown"))
            config = sol.get("bitwidth_assignment", {})
            int4_count = sum(1 for bw in config.values() if bw == 4)
            int8_count = sum(1 for bw in config.values() if bw == 8)
            total = max(int4_count + int8_count, 1)

            ebops = sol.get("ebops", 1.0)
            compression_ratio = self.fp32_ebops / max(ebops, EPS_GEOMETRIC)

            ebops_reduction = sol.get("ebops_reduction", 0.0)
            if ebops_reduction == 0.0 and self.fp32_ebops > 0:
                ebops_reduction = (self.fp32_ebops - ebops) / self.fp32_ebops * 100

            # Real model size in MiB — prefer the value carried on the
            # ParetoSolution (set with the canonical 1024² conversion);
            # fall back to ebops/1024² so this is correct regardless of
            # which constructor produced the solution.
            model_size_mb = sol.get("model_size_mb")
            if not model_size_mb:
                model_size_mb = float(ebops) / (1024.0 * 1024.0)

            enriched.append({
                "solution_id": sid,
                "accuracy": sol.get("accuracy", 0.0),
                "accuracy_loss": sol.get("accuracy_loss", 0.0),
                "ebops": ebops,
                "ebops_mb": float(model_size_mb),
                "model_size_mb": float(model_size_mb),
                "ebops_reduction": ebops_reduction,
                "compression_ratio": compression_ratio,
                "int4_count": int4_count,
                "int8_count": int8_count,
                "int4_percent": int4_count / total * 100,
                "crowding_distance": sol.get("crowding_distance", 0.0),
                # Wave 5: forward the third Pareto axis (ORT latency)
                # so the 3-D plot can read it without an extra lookup.
                "latency_mean_ms": sol.get("latency_mean_ms"),
                # Phase 2 merge marks dominated solutions so plots/tables
                # can show every candidate while highlighting the front.
                "is_dominated": sol.get("is_dominated", False),
            })

        return enriched

    # ------------------------------------------------------------------
    # Global Metrics
    # ------------------------------------------------------------------

    def compute_hypervolume(self) -> float:
        """
        Compute 2D hypervolume (area dominated by the Pareto front).

        Uses the correct incremental algorithm (not the spec's naive
        sum which double-counts overlapping rectangles).

        Objectives: minimise accuracy_loss, minimise ebops.
        Reference point: (max_loss + margin, max_ebops + margin).
        """
        if not self.public_solutions:
            return 0.0

        losses = [s.get("accuracy_loss", 0.0) for s in self.public_solutions]
        ebops_vals = [s.get("ebops", 0.0) for s in self.public_solutions]

        # Reference point: worst case + 10% margin
        ref_loss = max(losses) * 1.1 + 1.0
        ref_ebops = max(ebops_vals) * 1.1 + 1.0

        # Sort by accuracy_loss (ascending)
        points = sorted(zip(losses, ebops_vals), key=lambda p: p[0])

        # Incremental hypervolume: sweep from left to right
        hv = 0.0
        prev_ebops = ref_ebops

        for loss, ebops in points:
            if ebops < prev_ebops:
                width = ref_loss - loss
                height = prev_ebops - ebops
                hv += width * height
                prev_ebops = ebops

        return hv

    def compute_spacing(self) -> float:
        """
        Compute spacing: standard deviation of inter-solution distances.

        Lower spacing = more uniformly distributed solutions.
        Solutions are sorted by accuracy_loss before computing.
        """
        if len(self.public_solutions) <= 1:
            return 0.0

        # Sort by accuracy_loss
        sorted_sols = sorted(
            self.public_solutions, key=lambda s: s.get("accuracy_loss", 0.0)
        )

        # Normalise objectives to [0, 1] for fair distance computation
        losses = [s.get("accuracy_loss", 0.0) for s in sorted_sols]
        ebops_vals = [s.get("ebops", 0.0) for s in sorted_sols]

        loss_range = max(losses) - min(losses) if max(losses) != min(losses) else 1.0
        ebops_range = (
            max(ebops_vals) - min(ebops_vals)
            if max(ebops_vals) != min(ebops_vals) else 1.0
        )

        distances: List[float] = []
        for i in range(len(sorted_sols) - 1):
            d_loss = (losses[i + 1] - losses[i]) / loss_range
            d_ebops = (ebops_vals[i + 1] - ebops_vals[i]) / ebops_range
            dist = math.sqrt(d_loss ** 2 + d_ebops ** 2)
            distances.append(dist)

        if not distances:
            return 0.0

        mean_d = sum(distances) / len(distances)
        variance = sum((d - mean_d) ** 2 for d in distances) / len(distances)
        return math.sqrt(variance)

    def find_knee_point(self) -> Optional[Dict[str, Any]]:
        """
        Find the knee point: solution with the best balance of both
        objectives (maximum perpendicular distance from the line
        connecting the two extreme non-dominated solutions, restricted
        to candidates whose projection onto that line lies inside
        ``[0, 1]``).

        Two corrections vs the previous implementation:

        1. **Dominated solutions are excluded.** Including them was the
           reason the picker landed on PTQ_MIXED (worst-accuracy of all
           10 methods) — the dominated outlier had the largest
           perpendicular distance because it was *outside* the front.
        2. **Projection clamping.** A point's perpendicular distance to
           the extremes line is only meaningful when the perpendicular
           foot lies between the extremes. Otherwise the picker rewards
           solutions that "stick out" past the front rather than knee
           solutions inside it.
        """
        # Filter to non-dominated solutions (the actual Pareto front).
        non_dom = [
            s for s in self.public_solutions
            if not s.get("is_dominated", False)
        ]
        # Drop duplicates by (accuracy_loss, ebops) so the extremes loop
        # doesn't pair a knee with itself when the same configuration
        # was rerun.
        if len(non_dom) < 3:
            # Not enough points to define a knee; caller falls back to
            # best-accuracy.
            return None

        sorted_sols = sorted(
            non_dom, key=lambda s: s.get("accuracy_loss", 0.0)
        )

        best_acc = sorted_sols[0]
        best_ebops = min(sorted_sols, key=lambda s: s.get("ebops", float("inf")))

        if best_acc is best_ebops:
            return None

        x1, y1 = best_acc.get("accuracy_loss", 0), best_acc.get("ebops", 0)
        x2, y2 = best_ebops.get("accuracy_loss", 0), best_ebops.get("ebops", 0)

        x_range = max(abs(x2 - x1), EPS_GEOMETRIC)
        y_range = max(abs(y2 - y1), EPS_GEOMETRIC)

        # Pre-normalise the line direction so we don't recompute it per
        # candidate. Coordinates are scaled into [0, 1]-ish ranges; the
        # exact magnitudes don't matter, only the relative geometry.
        x2n = (x2 - x1) / x_range
        y2n = (y2 - y1) / y_range
        line_len_sq = x2n * x2n + y2n * y2n
        if line_len_sq < EPS_GEOMETRIC:
            return best_acc

        max_dist = -1.0
        knee: Optional[Dict[str, Any]] = None

        for sol in sorted_sols:
            x = (sol.get("accuracy_loss", 0) - x1) / x_range
            y = (sol.get("ebops", 0) - y1) / y_range

            # Projection of (x, y) onto the line in [0, 1]. Skip points
            # whose foot lies outside the extremes — they are not
            # "between" the extreme solutions in any meaningful sense.
            t = (x * x2n + y * y2n) / line_len_sq
            if t < 0.0 or t > 1.0:
                continue

            # Perpendicular distance.
            dist = abs(x * y2n - y * x2n) / math.sqrt(line_len_sq)
            if dist > max_dist:
                max_dist = dist
                knee = sol

        # Fall back to the highest-accuracy non-dominated solution if
        # no candidate sits inside the extremes line (this can happen
        # with a 2-point Pareto front).
        return knee or best_acc

    def find_extreme_solutions(self) -> Dict[str, Optional[Dict]]:
        """Identify extreme solutions and the balanced knee point."""
        if not self.public_solutions:
            return {"best_accuracy": None, "best_ebops": None, "balanced": None}

        best_acc = min(self.public_solutions, key=lambda s: s.get("accuracy_loss", float("inf")))
        best_ebops = min(self.public_solutions, key=lambda s: s.get("ebops", float("inf")))
        balanced = self.find_knee_point()

        return {
            "best_accuracy": best_acc,
            "best_ebops": best_ebops,
            "balanced": balanced or best_acc,
        }

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all global Pareto quality metrics."""
        losses = [s.get("accuracy_loss", 0.0) for s in self.public_solutions]
        ebops_vals = [s.get("ebops", 0.0) for s in self.public_solutions]
        accuracies = [s.get("accuracy", 0.0) for s in self.public_solutions]

        return {
            "num_solutions": len(self.public_solutions),
            "hypervolume": self.compute_hypervolume(),
            "spacing": self.compute_spacing(),
            "accuracy_min": min(accuracies) if accuracies else 0,
            "accuracy_max": max(accuracies) if accuracies else 0,
            "accuracy_range": (max(accuracies) - min(accuracies)) if accuracies else 0,
            "ebops_min": min(ebops_vals) if ebops_vals else 0,
            "ebops_max": max(ebops_vals) if ebops_vals else 0,
            "ebops_range": (max(ebops_vals) - min(ebops_vals)) if ebops_vals else 0,
            "loss_min": min(losses) if losses else 0,
            "loss_max": max(losses) if losses else 0,
            "fp32_accuracy": self.fp32_accuracy,
            "fp32_ebops": self.fp32_ebops,
            "convergence_generation": self.pareto_front.get("generation", 0),
            "total_evaluations": self.pareto_front.get("evaluations", 0),
        }

    # ------------------------------------------------------------------
    # Full Analysis
    # ------------------------------------------------------------------

    def analyze(self, output_dir: Optional[str] = None) -> ParetoAnalysisResult:
        """
        Run the complete Pareto analysis pipeline.

        Args:
            output_dir: Directory to save plots and reports.
                       Defaults to ./artifacts/ in the project root.

        Returns:
            ParetoAnalysisResult with all metrics, plots, and report.
        """
        logger.info("=" * 70)
        logger.info("Phase 2: Pareto Front Analysis & Visualization")
        logger.info("=" * 70)
        logger.info("  Model: %s", self.model_name)
        logger.info("  Solutions (public): %d", len(self.public_solutions))
        logger.info("  FP32 baseline: %.2f%% accuracy, %.2f MB",
                     self.fp32_accuracy, self.fp32_ebops / 1e6)

        # 1. Validate
        warnings = self.validate()
        if warnings:
            for w in warnings:
                logger.warning("  [WARN] %s", w)
        else:
            logger.info("  [OK] All solutions validated (non-dominated)")

        # 2. Per-solution metrics
        solution_metrics = self.compute_solution_metrics()

        # 3. Global metrics
        metrics = self.compute_all_metrics()
        logger.info("  Hypervolume: %.2f", metrics["hypervolume"])
        logger.info("  Spacing: %.4f", metrics["spacing"])
        logger.info("  Accuracy range: %.2f%% - %.2f%%",
                     metrics["accuracy_min"], metrics["accuracy_max"])

        # 4. Extreme solutions
        extremes = self.find_extreme_solutions()
        logger.info("  Best accuracy: %s (%.2f%%)",
                     extremes["best_accuracy"].get("solution_id", "?") if extremes["best_accuracy"] else "N/A",
                     extremes["best_accuracy"].get("accuracy", 0) if extremes["best_accuracy"] else 0)
        logger.info("  Best EBops: %s (%.2f)",
                     extremes["best_ebops"].get("solution_id", "?") if extremes["best_ebops"] else "N/A",
                     extremes["best_ebops"].get("ebops", 0) if extremes["best_ebops"] else 0)

        # 5. Compression ratios
        compression_ratios = [s["compression_ratio"] for s in solution_metrics]

        # 6. Visualize
        plot_paths: Dict[str, str] = {}
        if output_dir and HAS_MATPLOTLIB:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            viz = ParetoVisualizer(
                solution_metrics, metrics, extremes, self.model_name
            )

            plot_paths["pareto_scatter"] = str(
                viz.plot_pareto_scatter(out / "pareto_scatter.png")
            )
            plot_paths["bitwidth_distribution"] = str(
                viz.plot_bitwidth_distribution(out / "bitwidth_dist.png")
            )
            plot_paths["metrics_table"] = str(
                viz.plot_metrics_table(out / "metrics_table.png")
            )
            # Wave 5 G2: 3-objective Pareto when any solution carries an
            # ORT latency. The plot helpers themselves no-op gracefully
            # if no such solutions are present, so we always attempt the
            # render and let the result speak for itself.
            has_latency = any(
                s.get("latency_mean_ms") is not None
                for s in solution_metrics
            )
            if has_latency:
                try:
                    plot_paths["pareto_3d"] = str(
                        viz.plot_3d_pareto(out / "pareto_3d.png")
                    )
                except Exception as exc:
                    logger.warning("  3-D Pareto plot skipped: %s", exc)

            logger.info("  Plots saved to: %s", output_dir)
        elif output_dir and not HAS_MATPLOTLIB:
            logger.warning("  matplotlib not available; skipping plots")

        # 7. Generate report
        report = self._generate_report(solution_metrics, metrics, extremes)

        # 8. Export JSON
        if output_dir:
            json_path = Path(output_dir) / "pareto_front.json"
            self._export_json(json_path, solution_metrics, metrics)
            logger.info("  JSON exported to: %s", json_path)

        logger.info("=" * 70)

        return ParetoAnalysisResult(
            solutions_ranked=solution_metrics,
            metrics=metrics,
            extreme_solutions={
                k: v.get("solution_id", "N/A") if v else "N/A"
                for k, v in extremes.items()
            },
            compression_ratios=compression_ratios,
            plot_paths=plot_paths,
            summary_report=report,
        )

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        solutions: List[Dict],
        metrics: Dict[str, float],
        extremes: Dict[str, Optional[Dict]],
    ) -> str:
        """Generate a markdown summary report."""
        lines: List[str] = []
        lines.append(f"# Pareto Front Analysis: {self.model_name}")
        lines.append("")
        lines.append(f"## Overview")
        lines.append(f"- **Solutions analysed:** {len(solutions)}")
        lines.append(f"- **FP32 baseline:** {self.fp32_accuracy:.2f}% accuracy, "
                      f"{self.fp32_ebops / 1e6:.2f} MB")
        lines.append(f"- **Convergence:** Generation {metrics.get('convergence_generation', '?')}, "
                      f"{metrics.get('total_evaluations', '?')} evaluations")
        lines.append("")

        lines.append("## Quality Metrics")
        lines.append(f"- **Hypervolume:** {metrics['hypervolume']:.2f}")
        lines.append(f"- **Spacing:** {metrics['spacing']:.4f}")
        lines.append(f"- **Accuracy range:** {metrics['accuracy_min']:.2f}% - "
                      f"{metrics['accuracy_max']:.2f}% "
                      f"(delta {metrics['accuracy_range']:.2f}%)")
        lines.append(f"- **EBops range:** {metrics['ebops_min']:.0f} - "
                      f"{metrics['ebops_max']:.0f}")
        lines.append("")

        lines.append("## Extreme Solutions")
        for key, label in [("best_accuracy", "Highest Accuracy"),
                           ("best_ebops", "Lowest EBops"),
                           ("balanced", "Best Balanced (Knee)")]:
            sol = extremes.get(key)
            if sol:
                lines.append(f"- **{label}:** {sol.get('solution_id', 'N/A')} "
                              f"(acc={sol.get('accuracy', 0):.2f}%, "
                              f"ebops={sol.get('ebops', 0):.0f})")
        lines.append("")

        lines.append("## All Solutions (by accuracy)")
        lines.append("")
        lines.append("| Rank | Solution | Top-1 | Loss | Size (MiB) | "
                      "Compression | INT4 % | Status |")
        lines.append("|------|----------|-------|------|------------|"
                      "-------------|--------|--------|")
        for i, sol in enumerate(solutions):
            status = "Dominated" if sol.get("is_dominated", False) else "★ Pareto"
            lines.append(
                f"| {i+1} | {sol['solution_id']} | "
                f"{sol['accuracy']:.2f}% | "
                f"{sol['accuracy_loss']:.2f}% | "
                f"{sol.get('model_size_mb', sol['ebops_mb']):.2f} | "
                f"{sol['compression_ratio']:.1f}x | "
                f"{sol['int4_percent']:.0f}% | "
                f"{status} |"
            )
        lines.append("")

        lines.append("## Recommendations")
        if extremes.get("best_ebops"):
            s = extremes["best_ebops"]
            lines.append(f"1. **For deployment (low memory):** "
                          f"{s.get('solution_id', '')} "
                          f"({s.get('accuracy', 0):.2f}% accuracy)")
        if extremes.get("balanced"):
            s = extremes["balanced"]
            lines.append(f"2. **For balanced trade-off:** "
                          f"{s.get('solution_id', '')} "
                          f"({s.get('accuracy', 0):.2f}% accuracy)")
        if extremes.get("best_accuracy"):
            s = extremes["best_accuracy"]
            lines.append(f"3. **For accuracy:** "
                          f"{s.get('solution_id', '')} "
                          f"({s.get('accuracy', 0):.2f}% accuracy)")

        return "\n".join(lines)

    def _export_json(
        self,
        path: Path,
        solutions: List[Dict],
        metrics: Dict[str, float],
    ) -> None:
        """Export analysis to JSON."""
        data = {
            "model_name": self.model_name,
            "fp32_accuracy": self.fp32_accuracy,
            "fp32_ebops": self.fp32_ebops,
            "metrics": metrics,
            "solutions": solutions,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ParetoVisualizer — Publication-Quality Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ParetoVisualizer:
    """
    Generates publication-quality Pareto front visualisations.

    Uses the shared light publication style from ``visualization.style``
    so colours/markers stay consistent with the XAI plots. Methods that
    can be identified (FP32, PTQ, QAT, GPTQ, AWQ, SmoothQuant) get a
    consistent colour + marker; the rest fall back to neutral grey.
    All plots saved as high-DPI PNGs on a white background.
    """

    # Accent colours used for non-method-specific elements (extremes,
    # frontier line, knee marker). The per-method palette comes from
    # visualization.style.METHOD_STYLE.
    ACCENT = {
        "frontier": "#1f77b4",      # blue dashed connector
        "best_acc": "#2ca02c",      # green up-triangle
        "best_ebops": "#d62728",    # red down-triangle
        "knee":       "#ffb000",    # amber star
        "header_bg":  "#1f3a93",    # table header
    }

    def __init__(
        self,
        solutions: List[Dict[str, Any]],
        metrics: Dict[str, float],
        extremes: Dict[str, Optional[Dict]],
        model_name: str = "Generic CNN",
    ) -> None:
        self.solutions = solutions
        self.metrics = metrics
        self.extremes = extremes
        self.model_name = model_name

    def _setup_style(self) -> None:
        """Configure the shared light publication theme."""
        from neuroquant.visualization.style import apply_publication_style
        apply_publication_style()

    def plot_pareto_scatter(self, output_path: Path) -> Path:
        """Public 2-D Pareto scatter: Top-1 accuracy vs Model size (MiB).

        Shows ALL evaluated solutions. Non-dominated solutions are drawn
        fully opaque with method-specific colours; dominated solutions
        are rendered faded (alpha ≈ 0.35, dashed edge) so the Pareto
        front stands out clearly. The dashed frontier line connects
        only non-dominated points.
        """
        from neuroquant.visualization.style import style_for, family_of

        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 6.5))

        public_solutions = [
            s for s in self.solutions
            if not str(s.get("solution_id", "")).startswith("nsga_")
        ]

        if not public_solutions:
            ax.text(0.5, 0.5, "No public Pareto solutions",
                    ha="center", va="center", fontsize=14)
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        # Separate non-dominated and dominated sets.
        nd_solutions = [s for s in public_solutions if not s.get("is_dominated", False)]
        dom_solutions = [s for s in public_solutions if s.get("is_dominated", False)]

        # ── Latency encoding ──
        # When ORT latency is recorded on every public solution, use it
        # as the third visual channel: marker size ∝ latency (slow =
        # bigger bubble), so the user can read "fast vs slow" at a
        # glance alongside accuracy and size. When latency is missing
        # we keep the compression-ratio bubble sizing for backward
        # compatibility.
        latencies = [
            s.get("latency_mean_ms") for s in public_solutions
            if s.get("latency_mean_ms") is not None
        ]
        has_latency = len(latencies) >= 2 and (max(latencies) - min(latencies)) > 1e-6
        if has_latency:
            lat_min, lat_max = float(min(latencies)), float(max(latencies))

            def _bubble_size(sol: Dict[str, Any]) -> float:
                lat = sol.get("latency_mean_ms")
                if lat is None:
                    return 80.0
                # Map latency to [80, 380] linearly.
                t = (float(lat) - lat_min) / max(lat_max - lat_min, 1e-9)
                return float(80.0 + 300.0 * t)
        else:
            def _bubble_size(sol: Dict[str, Any]) -> float:
                comp = sol.get("compression_ratio", 1.0) or 1.0
                return float(max(60, min(comp * 28, 320)))

        # Frontier connector — only non-dominated, ordered by model size.
        if nd_solutions:
            sorted_pts = sorted(
                ((s["ebops_mb"], s["accuracy"]) for s in nd_solutions),
                key=lambda p: p[0],
            )
            ax.plot(
                [p[0] for p in sorted_pts],
                [p[1] for p in sorted_pts],
                color=self.ACCENT["frontier"], linestyle="--",
                linewidth=1.2, alpha=0.45, zorder=2, label="_frontier",
            )

        # --- Plot dominated solutions first (faded, behind) ---
        plotted_families_dom: set = set()
        for s in dom_solutions:
            tag = s.get("solution_id", "")
            fam = family_of(tag)
            color, marker = style_for(tag)
            size = _bubble_size(s)
            label_dom = f"{fam} (dominated)" if fam not in plotted_families_dom else None
            plotted_families_dom.add(fam)
            ax.scatter(
                s["ebops_mb"], s["accuracy"],
                marker=marker, s=size, c=color,
                edgecolors="#999999", linewidths=1.0,
                alpha=0.35, zorder=3, label=label_dom,
                linestyle="--",
            )
            ax.annotate(
                tag,
                xy=(s["ebops_mb"], s["accuracy"]),
                xytext=(7, 5), textcoords="offset points",
                fontsize=7.5, color="#888888",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec="#eeeeee", alpha=0.7),
                zorder=6,
            )

        # --- Plot non-dominated solutions (full opacity, on top) ---
        plotted_families_nd: set = set()
        for s in nd_solutions:
            tag = s.get("solution_id", "")
            fam = family_of(tag)
            color, marker = style_for(tag)
            size = _bubble_size(s)
            label = fam if fam not in plotted_families_nd else None
            plotted_families_nd.add(fam)
            ax.scatter(
                s["ebops_mb"], s["accuracy"],
                marker=marker, s=size, c=color,
                edgecolors="white", linewidths=1.0,
                alpha=0.9, zorder=5, label=label,
            )
            ax.annotate(
                tag,
                xy=(s["ebops_mb"], s["accuracy"]),
                xytext=(7, 5), textcoords="offset points",
                fontsize=8.5, color="#222222",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec="#dddddd", alpha=0.85),
                zorder=8,
            )

        # Highlight extremes with a second outlined marker on top.
        for key, marker, color, label in [
            ("best_accuracy", "^", self.ACCENT["best_acc"],   "Best Accuracy"),
            ("best_ebops",    "v", self.ACCENT["best_ebops"], "Smallest size"),
            ("balanced",      "*", self.ACCENT["knee"],       "Knee point"),
        ]:
            ext = self.extremes.get(key)
            if not ext:
                continue
            ext_id = ext.get("solution_id", "")
            if str(ext_id).startswith("nsga_"):
                continue  # don't highlight private NSGA solutions
            for s in nd_solutions:
                if s["solution_id"] != ext_id:
                    continue
                ax.scatter(
                    [s["ebops_mb"]], [s["accuracy"]],
                    marker=marker, s=240, facecolors="none",
                    edgecolors=color, linewidths=2.2,
                    zorder=11, label=label,
                )
                break

        ax.set_xlabel("Model size (MiB)")
        ax.set_ylabel("Top-1 accuracy (%)")
        n_nd = len(nd_solutions)
        n_total = len(public_solutions)
        ax.set_title(
            f"Accuracy vs Model size — {self.model_name}\n"
            f"{n_total} methods ({n_nd} Pareto-optimal) · "
            f"HV={self.metrics.get('hypervolume', 0):.2g}",
            pad=12,
        )
        # Marker-size legend hint as caption.
        if has_latency:
            caption = (
                f"marker size ∝ ORT latency "
                f"({lat_min:.2f}–{lat_max:.2f} ms) · faded = dominated"
            )
            # Add a discrete latency size-legend in the upper-left
            # so readers can decode bubble area into milliseconds.
            handles_size = []
            labels_size = []
            for frac, lab in [(0.0, f"fast ({lat_min:.2f} ms)"),
                              (0.5, f"~{(lat_min + lat_max) / 2:.2f} ms"),
                              (1.0, f"slow ({lat_max:.2f} ms)")]:
                size = 80.0 + 300.0 * frac
                handles_size.append(
                    plt.scatter([], [], s=size, c="#888888",
                                edgecolor="white", alpha=0.85)
                )
                labels_size.append(lab)
            size_legend = ax.legend(
                handles_size, labels_size,
                loc="upper left", title="ORT latency (size)",
                fontsize=8, title_fontsize=9, framealpha=0.92,
                labelspacing=1.2, borderpad=0.7, handletextpad=1.1,
                scatterpoints=1,
            )
            ax.add_artist(size_legend)
        else:
            caption = "marker size ∝ compression ratio · faded = dominated"
        ax.text(
            0.99, 0.02, caption,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#dddddd", alpha=0.85),
        )

        # Deduplicate legend entries (matplotlib repeats per-marker labels).
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l in seen or l == "_frontier":
                continue
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
        ax.legend(
            uniq_handles, uniq_labels,
            loc="lower left", ncol=2,
            title="Method / Highlight",
        )

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_3d_pareto(self, output_path: Path) -> Path:
        """3-objective Pareto scatter: accuracy vs size vs ORT latency.

        Generated only when latency numbers are present on the
        solutions (the hardware-aware search path or any method whose
        ONNX latency was measured). The plot uses the same per-method
        colour/marker palette as the 2-D plot so a reader can follow a
        method across both views.

        If matplotlib's 3D backend is unavailable we still write the
        file (an empty axis with a clear message), so callers can
        unconditionally check for ``output_path.exists()``.
        """
        from neuroquant.visualization.style import style_for, family_of
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D

        self._setup_style()

        public_solutions = [
            s for s in self.solutions
            if not str(s.get("solution_id", "")).startswith("nsga_")
            and s.get("latency_mean_ms") is not None
        ]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        if not public_solutions:
            ax.text2D(
                0.5, 0.5, "No latency-tagged Pareto solutions",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=12,
            )
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        plotted_families = set()
        for s in public_solutions:
            tag = s.get("solution_id", "")
            fam = family_of(tag)
            color, marker = style_for(tag)
            comp = s.get("compression_ratio", 1.0) or 1.0
            size = float(max(50, min(comp * 22, 240)))
            label = fam if fam not in plotted_families else None
            plotted_families.add(fam)
            ax.scatter(
                s["ebops_mb"],
                float(s["latency_mean_ms"]),
                s["accuracy"],
                marker=marker, s=size, c=color,
                edgecolors="white", linewidths=0.8,
                alpha=0.9, label=label,
            )

        ax.set_xlabel("Model size (MiB)")
        ax.set_ylabel("ORT latency (ms)")
        ax.set_zlabel("Top-1 accuracy (%)")
        ax.set_title(
            f"3-objective Pareto — {self.model_name}\n"
            f"{len(public_solutions)} method"
            f"{'' if len(public_solutions) == 1 else 's'} · "
            f"axes minimise size/latency, maximise accuracy",
            pad=14,
        )
        ax.legend(loc="upper left", title="Method", fontsize=8)
        try:
            ax.view_init(elev=18, azim=-60)
        except Exception:
            pass

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_bitwidth_distribution(self, output_path: Path) -> Path:
        """Stacked bar chart of INT4 + INT8 layer counts per solution.

        One bar per method; each bar is split into INT4 (orange,
        bottom) and INT8 (blue, top). For uniform-bitwidth solutions
        only one colour shows; for mixed-bitwidth solutions the bar is
        visually mixed — exactly the read the user asked for. INT4
        share is annotated inside the bar (when the segment is large
        enough to fit the label) and the absolute total sits on top.
        """
        self._setup_style()
        n = len(self.solutions)
        fig, ax = plt.subplots(
            figsize=(max(8.0, 0.55 * max(n, 1) + 4), 5.2),
        )

        if not self.solutions:
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        labels = [s["solution_id"] for s in self.solutions]
        int4_counts = np.asarray(
            [s.get("int4_count", 0) for s in self.solutions], dtype=float,
        )
        int8_counts = np.asarray(
            [s.get("int8_count", 0) for s in self.solutions], dtype=float,
        )
        totals = int4_counts + int8_counts
        x = np.arange(n)

        bar_width = 0.66
        bars_int4 = ax.bar(
            x, int4_counts, bar_width,
            label="INT4", color="#ef8a62",
            edgecolor="white", linewidth=0.7,
        )
        bars_int8 = ax.bar(
            x, int8_counts, bar_width, bottom=int4_counts,
            label="INT8", color="#1f77b4",
            edgecolor="white", linewidth=0.7,
        )

        # ── Annotations ──
        # 1. Inside each segment: percent of layers at that bitwidth, when
        #    the segment is tall enough to fit a label cleanly.
        # 2. Above each bar: the total layer count for the method.
        max_total = float(totals.max()) if totals.size else 1.0
        for i in range(n):
            tot = totals[i]
            if tot <= 0:
                continue
            int4_pct = 100.0 * int4_counts[i] / tot
            int8_pct = 100.0 * int8_counts[i] / tot

            # Label INT4 segment
            if int4_counts[i] > 0 and int4_counts[i] / max_total > 0.06:
                ax.text(
                    x[i], int4_counts[i] / 2,
                    f"{int4_pct:.0f}% INT4",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                )
            # Label INT8 segment
            if int8_counts[i] > 0 and int8_counts[i] / max_total > 0.06:
                ax.text(
                    x[i], int4_counts[i] + int8_counts[i] / 2,
                    f"{int8_pct:.0f}% INT8",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold",
                )
            # Total above bar
            ax.text(
                x[i], tot + max_total * 0.015,
                f"{int(tot)}",
                ha="center", va="bottom", fontsize=8.5,
                color="#333333",
            )

        ax.set_xlabel("Solution")
        ax.set_ylabel("Layer count")
        ax.set_title("Bitwidth distribution per solution")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.legend(loc="upper right", title="Bitwidth")
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylim(0, max(max_total * 1.15, 1.0))

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_metrics_table(self, output_path: Path) -> Path:
        """Visual table of ALL solution rankings with key metrics.

        Shows every evaluated method sorted by accuracy (highest first).
        A 'Status' column indicates whether each solution is
        Pareto-optimal (★) or dominated.  Dominated rows are drawn
        with a light grey background for at-a-glance distinction.
        """
        self._setup_style()

        # Sort all solutions by accuracy descending (lowest loss first).
        sorted_sols = sorted(
            self.solutions,
            key=lambda s: s.get("accuracy_loss", 0.0),
        )

        fig, ax = plt.subplots(
            figsize=(14, max(3, len(sorted_sols) * 0.42 + 1.6)),
        )
        ax.axis("off")

        if not sorted_sols:
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        # Latency is the third NSGA objective when the LUT / ONNX
        # runtime measurement is available. Add a Latency column when
        # any solution carries a real number so the table reflects
        # the same objectives the dominance check uses; rows without
        # latency render "-".
        has_latency = any(
            s.get("latency_mean_ms") is not None for s in sorted_sols
        )
        if has_latency:
            headers = ["Rank", "Solution", "Top-1", "Loss", "Size (MiB)",
                       "ORT(ms)", "Compression", "INT4 %", "Status"]
        else:
            headers = ["Rank", "Solution", "Top-1", "Loss", "Size (MiB)",
                       "Compression", "INT4 %", "Status"]

        cell_data = []
        is_dominated_flags = []
        for i, s in enumerate(sorted_sols):
            dominated = s.get("is_dominated", False)
            is_dominated_flags.append(dominated)
            row = [
                str(i + 1),
                s["solution_id"],
                f"{s['accuracy']:.2f}%",
                f"{s['accuracy_loss']:.2f}%",
                f"{s.get('model_size_mb', s['ebops_mb']):.2f}",
            ]
            if has_latency:
                lat = s.get("latency_mean_ms")
                row.append(f"{lat:.2f}" if lat is not None else "-")
            row.extend([
                f"{s['compression_ratio']:.1f}x",
                f"{s['int4_percent']:.0f}%",
                "Dominated" if dominated else "★ Pareto",
            ])
            cell_data.append(row)

        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.45)

        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.ACCENT["header_bg"])
                cell.set_text_props(color="white", fontweight="bold")
            else:
                data_idx = row - 1  # row 0 is header
                if data_idx < len(is_dominated_flags) and is_dominated_flags[data_idx]:
                    # Dominated rows: light grey
                    cell.set_facecolor("#f0f0f0")
                    cell.set_text_props(color="#777777")
                else:
                    cell.set_facecolor("#f7f7fa" if row % 2 == 0 else "white")
                    cell.set_text_props(color="#222222")
            cell.set_edgecolor("#bbbbbb")

        n_pareto = sum(1 for d in is_dominated_flags if not d)
        n_total = len(sorted_sols)
        ax.set_title(
            f"All solutions — {self.model_name}  "
            f"({n_pareto} Pareto-optimal / {n_total} total)",
            pad=18,
        )

        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

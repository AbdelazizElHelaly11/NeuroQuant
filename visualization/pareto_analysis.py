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

from config import (
    ParetoAnalysisResult,
    ParetoFront,
    ParetoSolution,
    QuantizationConfig,
)

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

        # Sort solutions by accuracy (highest first = lowest loss)
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
            compression_ratio = self.fp32_ebops / max(ebops, 1e-8)

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
        connecting the two extreme solutions).

        This is a standard technique in multi-objective optimisation.
        """
        if len(self.public_solutions) < 3:
            return None

        sorted_sols = sorted(
            self.public_solutions, key=lambda s: s.get("accuracy_loss", 0.0)
        )

        # Extreme points: best accuracy (lowest loss) and best EBops
        best_acc = sorted_sols[0]
        best_ebops = min(sorted_sols, key=lambda s: s.get("ebops", float("inf")))

        # Line from best_acc to best_ebops
        x1, y1 = best_acc.get("accuracy_loss", 0), best_acc.get("ebops", 0)
        x2, y2 = best_ebops.get("accuracy_loss", 0), best_ebops.get("ebops", 0)

        # Normalise
        x_range = max(abs(x2 - x1), 1e-8)
        y_range = max(abs(y2 - y1), 1e-8)

        max_dist = -1.0
        knee = None

        for sol in sorted_sols:
            x = (sol.get("accuracy_loss", 0) - x1) / x_range
            y = (sol.get("ebops", 0) - y1) / y_range
            x2n = (x2 - x1) / x_range
            y2n = (y2 - y1) / y_range

            # Perpendicular distance from point to line
            line_len = math.sqrt(x2n ** 2 + y2n ** 2)
            if line_len < 1e-8:
                continue
            dist = abs(x * y2n - y * x2n) / line_len

            if dist > max_dist:
                max_dist = dist
                knee = sol

        return knee

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

        lines.append("## Solution Rankings")
        lines.append("")
        lines.append("| Rank | Solution | Top-1 | Loss | Size (MiB) | "
                      "Compression | INT4 % |")
        lines.append("|------|----------|-------|------|------------|"
                      "-------------|--------|")
        for i, sol in enumerate(solutions):
            lines.append(
                f"| {i+1} | {sol['solution_id']} | "
                f"{sol['accuracy']:.2f}% | "
                f"{sol['accuracy_loss']:.2f}% | "
                f"{sol.get('model_size_mb', sol['ebops_mb']):.2f} | "
                f"{sol['compression_ratio']:.1f}x | "
                f"{sol['int4_percent']:.0f}% |"
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
        from visualization.style import apply_publication_style
        apply_publication_style()

    def plot_pareto_scatter(self, output_path: Path) -> Path:
        """Public 2-D Pareto scatter: Top-1 accuracy vs Model size (MiB).

        Model size is the first-class second objective: accurate, small
        and quantization-method aware. Each point is labelled with its
        canonical bitwidth-tagged ID (``GPTQ_INT8``, ``AWQ_INT4``, …)
        and uses the per-method colour/marker from
        ``visualization.style.METHOD_STYLE``. NSGA internal solutions
        (IDs starting ``nsga_``) are filtered out before plotting; they
        live in checkpoints for reproducibility.
        """
        from visualization.style import style_for, family_of

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

        # Frontier connector ordered by model size — illustrative only.
        sorted_pts = sorted(
            ((s["ebops_mb"], s["accuracy"]) for s in public_solutions),
            key=lambda p: p[0],
        )
        ax.plot(
            [p[0] for p in sorted_pts],
            [p[1] for p in sorted_pts],
            color=self.ACCENT["frontier"], linestyle="--",
            linewidth=1.2, alpha=0.45, zorder=2, label="_frontier",
        )

        # Per-method scatter + per-point ID label so the bitwidth is
        # always visible alongside the marker.
        plotted_families = set()
        for s in public_solutions:
            tag = s.get("solution_id", "")
            fam = family_of(tag)
            color, marker = style_for(tag)
            comp = s.get("compression_ratio", 1.0) or 1.0
            size = float(max(60, min(comp * 28, 320)))
            label = fam if fam not in plotted_families else None
            plotted_families.add(fam)
            ax.scatter(
                s["ebops_mb"], s["accuracy"],
                marker=marker, s=size, c=color,
                edgecolors="white", linewidths=1.0,
                alpha=0.9, zorder=5, label=label,
            )
            # Inline bitwidth-aware label next to every public point.
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
            for s in public_solutions:
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
        ax.set_title(
            f"Accuracy vs Model size — {self.model_name}\n"
            f"{len(public_solutions)} public methods · "
            f"HV={self.metrics.get('hypervolume', 0):.2g}",
            pad=12,
        )
        # Marker-size legend hint as caption.
        ax.text(
            0.99, 0.02,
            "marker size ∝ compression ratio · labels show INT bitwidth",
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

    def plot_bitwidth_distribution(self, output_path: Path) -> Path:
        """Stacked bar chart of INT4 vs INT8 layer counts per solution."""
        self._setup_style()
        fig, ax = plt.subplots(
            figsize=(max(8.0, 0.45 * max(len(self.solutions), 1) + 4), 5),
        )

        if not self.solutions:
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        labels = [s["solution_id"] for s in self.solutions]
        int4_counts = [s["int4_count"] for s in self.solutions]
        int8_counts = [s["int8_count"] for s in self.solutions]
        x = np.arange(len(labels))

        bar_width = 0.62
        ax.bar(x, int4_counts, bar_width,
               label="INT4", color="#ef8a62",
               edgecolor="white", linewidth=0.7)
        ax.bar(x, int8_counts, bar_width, bottom=int4_counts,
               label="INT8", color="#1f77b4",
               edgecolor="white", linewidth=0.7)

        ax.set_xlabel("Solution")
        ax.set_ylabel("Layer count")
        ax.set_title("Bitwidth distribution per Pareto solution")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.legend(loc="upper right", title="Bitwidth")
        ax.grid(True, axis="y", alpha=0.4)

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_metrics_table(self, output_path: Path) -> Path:
        """Visual table of solution rankings with key metrics (light theme)."""
        self._setup_style()
        fig, ax = plt.subplots(
            figsize=(12, max(3, len(self.solutions) * 0.42 + 1.6)),
        )
        ax.axis("off")

        if not self.solutions:
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        headers = ["Rank", "Solution", "Top-1", "Loss", "Size (MiB)",
                   "Compression", "INT4 %"]
        cell_data = []
        for i, s in enumerate(self.solutions):
            cell_data.append([
                str(i + 1),
                s["solution_id"],
                f"{s['accuracy']:.2f}%",
                f"{s['accuracy_loss']:.2f}%",
                f"{s.get('model_size_mb', s['ebops_mb']):.2f}",
                f"{s['compression_ratio']:.1f}x",
                f"{s['int4_percent']:.0f}%",
            ])

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
                cell.set_facecolor("#f7f7fa" if row % 2 == 0 else "white")
                cell.set_text_props(color="#222222")
            cell.set_edgecolor("#bbbbbb")

        ax.set_title(
            f"Pareto solution rankings — {self.model_name}",
            pad=18,
        )

        fig.savefig(output_path)
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

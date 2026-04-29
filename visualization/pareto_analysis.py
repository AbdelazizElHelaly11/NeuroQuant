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
    - Dark-themed publication-quality plots
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
        self.fp32_accuracy = fp32_accuracy
        self.fp32_ebops = fp32_ebops
        self.model_name = model_name

        # Sort solutions by accuracy (highest first = lowest loss)
        self.solutions_ranked = sorted(
            self.solutions, key=lambda s: s.get("accuracy_loss", 0.0)
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

        if not self.solutions:
            warnings.append("Pareto front is empty!")
            return warnings

        for i, sol in enumerate(self.solutions):
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
        for i, sol_i in enumerate(self.solutions):
            for j, sol_j in enumerate(self.solutions):
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
            config = sol.get("bitwidth_assignment", {})
            int4_count = sum(1 for bw in config.values() if bw == 4)
            int8_count = sum(1 for bw in config.values() if bw == 8)
            total = max(int4_count + int8_count, 1)

            ebops = sol.get("ebops", 1.0)
            compression_ratio = self.fp32_ebops / max(ebops, 1e-8)

            ebops_reduction = sol.get("ebops_reduction", 0.0)
            if ebops_reduction == 0.0 and self.fp32_ebops > 0:
                ebops_reduction = (self.fp32_ebops - ebops) / self.fp32_ebops * 100

            enriched.append({
                "solution_id": sol.get("solution_id", "unknown"),
                "accuracy": sol.get("accuracy", 0.0),
                "accuracy_loss": sol.get("accuracy_loss", 0.0),
                "ebops": ebops,
                "ebops_mb": ebops / 1e6,
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
        if not self.solutions:
            return 0.0

        losses = [s.get("accuracy_loss", 0.0) for s in self.solutions]
        ebops_vals = [s.get("ebops", 0.0) for s in self.solutions]

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
        if len(self.solutions) <= 1:
            return 0.0

        # Sort by accuracy_loss
        sorted_sols = sorted(
            self.solutions, key=lambda s: s.get("accuracy_loss", 0.0)
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
        if len(self.solutions) < 3:
            return None

        sorted_sols = sorted(
            self.solutions, key=lambda s: s.get("accuracy_loss", 0.0)
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
        if not self.solutions:
            return {"best_accuracy": None, "best_ebops": None, "balanced": None}

        best_acc = min(self.solutions, key=lambda s: s.get("accuracy_loss", float("inf")))
        best_ebops = min(self.solutions, key=lambda s: s.get("ebops", float("inf")))
        balanced = self.find_knee_point()

        return {
            "best_accuracy": best_acc,
            "best_ebops": best_ebops,
            "balanced": balanced or best_acc,
        }

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all global Pareto quality metrics."""
        losses = [s.get("accuracy_loss", 0.0) for s in self.solutions]
        ebops_vals = [s.get("ebops", 0.0) for s in self.solutions]
        accuracies = [s.get("accuracy", 0.0) for s in self.solutions]

        return {
            "num_solutions": len(self.solutions),
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
        logger.info("  Solutions: %d", len(self.solutions))
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
        lines.append("| Rank | Solution | Accuracy | Loss | EBops (MB) | "
                      "Compression | INT4 % |")
        lines.append("|------|----------|----------|------|------------|"
                      "-------------|--------|")
        for i, sol in enumerate(solutions):
            lines.append(
                f"| {i+1} | {sol['solution_id']} | "
                f"{sol['accuracy']:.2f}% | "
                f"{sol['accuracy_loss']:.2f}% | "
                f"{sol['ebops_mb']:.2f} | "
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

    Uses a dark theme with carefully chosen colours for clarity.
    All plots saved as high-DPI PNGs.
    """

    # Colour palette (accessible, dark-theme friendly)
    COLORS = {
        "primary": "#4fc3f7",       # Light blue
        "secondary": "#81c784",     # Green
        "accent": "#ff8a65",        # Orange
        "highlight": "#e57373",     # Red
        "knee": "#ffd54f",          # Amber/gold
        "text": "#e0e0e0",          # Light grey
        "grid": "#424242",          # Dark grey
        "bg": "#1e1e2e",            # Dark background
        "panel": "#2d2d3d",         # Panel background
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
        """Configure dark theme for all plots."""
        plt.rcParams.update({
            "figure.facecolor": self.COLORS["bg"],
            "axes.facecolor": self.COLORS["panel"],
            "axes.edgecolor": self.COLORS["grid"],
            "axes.labelcolor": self.COLORS["text"],
            "text.color": self.COLORS["text"],
            "xtick.color": self.COLORS["text"],
            "ytick.color": self.COLORS["text"],
            "grid.color": self.COLORS["grid"],
            "grid.alpha": 0.3,
            "font.family": "sans-serif",
            "font.size": 11,
        })

    def plot_pareto_scatter(self, output_path: Path) -> Path:
        """
        2D scatter plot: Accuracy vs EBops reduction.

        Points coloured by INT4 percentage, sized by compression ratio.
        Extreme solutions and knee point labelled.
        """
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 7))

        if not self.solutions:
            ax.text(0.5, 0.5, "No Pareto solutions",
                    ha="center", va="center", fontsize=14,
                    color=self.COLORS["text"])
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path

        accuracies = [s["accuracy"] for s in self.solutions]
        ebops_red = [s["ebops_reduction"] for s in self.solutions]
        int4_pct = [s["int4_percent"] for s in self.solutions]
        comp_ratios = [s["compression_ratio"] for s in self.solutions]

        # Scale point sizes by compression ratio
        sizes = [max(40, min(r * 30, 300)) for r in comp_ratios]

        # Scatter with INT4% colour mapping
        scatter = ax.scatter(
            ebops_red, accuracies,
            c=int4_pct, cmap="YlOrRd", s=sizes,
            edgecolors="white", linewidths=0.8,
            alpha=0.85, zorder=5,
        )

        # Connect frontier with a line (sorted by EBops reduction)
        sorted_by_ebops = sorted(
            zip(ebops_red, accuracies), key=lambda p: p[0]
        )
        ax.plot(
            [p[0] for p in sorted_by_ebops],
            [p[1] for p in sorted_by_ebops],
            color=self.COLORS["primary"], alpha=0.4,
            linewidth=1.5, linestyle="--", zorder=3,
        )

        # Label extreme solutions
        for key, marker, color, label in [
            ("best_accuracy", "^", self.COLORS["secondary"], "Best Accuracy"),
            ("best_ebops", "v", self.COLORS["accent"], "Best EBops"),
            ("balanced", "*", self.COLORS["knee"], "Knee Point"),
        ]:
            ext = self.extremes.get(key)
            if ext:
                ext_id = ext.get("solution_id", "")
                # Find matching solution in our enriched list
                for s in self.solutions:
                    if s["solution_id"] == ext_id:
                        ax.scatter(
                            [s["ebops_reduction"]], [s["accuracy"]],
                            marker=marker, s=200, c=color,
                            edgecolors="white", linewidths=2,
                            zorder=10, label=label,
                        )
                        break

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("INT4 Layers (%)", fontsize=11)
        cbar.ax.yaxis.set_tick_params(color=self.COLORS["text"])

        # Labels and title
        ax.set_xlabel("EBops Reduction (%)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Pareto Front: {self.model_name}\n"
            f"{len(self.solutions)} solutions | HV={self.metrics.get('hypervolume', 0):.1f}",
            fontsize=14, fontweight="bold", pad=15,
        )

        ax.legend(loc="lower left", framealpha=0.7, fontsize=10)
        ax.grid(True, alpha=0.2)

        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_bitwidth_distribution(self, output_path: Path) -> Path:
        """
        Stacked bar chart: INT4 vs INT8 layer counts per solution.
        """
        self._setup_style()
        fig, ax = plt.subplots(figsize=(10, 5))

        if not self.solutions:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path

        labels = [s["solution_id"] for s in self.solutions]
        int4_counts = [s["int4_count"] for s in self.solutions]
        int8_counts = [s["int8_count"] for s in self.solutions]
        x = np.arange(len(labels))

        bar_width = 0.6
        ax.bar(x, int4_counts, bar_width,
               label="INT4", color=self.COLORS["accent"], alpha=0.85)
        ax.bar(x, int8_counts, bar_width, bottom=int4_counts,
               label="INT8", color=self.COLORS["primary"], alpha=0.85)

        ax.set_xlabel("Solution", fontsize=12, fontweight="bold")
        ax.set_ylabel("Layer Count", fontsize=12, fontweight="bold")
        ax.set_title("Bitwidth Distribution per Pareto Solution",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="upper right", framealpha=0.7)
        ax.grid(True, axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

    def plot_metrics_table(self, output_path: Path) -> Path:
        """
        Visual table of solution rankings with key metrics.
        """
        self._setup_style()
        fig, ax = plt.subplots(figsize=(12, max(3, len(self.solutions) * 0.45 + 1.5)))
        ax.axis("off")

        if not self.solutions:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path

        headers = ["Rank", "Solution", "Accuracy", "Loss", "EBops (MB)",
                    "Compression", "INT4 %"]
        cell_data = []
        for i, s in enumerate(self.solutions):
            cell_data.append([
                str(i + 1),
                s["solution_id"],
                f"{s['accuracy']:.2f}%",
                f"{s['accuracy_loss']:.2f}%",
                f"{s['ebops_mb']:.2f}",
                f"{s['compression_ratio']:.1f}x",
                f"{s['int4_percent']:.0f}%",
            ])

        table = ax.table(
            cellText=cell_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

        # Header styling
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#3f51b5")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(
                    self.COLORS["panel"] if row % 2 == 0 else "#252535"
                )
                cell.set_text_props(color=self.COLORS["text"])
            cell.set_edgecolor(self.COLORS["grid"])

        ax.set_title(
            f"Pareto Solution Rankings: {self.model_name}",
            fontsize=13, fontweight="bold", pad=20,
            color=self.COLORS["text"],
        )

        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor=self.COLORS["bg"])
        plt.close(fig)
        logger.info("    Saved: %s", output_path.name)
        return output_path

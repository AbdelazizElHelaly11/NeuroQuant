"""
NeuroQuant v2.0 — Automatic HTML Report Generation.

Compiles all pipeline artifacts (Pareto plots, Grad-CAM grids, metrics
tables, sensitivity heatmaps, error attribution charts) into a single,
self-contained HTML file with embedded base64 images.

The report is generated at the end of Phase 4 and saved as
``artifacts/neuroquant_report.html``.  It can be opened in any browser
and shared without external dependencies.

Usage::

    from neuroquant.visualization.report import generate_html_report
    generate_html_report(output_dir, config, results)
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("neuroquant")


def generate_html_report(
    output_dir: str,
    config: Any,
    results: Dict[str, Any],
    *,
    report_name: str = "neuroquant_report.html",
) -> Optional[str]:
    """Generate a self-contained HTML report from all pipeline artifacts.

    Scans the ``output_dir`` for images (PNG/JPG), JSON summaries, and
    text reports, embedding everything into a single HTML file.

    Args:
        output_dir:  Path to the artifacts directory.
        config:      QuantizationConfig instance.
        results:     Pipeline results dict.
        report_name: Output filename for the report.

    Returns:
        Path to the generated HTML file, or None on failure.
    """
    out = Path(output_dir)
    if not out.exists():
        logger.warning("Output dir %s not found; skipping report.", output_dir)
        return None

    try:
        html = _build_html(out, config, results)
        report_path = out / report_name
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML report generated: %s", report_path)
        return str(report_path)
    except Exception as exc:
        logger.warning("HTML report generation failed: %s", exc)
        return None


def _build_html(
    out: Path,
    config: Any,
    results: Dict[str, Any],
) -> str:
    """Build the complete HTML string."""
    model_name = getattr(config, "model_name", "Unknown")
    dataset_name = getattr(config, "dataset_name", "Unknown")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    sections: List[str] = []

    # ── Header ──
    sections.append(f"""
    <div class="header">
        <h1>🧠 NeuroQuant v2.0 — Experiment Report</h1>
        <p class="subtitle">
            Model: <strong>{model_name}</strong> &nbsp;|&nbsp;
            Dataset: <strong>{dataset_name}</strong> &nbsp;|&nbsp;
            Generated: {timestamp}
        </p>
    </div>
    """)

    # ── Executive Summary ──
    fp32_acc = results.get("fp32_acc", 0.0)
    hv = results.get("hypervolume", 0.0)
    n_methods = len(results.get("method_results", []))
    sections.append(f"""
    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{fp32_acc:.2f}%</div>
                <div class="metric-label">FP32 Baseline</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{n_methods}</div>
                <div class="metric-label">Methods Evaluated</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{hv:.2f}</div>
                <div class="metric-label">Pareto Hypervolume</div>
            </div>
        </div>
    </div>
    """)

    # ── Sensitivity Analysis (if exists) ──
    sens_img = _find_image(out, "sensitivity_heatmap.png")
    tier_img = _find_image(out, "tier_distribution.png")
    if sens_img or tier_img:
        sections.append('<div class="section"><h2>🔍 Layer Sensitivity Analysis</h2>')
        if sens_img:
            sections.append(f'<img src="{sens_img}" alt="Sensitivity Heatmap">')
        if tier_img:
            sections.append(f'<img src="{tier_img}" alt="Tier Distribution">')
        sections.append('</div>')

    # ── Pareto Analysis ──
    pareto_scatter = _find_image(out, "pareto_scatter.png", subdir="pareto")
    bitwidth_dist = _find_image(out, "bitwidth_dist.png", subdir="pareto")
    metrics_table = _find_image(out, "metrics_table.png", subdir="pareto")
    pareto_3d = _find_image(out, "pareto_3d.png", subdir="pareto")

    if any([pareto_scatter, bitwidth_dist, metrics_table]):
        sections.append('<div class="section"><h2>📈 Pareto Front Analysis</h2>')
        if pareto_scatter:
            sections.append(f'<img src="{pareto_scatter}" alt="Pareto Scatter">')
        if metrics_table:
            sections.append(f'<img src="{metrics_table}" alt="Metrics Table">')
        if bitwidth_dist:
            sections.append(f'<img src="{bitwidth_dist}" alt="Bitwidth Distribution">')
        if pareto_3d:
            sections.append(f'<img src="{pareto_3d}" alt="3D Pareto">')
        sections.append('</div>')

    # ── Quantization Error Attribution (if exists) ──
    # Per-method files are saved as ``error_attribution_<method>.png``;
    # the cross-method comparison is ``error_comparison.png``. As of the
    # latest pipeline change these live under
    # ``artifacts/error_attribution/`` rather than the artifacts root,
    # so we look in the subdirectory first and fall back to root for
    # older runs.
    ea_subdir = out / "error_attribution"
    cmp_img = _find_image(out, "error_comparison.png", subdir="error_attribution")
    if cmp_img is None:
        cmp_img = _find_image(out, "error_comparison.png")
    if ea_subdir.is_dir():
        per_method = sorted(ea_subdir.glob("error_attribution_*.png"))
    else:
        per_method = sorted(out.glob("error_attribution_*.png"))
    if cmp_img or per_method:
        sections.append('<div class="section"><h2>🎯 Quantization Error Attribution</h2>')
        if cmp_img:
            sections.append(f'<img src="{cmp_img}" alt="Cross-Method Error Comparison">')
        for img in per_method:
            b64 = _embed_image(img)
            if b64:
                sections.append(
                    f'<img src="{b64}" alt="{img.stem}">'
                )
        sections.append('</div>')

    # ── XAI Explainability ──
    gradcam_grid = _find_image(out, "comparison_matrix.png", subdir="xai")
    if gradcam_grid:
        sections.append('<div class="section"><h2>🧪 XAI Explainability</h2>')
        sections.append(f'<img src="{gradcam_grid}" alt="XAI Comparison Matrix">')
        sections.append('</div>')

    # ── Individual Grad-CAM images ──
    xai_dir = out / "xai"
    if xai_dir.exists():
        gradcam_imgs = sorted(xai_dir.glob("gradcam_*.png"))[:10]
        shap_imgs = sorted(xai_dir.glob("shap_*.png"))[:10]
        if gradcam_imgs:
            sections.append('<div class="section"><h2>🔬 Grad-CAM Heatmaps</h2>')
            sections.append('<div class="image-grid">')
            for img in gradcam_imgs:
                b64 = _embed_image(img)
                if b64:
                    sections.append(
                        f'<img src="{b64}" alt="{img.stem}" class="grid-img">'
                    )
            sections.append('</div></div>')

    # ── Deployment Info ──
    backends_section = _build_deployment_section(out, results)
    if backends_section:
        sections.append(backends_section)

    # ── Pipeline Report ──
    report_txt = out / "pipeline_report.txt"
    if report_txt.exists():
        report_content = report_txt.read_text(encoding="utf-8", errors="replace")
        sections.append(f"""
        <div class="section">
            <h2>📋 Pipeline Report</h2>
            <pre class="report-text">{report_content}</pre>
        </div>
        """)

    # ── Reproducibility ──
    manifest = out / "reproducibility_manifest.json"
    if manifest.exists():
        try:
            manifest_data = json.loads(manifest.read_text())
            env = manifest_data.get("environment", {})
            sections.append(f"""
            <div class="section">
                <h2>🔄 Reproducibility</h2>
                <table class="info-table">
                    <tr><td>Python</td><td>{env.get('python_version', 'N/A')}</td></tr>
                    <tr><td>PyTorch</td><td>{env.get('torch_version', 'N/A')}</td></tr>
                    <tr><td>CUDA</td><td>{env.get('cuda_version', 'N/A')}</td></tr>
                    <tr><td>GPU</td><td>{env.get('gpu_name', 'N/A')}</td></tr>
                    <tr><td>OS</td><td>{env.get('os', 'N/A')}</td></tr>
                </table>
            </div>
            """)
        except Exception:
            pass

    body = "\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroQuant Report — {model_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #f5f6fa;
            color: #2c3e50;
            line-height: 1.6;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        }}
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 8px;
        }}
        .subtitle {{
            opacity: 0.85;
            font-size: 1rem;
        }}
        .section {{
            background: white;
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}
        .section h2 {{
            font-size: 1.4rem;
            margin-bottom: 16px;
            color: #1a1a2e;
            border-bottom: 2px solid #e8ecf1;
            padding-bottom: 8px;
        }}
        .section img {{
            max-width: 100%;
            border-radius: 8px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
        }}
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.85;
            margin-top: 4px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 12px;
        }}
        .grid-img {{
            width: 100%;
            border-radius: 8px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .info-table td {{
            padding: 10px 16px;
            border-bottom: 1px solid #e8ecf1;
        }}
        .info-table tr:nth-child(even) {{
            background: #f8f9fc;
        }}
        .info-table td:first-child {{
            font-weight: 600;
            width: 140px;
            color: #555;
        }}
        .report-text {{
            background: #f8f9fc;
            padding: 16px;
            border-radius: 8px;
            font-family: 'Cascadia Code', 'Fira Code', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    {body}
    <footer>
        Generated by NeuroQuant v2.0 &mdash;
        Neural Network Quantization Framework
    </footer>
</body>
</html>"""


def _find_image(
    base: Path, filename: str, subdir: str = ""
) -> Optional[str]:
    """Find an image and return as a base64 data URI, or None."""
    search = base / subdir / filename if subdir else base / filename
    if not search.exists():
        # Also try the base directory directly
        alt = base / filename
        if not alt.exists():
            return None
        search = alt
    return _embed_image(search)


def _embed_image(path: Path) -> Optional[str]:
    """Convert an image file to a base64-encoded data URI."""
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(suffix, "image/png")
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _build_deployment_section(
    out: Path,
    results: Dict[str, Any],
) -> Optional[str]:
    """Build the deployment backends section if info is available."""
    try:
        from neuroquant.utils.deployment_export import available_backends
        backends = available_backends()
    except ImportError:
        backends = ["onnxruntime"]

    rows = []
    for b in backends:
        status = "Available"
        rows.append(f"<tr><td>{b}</td><td>{status}</td></tr>")

    if not rows:
        return None

    # Detection-only: list any TRT / OpenVINO artefacts that were
    # actually built this run. Each entry shows the method, backend,
    # output path, and on-disk size.
    exports = results.get("deployment_exports") or []
    export_rows = []
    for e in exports:
        backend = e.get("backend", "?")
        method = e.get("method", "?")
        path = (
            e.get("engine_path") or e.get("xml_path") or ""
        )
        size_mb = (
            e.get("engine_size_mb") or e.get("ir_size_mb") or 0.0
        )
        export_rows.append(
            f"<tr><td>{method}</td><td>{backend}</td>"
            f"<td>{path}</td><td>{size_mb:.2f} MiB</td></tr>"
        )

    exported_html = ""
    if export_rows:
        exported_html = f"""
        <h3 style="margin-top:18px;font-size:1.1rem;">Exported Artefacts</h3>
        <table class="info-table">
            <tr>
                <td><strong>Method</strong></td>
                <td><strong>Backend</strong></td>
                <td><strong>Path</strong></td>
                <td><strong>Size</strong></td>
            </tr>
            {"".join(export_rows)}
        </table>
        """

    return f"""
    <div class="section">
        <h2>Deployment Backends</h2>
        <table class="info-table">
            <tr><td><strong>Backend</strong></td><td><strong>Status</strong></td></tr>
            {"".join(rows)}
        </table>
        {exported_html}
    </div>
    """

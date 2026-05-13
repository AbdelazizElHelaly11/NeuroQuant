# Getting Started

NeuroQuant is published on PyPI as **`neuroquant`** and works on Python 3.10+.

## 1 · Install

=== "Standard (CPU)"

    ```bash
    pip install neuroquant
    ```

=== "GPU / CUDA"

    NeuroQuant follows the official PyTorch wheel index for CUDA builds.
    Install Torch first, then NeuroQuant on top:

    ```bash
    # CUDA 12.1 example — check pytorch.org for your driver/CUDA combo
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install neuroquant
    ```

=== "Editable / from source"

    For local development against the repo:

    ```bash
    git clone https://github.com/AbdelazizElHelaly11/NeuroQuant.git
    cd NeuroQuant
    pip install -e .
    ```

=== "With docs toolchain"

    To build this very site locally:

    ```bash
    pip install -e .[docs]
    mkdocs serve
    ```

## 2 · Optional extras

NeuroQuant keeps its core install small. Heavier deps that only a subset
of users need are exposed as extras:

| Extras            | Installs                                  | When you need it                                                               |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------ |
| `neuroquant[xai]` | `shap`                                    | Phase 3 SHAP attribution (Grad-CAM works without it).                          |
| `neuroquant[dev]` | `ruff`, `build`                           | Linting and wheel building for contributors.                                   |
| `neuroquant[docs]`| `mkdocs-material`, `mkdocstrings[python]` | Building / serving this documentation site.                                    |

Combine them with comma syntax: `pip install neuroquant[xai,dev]`.

## 3 · Verify the install

A clean install exposes a console-script *and* a flat Python API. Try both:

=== "Command line"

    ```console
    $ neuroquant --help
    usage: neuroquant [-h] [--config CONFIG] [--init] [--force] [--resume]
                     [--epochs EPOCHS] ...
    ```

=== "Python"

    ```pycon
    >>> from neuroquant import PTQQuantizer, __version__
    >>> __version__
    '2.0.0'
    >>> PTQQuantizer  # quantizer class is importable
    <class 'neuroquant.quantization.ptq.PTQQuantizer'>
    ```

If both work you're ready to go.

## 4 · Pick your front door

NeuroQuant has two coherent entry points — choose the one that matches
the kind of work you're doing:

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **CLI pipeline**

    ---

    For researchers who want a full, reproducible run (Pareto front,
    ONNX exports, MLflow, HTML report) from a single YAML.

    [:octicons-arrow-right-24: Continue to the CLI guide](pipeline_mode.md)

-   :material-package-variant:{ .lg .middle } **Python library**

    ---

    For developers integrating quantization into their own training
    or evaluation scripts — no YAML required.

    [:octicons-arrow-right-24: Continue to the library guide](library_mode.md)

</div>

## 5 · Troubleshooting

??? question "ImportError: cannot import name 'PTQQuantizer' from 'neuroquant'"

    Usually means you have a stale install from before the v2.0
    package restructure. Re-install fresh:

    ```bash
    pip uninstall -y neuroquant
    pip install --no-cache-dir neuroquant
    ```

??? question "ModuleNotFoundError: No module named 'torch'"

    NeuroQuant intentionally does **not** vendor a specific CUDA wheel
    of PyTorch — that decision belongs to you. Install Torch first
    (see the GPU tab above), then re-install NeuroQuant.

??? question "Why doesn't `import shap` work out of the box?"

    SHAP pulls in a heavy transitive dep tree, so we made it an
    optional extra. Install with `pip install neuroquant[xai]` if you
    want SHAP-based attribution. Grad-CAM is always available without
    the extra — and on detection / segmentation tasks the library
    automatically falls back to the gradient×input variant of SHAP
    that doesn't need the package.

??? question "`mkdocs serve` errors on Windows about an emoji extension"

    The Material theme uses `pymdownx.emoji` with twemoji generators
    declared via PyYAML `!!python/name:` tags. Newer PyYAML versions
    refuse to load those tags by default. Make sure you installed via
    `pip install -e .[docs]` (which pulls the correct extras) and that
    you're invoking the `mkdocs` shipped inside the same virtual
    environment.

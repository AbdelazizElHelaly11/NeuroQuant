# NeuroQuant

[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()
[![docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE)](https://AbdelazizElHelaly11.github.io/NeuroQuant/)

**Production-grade neural-network quantization framework — multi-objective NSGA search, ONNX deployment fidelity, and built-in explainability across classification, detection, segmentation, regression, and HuggingFace NLP.**

---

## Team Members
| Name | Student ID | Program |
| :--- | :--- | :--- |
| **Abdelaziz Elhelaly** | **202201827** | **Computer Science (CSAI)** |
| **Abdelwahab Hassan** | **202201281** | **Computer Science (CSAI)** |
| **Mostafa Nashaat** | **202202075** | **Computer Science (CSAI)** |
| **Abdelrahman Elsayed** | **202202049** | **Computer Science (CSAI)** |

**Supervisors:** 
- **Dr. Mohamed Fakhry Eldin Ghalwash (Main)**
- **Prof. Ahmed Abdelsamea (Joint)**

---

## Problem Statement

The deployment of state-of-the-art deep learning models in production environments and on edge devices is fundamentally hindered by their significant computational costs, memory footprint, and high inference latency. While quantization reduces these overheads by representing weights and activations in lower bitwidths (e.g., INT8), existing academic tools frequently fail in real-world scenarios. They often rely on simulated inference that masks actual hardware performance, lack comprehensive multi-objective optimization (balancing accuracy, true on-disk size, and hardware latency), and are opaque regarding how quantization degrades model interpretability. 

There is a critical need for a production-grade framework that addresses these gaps by delivering true INT8 artifacts, incorporating hardware-aware Pareto optimization, and integrating Explainable AI (XAI) to ensure both uncompromising performance and trustworthiness in deployment.

---

## Features

- **Comprehensive Quantization Arsenal:** Out-of-the-box support for PTQ, QAT, GPTQ, SmoothQuant, AWQ, SmoothQuant→GPTQ, and AdaRound.
- **Hardware-Aware Multi-Objective Search:** Utilizes Surrogate-assisted NSGA-II to optimize for model accuracy, true `.onnx` filesystem size, and real ONNX Runtime latency.
- **True Deployment Fidelity:** Emits real static-INT8 ONNX graphs rather than relying on FP32 simulations.
- **Explainable AI (XAI) Integration:** Built-in Grad-CAM, task-aware Grad-CAM, Attention Rollout (for ViTs), and SHAP for analyzing per-layer quantization error attribution.
- **Versatile Task Support:** Seamlessly quantizes models for classification, detection (Faster/Mask R-CNN, SSD, RetinaNet), segmentation, regression, and HuggingFace NLP tasks.
- **Strict Pipeline Determinism:** Guarantees reproducibility through seeded loaders, CuDNN deterministic flags, and robust `weights_only=True` checkpointing.

---

## System Architecture

NeuroQuant transforms a pre-trained PyTorch model into a deployable INT8 or mixed-precision artifact through a rigorous and deterministic 9-phase pipeline.

### High-Level AI/ML Pipeline
1. **P0 (Preparation):** FP32 baseline evaluation on the test split, establishing a true baseline.
2. **P1a (Sensitivity Analysis):** Hessian / Fisher per-layer sensitivity and 3-tier clustering.
3. **P1c (NSGA Search):** Surrogate-assisted NSGA-II multi-objective search (2- or 3-objective).
4. **P1d (AdaRound):** Canonical input→output weight rounding optimization.
5. **P1e (QAT):** Real Weight+Activation Quantization-Aware Training with FP32-teacher knowledge distillation.
6. **P1f (Advanced PTQ):** Implementation of GPTQ, SmoothQuant, AWQ, and SmoothQuant→GPTQ (INT4 + INT8 each).
7. **P2 (Pareto Analysis):** Generation of Pareto frontiers, accuracy vs. latency/size tradeoffs, and bitwidth distributions.
8. **P3 (Explainability):** Grad-CAM / Attention Rollout heatmaps + SHAP attribution for comprehensive model interpretability.
9. **P4 (Finalization):** MLflow logging, self-contained HTML report generation, and reproducibility manifest creation.

### Design Decisions & Scalability
- **Dual Interface:** Accessible via a fully automated CLI pipeline using a single YAML configuration, or as a flat Python library for config-free integration (`from neuroquant import PTQQuantizer`).
- **Data Fidelity:** Calibration relies strictly on inference-time "eval-transform" data pipelines rather than randomly augmented images to avoid distribution shifts.
- **Evaluation Integrity:** There is no data leakage between splits. The FP32 baseline and every method's headline metric are evaluated exclusively on the test split.

---

## Technologies Used

- **Deep Learning Framework:** PyTorch, Torchvision, Torchaudio
- **Inference & Deployment:** ONNX, ONNX Runtime (`onnxruntime`), ONNX Script
- **Optimization & Search:** Pymoo (NSGA-II), Scikit-Learn
- **Experiment Tracking:** MLflow
- **Explainable AI:** SHAP, Custom Grad-CAM/Attention Rollout implementations
- **Configuration Management:** Pydantic v2, PyYAML
- **Data Handling:** Pandas, HuggingFace `datasets` & `transformers`
- **Visualization:** Matplotlib, Seaborn
- **Documentation:** MkDocs Material

---

## Setup Instructions

### Environment Requirements
- **OS:** Linux, Windows, or macOS
- **Python:** 3.10, 3.11, or 3.12
- **Hardware:** NVIDIA GPU with CUDA 11.8/12.x is highly recommended for QAT and GPTQ processing, though CPU fallback is supported.

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/AbdelazizElHelaly11/NeuroQuant.git
   cd NeuroQuant
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install PyTorch:**
   Ensure you install the CUDA-enabled version of PyTorch if using a GPU.
   ```bash
   # Example for CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install NeuroQuant:**
   ```bash
   # Install the package in editable mode with development dependencies
   pip install -e ".[dev]"
   ```
   *Note on Optional Extras:* Install with `pip install -e ".[xai,nlp]"` if you require SHAP attribution or HuggingFace NLP support.

---

## Deployment Instructions

NeuroQuant outputs directly deployable ONNX models.

1. **Initialize the Pipeline Configuration:**
   ```bash
   neuroquant --init
   ```
   This generates a `config.yaml` file in the current directory.

2. **Execute the Pipeline:**
   ```bash
   neuroquant --config config.yaml --epochs 20
   ```
   *(To resume an interrupted run safely without discarding checkpoints, append the `--resume` flag).*

3. **Artifact Retrieval:**
   Upon completion, navigate to the `artifacts/` directory.
   - Deployable INT8 graphs are stored in `artifacts/onnx/`.
   - Comprehensive metrics, hardware look-up tables (`latency_lut.json`), and reproducibility manifests are found alongside them.
   - The interactive `neuroquant_report.html` serves as your deployment dashboard.

---

## Usage Guide

### 1. Using the Python Library (Direct Integration)
For developers integrating quantization within existing training scripts, NeuroQuant works config-free.

```python
from neuroquant import PTQQuantizer
import torchvision.models as models

# Load your model and calibration data loader
model = models.resnet18(weights='DEFAULT')
calib_loader = ... # Your DataLoader

# Initialize and run config-free post-training quantization
ptq = PTQQuantizer(model)
q_model = ptq.quantize(calib_loader, bitwidth=8)
```

### 2. Using the CLI (Research & Automated Sweeps)
For a fast smoke test on CPU without training:
```bash
neuroquant --config config.yaml --epochs 0 --device cpu \
  --phases phase_0_preparation phase_1a_hessian_clustering phase_1c_nsga_search
```

**Customizing Configurations (`config.yaml`):**
```yaml
model:
  name: resnet18
  task: classification

dataset:
  name: cifar10
  batch_size: 128

hyperparams:
  hardware_aware_search: true
  onnx_export_enabled: true
```

*For comprehensive API documentation and detailed usage workflows, visit the [NeuroQuant Documentation Site](https://AbdelazizElHelaly11.github.io/NeuroQuant/).*

---
*This project was developed in partial fulfillment of the requirements for the Degree of Bachelor of Science in CSAI at Zewail City of Science and Technology.*

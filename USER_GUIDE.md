# LENS-XAI Quickstart & Operations Guide

## 1. Environment Setup

### 1.1 Prerequisites
Ensure you are running Python 3.9+ and have navigated to the project root directory.

### 1.2 Installation
Run the following command to install the required PyTorch, scikit-learn, and XAI dependencies:
```bash
pip install -r requirements.txt
```

*(Optional)* Ensure Graphviz and SHAP/LIME are properly configured in your environment for visualizing the Neural Network's explanations.

---

## 2. Dataset Preparation

LENS-XAI is designed for the NSL-KDD benchmark out-of-the-box, simulating real-world data scarcity by enforcing a rigid 10% Training / 90% Testing paradigm.

**Execute:**
```bash
python src/data/make_dataset.py
```
**Expected Output:** This script dynamically processes raw `.csv` captures (handling mixed numeric/categorical features) and generates analysis-ready numpy arrays (`.npy`) in the `data/processed/nsl-kdd/` folder.

---

## 3. Running the Base System (Phase 1 & 2)

Validate the initial Knowledge Distillation and VAE core algorithms.

**Execute:**
```bash
python demo_phase1_2.py
```

**Key Operations:**
1. **Teacher Training**: The complex parent network trains on the sparse 10% data pool.
2. **Knowledge Distillation**: The ultra-lightweight Student network (MLP) learns via the Teacher's soft labels (KL-Divergence).
3. **Evaluation**: Both networks are benchmarked against the massive 90% testing set to confirm the Student achieves comparable (or better) accuracy at a fraction of the parameter count.

---

## 4. Running the Advanced Network Security Suite (Phase 3 & 4)

Execute the comprehensive demonstration encompassing all high-level cybersecurity features.

**Execute:**
```bash
python demo_phase3.py
```

**Expected Telemetry:**
* **Federated Learning Segment**: You will observe the console instantiate 3 mock edge clients processing localized data before aggregating via FedAvg on a central server.
* **Adversarial Resiliency Segment**: The system evaluates its vulnerability against an FGSM attack (with Epsilon=0.05), notes the accuracy drop, and performs adversarial retraining to recover.
* **Continual Learning Segment**: Using EWC (Elastic Weight Consolidation), the model restricts critical weights from "forgetting" past traffic flows when introduced to a new task.
* **Edge Optimization**: The FP32 Student network is dynamically quantized to INT8, shrinking the memory footprint natively for CPU deployment.
* **XAI Analytics**: A SHAP or Integrated Gradients pipeline runs over the processed data. The result is saved locally as an interpretability plot mapping feature attributions.

---

## 5. Using the Interactive Notebooks

For research analysts who prefer visual cell-by-cell execution, launch the Jupyter environment:
```bash
jupyter notebook
```

Navigate to `notebooks/03_lens_xai_full_pipeline.ipynb`. This notebook provides a rich, interactive environment mapping out every step from data ingest to SHAP feature contribution plots.

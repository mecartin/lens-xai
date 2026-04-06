# LENS-XAI — Step-by-Step Walkthrough

This guide explains how to run the LENS-XAI project from start to finish, demonstrating the baseline Knowledge Distillation architecture (Phase 1 & 2) as well as the advanced features like Federated Learning, Adversarial Robustness, and Incremental Learning (Phase 3 & 4).

---

## Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional but recommended) Ensure `shap` and `lime` are correctly installed for the XAI visualizations.

---

## Step 1: Dataset Preparation

The pipeline requires pre-processed numpy arrays for training. We use the NSL-KDD benchmark dataset for quick demonstrations.

**Command:**
```bash
python src/data/make_dataset.py
```

**What happens:**
- Reads the raw network traffic CSVs from `data/raw/nsl-kdd/`.
- Performs feature standardization, one-hot encoding for categorical variables, and label binarization.
- Splits the data into a strict 10% Training / 90% Testing paradigm (simulating real-world security constraints).
- Saves the processed `.npy` arrays to `data/processed/nsl-kdd/`.

---

## Step 2: The Baseline Pipeline (Phases 1 & 2)

This step demonstrates the core concept of the paper: Training a heavy "Teacher" Model (with Attention and a Hierarchical VAE) and distilling its knowledge into a lightweight "Student" Model (an MLP) for edge devices.

**Command:**
```bash
python demo_phase1_2.py
```

**What to expect:**
- **Model Initialization:** Prints the massive parameter difference between the Teacher (~100k+ params) and the Student (~10k params; >80% reduction).
- **Hard Training:** The Teacher trains on the 10% target dataset.
- **Distillation:** The Student trains using a combined Cross-Entropy (hard labels) and KL-Divergence (soft labels from Teacher) loss.
- **Evaluation:** Both evaluate on the massive 90% test set. You will observe the Student achieving nearly identical (or sometimes slightly better) accuracy to the Teacher due to the regularization effect of Knowledge Distillation.

---

## Step 3: Advanced Capabilities (Phases 3 & 4)

This step demonstrates the end-to-end integration of robust cybersecurity and local device deployment features.

**Command:**
```bash
python demo_phase3.py
```

**What to expect line-by-line in the console:**

1. **Federated Learning:**
   - The data is partitioned across 3 mock edge clients.
   - Each client trains locally.
   - The `FederatedServer` aggregates their weight matrices using **FedAvg** and applies a Differential Privacy (DP) Gaussian noise budget.
   - The Global Model accuracy is reported.

2. **Adversarial Defense:**
   - The system evaluates the model against an **FGSM** (Fast Gradient Sign Method) evasion attack at an epsilon of 0.05. Accuracy typically drops.
   - The `AdversarialTrainer` then conducts an epoch of robust training, feeding the network a 50/50 mix of clean and adversarially-perturbed traffic.
   - Post-defense robustness is re-evaluated and will show a marked recovery against the attack.

3. **Incremental Learning (EWC):**
   - The model trains on "Task A" (a specific temporal window or dataset subset).
   - Knowledge is consolidated by computing the **Fisher Information Matrix** mapping the importance of every neural weight to Task A. A Replay Buffer samples old data.
   - The model trains on "Task B".
   - Normally, this causes *catastrophic forgetting* of Task A. However, the EWC penalty restricts the movement of critical Task A weights.
   - Task A retained accuracy is printed (typically >90% retention).

4. **Edge Quantization:**
   - The Student Model represents the deployable Edge asset.
   - The PyTorch Dynamic Quantization API converts the FP32 Linear layers into **INT8**.
   - The console prints the physical memory footprint reduction (e.g., 0.5 MB -> 0.1 MB) and benchmarks the CPU inference latency per batch (milliseconds).

5. **XAI Interpretability:**
   - If installed, the script initializes a `SHAPExplainer`.
   - It computes Shapley values to identify which network packet features (e.g., `src_bytes`, `duration`, `flag`) had the highest impact on the IDS's decision pipeline.
   - Generates a local file named `shap_demo_plot.png` visualizing these contributions.

---

## Directory Reference

- `src/models/networks.py` — Teacher (Attention + H-VAE) & Student (MLP) architectures
- `src/models/train.py` — Main Knowledge Distillation training loop
- `src/federated/federated_trainer.py` — Decentralized FedAvg simulation
- `src/models/adversarial.py` — FGSM/PGD attacks and Robust Training
- `src/models/incremental.py` — EWC computation and Replay buffering
- `src/models/quantize.py` — INT8 deployment reduction
- `src/utils/metrics.py` & `src/utils/xai.py` — Reporting and SHAP/LIME analytics

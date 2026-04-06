# LENS-XAI System Architecture & Working Document

## 1. Introduction

LENS-XAI (Lightweight Explainable Network Security through Knowledge Distillation and Variational Autoencoders) is an advanced intrusion detection system (IDS) built to secure resource-constrained IoT/IIoT networks. The system bridges the gap between high-performance deep learning models and low-power edge devices.

It addresses key challenges:
- **Resource Constraints**: Reduces model size by 40-60% while maintaining >95% accuracy.
- **Black-Box AI**: Employs XAI to make predictions transparent.
- **Evolving Threats**: Supports incremental learning and adversarial defenses without full retraining.
- **Distributed Networks**: Implements federated learning to preserve privacy across geographically distributed devices.

---

## 2. Core Architecture

The architecture consists of two primary networks trained via a **Teacher-Student Paradigm** (Knowledge Distillation).

### 2.1 The Teacher Model (Heavyweight)
The Teacher is a robust, complex model designed to learn intricate patterns from the raw network traffic data. It is run exclusively on high-performance servers during the training phase.
* **Component 1: Hierarchical Variational Autoencoder (H-VAE)**
  * Captures multi-scale latent representations (packet, flow, and session levels).
  * Excellent for detecting subtle anomalies and zero-day attacks by modeling the underlying distribution of normal network behavior.
* **Component 2: Attention Module**
  * Dynamically weights the input features, focusing on the most critical packets/flags indicative of a cyber-attack.
* **Component 3: Classifier**
  * Outputs the final intrusion detection probabilities.

### 2.2 The Student Model (Lightweight)
The Student is a highly compressed Multi-Layer Perceptron (MLP) designed for deployment on edge devices (e.g., Raspberry Pi, IoT gateways).
* **Knowledge Distillation**: The student is trained not just on the raw data, but on the "soft labels" (probability distributions) produced by the Teacher. This allows the Student to mimic the Teacher's advanced reasoning capabilities despite having >80% fewer parameters.
* **Quantization**: The Student model is converted from FP32 (32-bit floating point) to INT8 (8-bit integer) representation using PyTorch's dynamic quantization API. This further dramatically reduces the physical memory footprint and speeds up CPU inference.

---

## 3. Advanced Capabilities & Working Mechanisms

### 3.1 Federated Learning (Distributed Training)
* **Goal**: Train the IDS across multiple organizations or edge devices without centralizing sensitive raw data.
* **Mechanism**: 
  1. The Global Server sends the current model weights to Edge Clients.
  2. Edge Clients train locally on their proprietary network traffic.
  3. Clients send only the updated *gradients/weights* back to the Server.
  4. The Server aggregates these updates using the **FedAvg** algorithm.
  5. Differential Privacy (DP) Gaussian noise is injected during aggregation to prevent reverse-engineering of client data.

### 3.2 Adversarial Robustness Layer
* **Goal**: Defend the IDS against evasion attacks crafted by malicious actors.
* **Mechanism**:
  1. The system utilizes algorithms like **FGSM** (Fast Gradient Sign Method) or **PGD** (Projected Gradient Descent) to generate adversarial examples (network traffic with imperceptible, malicious perturbations).
  2. The model undergoes *Adversarial Training*, where it is explicitly fed a mix of clean and adversarial traffic, forcing it to learn a decision boundary that is resilient against tampering.

### 3.3 Incremental Learning (Continual Learning)
* **Goal**: Learn new threat profiles sequentially without forgetting previously learned attacks (*Catastrophic Forgetting*).
* **Mechanism**:
  1. **Elastic Weight Consolidation (EWC)**: Calculates the Fisher Information Matrix to identify which neural weights are critical for past tasks. It penalizes the network for changing these specific weights when learning a new task.
  2. **Replay Buffer**: Stores a tiny, representative subset of old data to occasionally mix with new data, reinforcing past knowledge.

### 3.4 Explainable AI (XAI) Module
* **Goal**: Ensure network administrators can trust and verify the DL-IDS decisions.
* **Mechanism**:
  1. Integrates **SHAP** (SHapley Additive exPlanations) and **LIME**.
  2. Calculates feature attributions, revealing precisely *which* features of a network packet (e.g., source bytes, connection duration, protocol type) triggered the classification of an intrusion.

---

## 4. End-to-End Execution Flow

1. **Pre-processing**: Raw network CSVs undergo standard scaling, one-hot encoding, and label binarization. The dataset is split rigidly into 10% Training and 90% Testing to mimic data-scarce real-world conditions.
2. **Teacher Training**: The H-VAE + Attention model is trained on the 10% dataset.
3. **Knowledge Distillation**: The Student MLP observes the Teacher and trains on a combined loss function (Cross-Entropy for hard labels + KL-Divergence for soft labels).
4. **Federated & Adversarial Tuning**: The model is distributed via FedAvg and hardened against FGSM perturbations.
5. **Incremental Tasking**: The model learns sequential temporal windows via EWC.
6. **Edge Deployment**: The final Student model undergoes INT8 quantization and is embedded into the simulated IoT device.
7. **XAI Interpretability**: The `xai.py` module visualizes the decision payload for cybersecurity operators.

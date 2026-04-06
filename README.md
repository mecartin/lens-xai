# LENS-XAI
**Lightweight Explainable Network Security through Knowledge Distillation and Variational Autoencoders**


---

## 1. Base Paper Overview

| Field | Details |
| :--- | :--- |
| **Paper Title** | LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and Variational Autoencoders for Scalable Intrusion Detection in Cybersecurity |
| **Authors** | Muhammet Anil Yagiz & Polat Goktas |
| **Institution** | Kirikkale University & University College Dublin |
| **Year / Status** | 2025 \| Submitted — Applied Soft Computing |
| **arXiv** | arXiv:2501.00790v2 |

**Key Innovations**
* Combines Knowledge Distillation + Variational Autoencoders (VAE) for lightweight Intrusion Detection Systems (IDS)
* Achieves 95–99% accuracy across 4 benchmark datasets
* Variable attribution-based Explainable AI (XAI) for transparency in predictions
* Uses only 10% training data and 90% testing data to simulate real-world constraints

## 2. Research Gaps Identified

| Gap | Description |
| :--- | :--- |
| **Black-Box Models** | Lack of interpretability in DL-based IDS undermines trust in critical applications |
| **Resource Constraints** | High computational demands limit deployment in IoT/edge environments |
| **Dataset Imbalance** | Static feature selection fails to handle imbalanced and evolving datasets |
| **Limited Adaptability** | Insufficient dynamic threat adaptation for real-time evolving cyberattacks |
| **Generalization Issues** | Dataset-specific models with limited cross-domain applicability |
| **Real-Time Performance** | Computational overhead restricts real-time intrusion detection capability |

## 3. Objectives of the Proposed System

| # | Objective | Description |
| :--- | :--- | :--- |
| 1 | **High Detection Accuracy** | Achieve >95% accuracy across diverse attack types and datasets |
| 2 | **Computational Efficiency** | Reduce model complexity by 40–60% through knowledge distillation |
| 3 | **Enhanced Explainability** | Provide transparent feature attribution for trust and validation |
| 4 | **Scalable Architecture** | Deploy efficiently in resource-constrained IoT/IIoT environments |
| 5 | **Real-World Robustness** | Handle imbalanced datasets and rare attack scenarios effectively |

## 4. Novelty & Proposed Contributions

### 4.1 Paper Contributions
**Knowledge Distillation + VAE Integration**
* Teacher-student architecture for model compression
* VAE learns latent representations for anomaly detection
* Maintains 95%+ accuracy with 40–60% fewer parameters

**Variable Attribution-Based XAI**
* Local accuracy property ensures transparent predictions
* Feature contribution decomposition at instance level
* Conditional expectation-based marginal contributions

**Strategic Data Partitioning**
* 10% training / 90% testing split simulates real-world constraints
* Tests generalization under limited labeled data
* Validates performance in data-scarce scenarios

### 4.2 Proposed Enhancements

| Enhancement | Mechanism | Impact |
| :--- | :--- | :--- |
| **Incremental Learning Module** | Continuous model updates without full retraining | Adapt to new attack patterns in real-time |
| **Federated Learning Integration** | Distributed training across multiple IoT nodes | Preserve privacy while leveraging collective intelligence |
| **Attention-Based Feature Selection** | Dynamic feature weighting using self-attention mechanisms | Improve detection of rare and evolving attacks |
| **Edge-Optimized Quantization** | 8-bit/4-bit model quantization for ultra-low-power devices | Deploy on resource-constrained edge devices |
| **Multi-Scale Anomaly Detection** | Hierarchical VAE for packet, flow, and session-level analysis | Capture anomalies at different temporal granularities |
| **Adversarial Robustness Layer** | Defense against adversarial ML attacks on IDS | Ensure resilience against evasion techniques |

## 5. System Architecture

### 5.1 Existing System (Base Paper)
The existing LENS-XAI pipeline follows a teacher-student knowledge distillation paradigm:
* Raw network traffic data is collected and preprocessed (normalization, encoding)
* A Teacher Model (VAE + Classifier) is trained on the full feature set
* Knowledge Distillation transfers compressed representations to a lightweight Student Model
* The Student Model performs real-time Intrusion Detection

**Known Limitations:**
* No incremental learning — requires full retraining for new attack types
* Limited adversarial robustness against evasion attacks
* Single-scale analysis misses multi-granularity anomalies
* Requires centralized training — privacy concern for distributed deployments

### 5.2 Proposed Enhanced System
The proposed architecture introduces six major enhancements over the base system:
* Multi-Source Data Streams feed into a Preprocessing + Attention module
* Attention-based preprocessing dynamically selects the most informative features
* Hierarchical VAE (Multi-Scale) captures anomalies at packet, flow, and session levels
* Federated Aggregation enables privacy-preserving distributed training across IoT nodes
* Teacher Model augmented with Adversarial Defense for robustness
* Quantized Student Model for ultra-low-power edge deployment
* Incremental Learning Module allows real-time model updates
* Enhanced XAI Module provides richer, multi-scale explanations


## 7. Expected Outcomes

### 7.1 Performance Targets

| Metric | Target |
| :--- | :--- |
| Detection Accuracy | >96% across all benchmark datasets |
| False Positive Rate | <2% |
| Inference Time | <10ms per batch |
| Model Size Reduction | 50–60% vs baseline LENS-XAI |

### 7.2 Novel Contributions
* First federated IDS with hierarchical VAE architecture
* Edge-optimized quantized models for IIoT deployment
* Real-time incremental learning capability for evolving threats
* Adversarially robust XAI framework with enhanced explanations

### 7.3 Practical Impact
* Deployable on resource-constrained IoT/IIoT devices
* Privacy-preserving distributed training across federated nodes
* Adaptive to evolving threat landscapes without full retraining
* Transparent decision-making for network security operators

## 8. Benchmark Datasets

| Dataset | Samples | Features | Attack Types | Achieved Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Edge-IIoTset | 2,200,000 | 61 | 14 attack types | 95.34% |
| UKM-IDS20 | 12,887 | 46 | 8 types (UAV) | 99.92% |
| CTU-13 | 92,212 | Flow-based | Botnet traffic | 98.42% |
| NSL-KDD | 148,517 | 41 | 5 categories | 99.34% |

## 9. Comparative Analysis

| Method | Accuracy | Parameters | XAI | Edge | Federated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RNN-IDS | 81.29% | High | No | No | No |
| MCNN-DFS | 81.44% | High | No | No | No |
| bot-DL | 96.60% | Medium | No | No | No |
| **LENS-XAI (Paper)** | 95–99% | Medium | Yes | Yes | No |
| **LENS-XAI (Proposed)** | 96–99% | Low | Enhanced | Enhanced | Yes |

*Note: "Enhanced" denotes capabilities significantly improved beyond the base paper implementation.*

## 10. Conclusion
LENS-XAI provides a strong foundation for lightweight, explainable intrusion detection. The expanded implementation addresses the critical gaps of adaptability, privacy, and robustness:

* **Federated Learning (FedAvg)** enables scalable, privacy-preserving deployment across distributed IoT edges.
* The explicit **Adversarial Robustness Layer (FGSM/PGD)** ensures resilience against evasion attacks.
* The **Incremental Learning Module (EWC + Replay Buffer)** enables real-time adaptation without catastrophic forgetting.
* The **Edge Quantization (INT8)** reduces model footprint for IoT deployment by >50%.
* The **XAI Module (SHAP + LIME)** provides unmatched interpretability for security analysts.

## 11. References
**Base Paper**
* Yagiz, M. A., & Goktas, P. (2025). LENS-XAI: Redefining Lightweight and Explainable Network Security through Knowledge Distillation and Variational Autoencoders for Scalable Intrusion Detection in Cybersecurity. Applied Soft Computing (Submitted). arXiv:2501.00790v2

**Datasets**
* Edge-IIoTset: https://ieee-dataport.org/documents/edge-iiotset
* UKM-IDS20: https://kaggle.com/datasets/muatazsalam/ukm-ids20
* CTU-13: https://stratosphereips.org/datasets-ctu13
* NSL-KDD: https://kaggle.com/datasets/hassan06/nslkdd

**Related Work**
* Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
* Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes.
* Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
* Ribeiro, M. T., et al. (2016). Why Should I Trust You? Explaining the Predictions of Any Classifier (LIME).

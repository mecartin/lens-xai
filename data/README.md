# LENS-XAI Network Security Datasets

This directory houses the foundational datasets used for training and evaluating the Knowledge Distilled, Edge-optimized IDS (Intrusion Detection System). 

## 1. Directory Structure
- `raw/`: Unaltered CSV/PCAP flow extracts pulled directly from public sources. This is immutable raw data.
- `processed/`: The result of running `src/data/make_dataset.py`. Features are standardized, categorical variables are one-hot encoded, and labels are binarized. Includes `.npy` arrays split into a rigid **10% Training** and **90% Testing** paradigm.

---

## 2. Benchmark Datasets Covered

### 2.1 NSL-KDD (Core Demonstration Set)
* **Description:** An improved, condensed successor to the classic KDDCup99 dataset. It removes redundant records to strictly evaluate the generalized detection capabilities of learning algorithms against DoS, Probe, R2L, and U2R attacks.
* **Samples:** ~148,517 (Train/Test combined limit)
* **Features:** 41 raw traffic features (Protocol, Src/Dst bytes, connection flags).
* **Reference:** [kaggle.com/datasets/hassan06/nslkdd](https://kaggle.com/datasets/hassan06/nslkdd)

### 2.2 Edge-IIoTset
* **Description:** A highly modern (2022) cybersecurity dataset mapped specifically to the Internet of Things (IoT) and Industrial IoT. Features telemetry from flame sensors, water level systems, and smart home modules. Excellent for validating the INT8 Quantized models.
* **Attack Types:** Ransomware, DoS, DDoS, MITM, Injection (14 types).
* **Reference:** [ieee-dataport.org/documents/edge-iiotset](https://ieee-dataport.org/documents/edge-iiotset)

### 2.3 CTU-13
* **Description:** A dataset of Botnet traffic captured at the CTU University. The data provides extremely imbalanced scenarios (millions of normal flows vs a few thousand botnet flows) representing realistic enterprise monitoring conditions.
* **Focus Topic:** Botnet lifecycles and Command & Control traffic profiling.
* **Reference:** [stratosphereips.org/datasets-ctu13](https://stratosphereips.org/datasets-ctu13)

### 2.4 UKM-IDS20
* **Description:** Represents traffic scenarios tied specifically to UAV (Unmanned Aerial Vehicle) / Drone networks.
* **Focus Topic:** Edge-node hacking and localized wireless network perimeter attacks.

---

## 3. Data Processing Mechanism
If for any reason the `processed/` arrays are lost, you can regenerate them purely from the `raw/` CSV files using the data processing module:
```bash
python src/data/make_dataset.py
```

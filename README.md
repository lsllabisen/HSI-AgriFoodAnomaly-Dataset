# HSI-AgriFoodAnomaly

<div align="center">
  <img src="Figures/Convoyeur3.png" alt="HSI-AgriFoodAnomaly overview" width="80%">
</div>

**HSI-AgriFoodAnomaly** is the first open-access hyperspectral dataset specifically designed for **anomaly detection in industrial agri-food production environments**. The dataset contains 147 annotated hyperspectral image cubes acquired under realistic conveyor-based conditions, with a wide variety of **visually ambiguous and challenging anomalies** (e.g., plastic, textile, metal, glass, paper, wood, mineral).

This repository includes:
- The full dataset with RGB projections and pixel-level annotations.
- A baseline deep learning pipeline for anomaly detection using 2D CNNs adapted to 300 HSI-band as input.
- Scripts for patch extraction, training, evaluation, and visualization.

---

## Dataset Overview

The dataset was constructed through a carefully controlled acquisition pipeline that simulates industrial food inspection. It captures a wide variety of **realistic and challenging anomalies** placed in **diverse, non-trivial scenes**. Each hyperspectral cube is annotated at the pixel level, allowing the training of models for multiple computer vision tasks.

**Use case:** Inline inspection of oat-based food products (e.g., oat flakes with chocolate chips), simulating real-world production conditions (occlusion, clutter, varying densities).

**Anomaly types:**  
- Plastics (rigid fragments and films)  
- Textiles (cotton, mesh, threads)  
- Metals (nails, foil, clips, sponges)  
- Paper (kraft, printed, glossy)  
- Wood/plant fragments  
- Minerals/glass (stones, shards)
- Mixed anomaly scenes
- Anomaly-free samples
- New anomalies

**Annotations:**  
- Pixel-level binary masks  


**Setup Description:**  

| Property             | Value                          |
|----------------------|--------------------------------|
| Number of scenes     | 147 HSI cubes                  |
| Spectral range       | 400–1000 nm (VNIR)             |
| Spectral bands       | 300                            |
| Spatial resolution   | 1000 × 900 pixels              |
| Acquisition setup    | Conveyor-based pushbroom system|
| Formats              | `.bil`, `.bil.hdr`, `.png`     |
---


**Dataset Split Summary:**  


| Category                              | Train | Val | Test | Total |
|---------------------------------------|:-----:|:---:|:----:|:-----:|
| Textile and fiber-based materials     |  40   |  8  |  16  |  64   |
| Plastics                              |  20   |  4  |   8  |  32   |
| Paper-based materials                 |   5   |  1  |   2  |   8   |
| Metals                                |   5   |  1  |   2  |   8   |
| Wood, plant-based, minerals, glass    |   5   |  1  |   2  |   8   |
| Mixed (multi-object scenes)          |   5   |  1  |   2  |   8   |
| Normal (anomaly-free)                |   9   |  1  |   3  |  13   |
| For inference only                    |   –   |  –  |   5  |   5   |
| New anomaly objects (OOD)            |   –   |  –  |   1  |   1   |
| **Total**                             | **89**|**17**|**41**|**147**|


**Dataset Download:**

> **Coming Soon:** Dataset will be downloadable from [data.gouv.fr link]  
> A guide will be provided for downloading, verifying integrity, and organizing the files.

**Dataset Structure:**

> To Do

## 🧪 Baseline Method

We provide a simple but effective **real-time anomaly detection pipeline** using standard 2D convolutional neural networks (CNNs) adapted to hyperspectral inputs.

- CNN models used: MobileNetV2, ResNet18/50, TinyNet, EfficientNet, MixNet
- Input patch sizes: 300x100×100, 300x200×200, 300x300×300
- Binary classification: with anomaly vs. without anomaly
- Performance reported using accuracy, F1-score, AUC, MCC
- Comparison with RGB-only models also included

<div align="center">
  <img src="Figures/training2.png" alt="Baseline Method" width="90%">
</div>

## How to Run

> Python 3.8+ and a CUDA-enabled GPU are recommended.

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

#### 2. Extract patches from annotated HSI cubes

```bash
nohup python3 run_data_patching.py
```

#### 3. Train a CNN model

```bash
nohup python3 run_train.py
```

---

## 🔗 Citation

If you use this dataset or code, please cite our work:

```bibtex
@article{bechar2025hsiagrifoodanomaly,
  title={HSI-AgriFoodAnomaly: An Open Hyperspectral Benchmark for Realistic Anomaly Detection in Food Production},
  author={Bechar, Mohammed El Amine},
  journal={Computers and Electronics in Agriculture},
  year={2025}
}

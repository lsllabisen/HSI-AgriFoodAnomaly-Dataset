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

## Overview

**Use case:** Inline inspection of oat-based food products (e.g., oat flakes with chocolate chips), simulating real-world production conditions (occlusion, clutter, varying densities).

**Anomaly types:**  
- Plastics (rigid fragments and films)  
- Textiles (cotton, mesh, threads)  
- Metals (nails, foil, clips, sponges)  
- Paper (kraft, printed, glossy)  
- Wood/plant fragments  
- Minerals/glass (stones, shards)

**Annotations:**  
- Pixel-level binary masks  
- RGB projection images  
- Anomaly-free samples included  

**Dataset Description:**  

| Property             | Value                          |
|----------------------|--------------------------------|
| Number of scenes     | 147 HSI cubes                  |
| Spectral range       | 400–1000 nm (VNIR)             |
| Spectral bands       | 300                            |
| Spatial resolution   | 1000 × 900 pixels              |
| Acquisition setup    | Conveyor-based pushbroom system|
| Formats              | `.bil`, `.bil.hdr`, `.png`     |
---


## 🧪 Baseline Method

We provide a simple but effective **real-time anomaly detection pipeline** using standard 2D convolutional neural networks (CNNs) adapted to hyperspectral inputs.

- CNN models used: MobileNetV2, ResNet18/50, TinyNet, EfficientNet, MixNet
- Input patch sizes: 100×100, 200×200, 300×300
- Binary classification: anomaly vs. clean
- Performance reported using accuracy, F1-score, AUC, MCC
- Comparison with RGB-only models also included

> 🔧 Full code and training instructions will be provided soon.

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

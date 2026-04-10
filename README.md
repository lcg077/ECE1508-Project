# Graph Neural Networks for Traffic Flow Prediction

ECE1508 Applied Deep Learning — Course Project (Winter 2026)

**Authors:** Chenguang Li, Mingchen Li — University of Toronto

---

## Overview

This project implements and evaluates deep learning models for short-term traffic speed forecasting on the [METR-LA](https://github.com/liyaguang/DCRNN) dataset. We progressively build from temporal baselines to graph-based spatio-temporal models, investigating how incorporating road network topology improves prediction accuracy.

### Models Implemented

| Model | Description |
|---|---|
| LSTM | Two-layer LSTM baseline (temporal only) |
| TCN | Temporal Convolutional Network baseline (temporal only) |
| STGCN (no graph) | Ablation: STGCN with identity adjacency matrix |
| STGCN (first-order) | STGCN with first-order graph convolution |
| STGCN (Cheb K=3) | STGCN with Chebyshev polynomial graph convolution |

### Key Results (METR-LA, mean over 4 random seeds)

| Model | MAE (15min) | MAE (30min) | MAE (60min) |
|---|---|---|---|
| LSTM | 3.656 | 3.732 | 3.944 |
| TCN | 2.992 | 3.652 | 4.617 |
| STGCN (first-order) | 3.019 | 3.625 | 4.508 |
| **STGCN (Cheb K=3)** | **2.915** | **3.438** | **4.188** |

---

## Dataset

**METR-LA**: Traffic speed recordings from 207 loop detectors on the Los Angeles highway network, collected at 5-minute intervals (March–June 2012).

Download the dataset (`ECE1508Dataset.zip`) and place it in the same directory as the notebook before running. The notebook will automatically extract and load the data.

---

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tables`

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload `ECE1508Project.ipynb` and `ECE1508Dataset.zip` to Google Colab
2. Set runtime to **GPU (T4 or A100)**
3. Run all cells sequentially

### Option 2: Local

```bash
git clone https://github.com/lcg077/ECE1508-Project.git
cd ECE1508-Project
pip install -r requirements.txt
jupyter notebook ECE1508Project.ipynb
```

---

## Repository Structure

```
ECE1508-Project/
├── ECE1508Project.ipynb        # Main notebook
├── ECE1508Dataset.zip          # METR-LA dataset
├── README.md
├── requirements.txt
└── seeds/                      # Multi-seed experiment results
    ├── ECE1508ProjectSeed0.ipynb
    ├── ECE1508ProjectSeed7.ipynb
    ├── ECE1508ProjectSeed42.ipynb
    └── ECE1508ProjectSeed123.ipynb
```

The `seeds/` folder contains independent runs with different random seeds (0, 7, 42, 123) to verify result stability. All reported metrics are averaged across these four runs.

---

## Reference

> Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. *IJCAI 2018*. [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)

# 🚀 Adaptive Attention Project: Quickstart

This repository contains the implementation, training, and evaluation of an **Adaptive Attention** mechanism for Chronos-2.

## 🛠️ Main Scripts

1.  **`train_universal.py`**: Unified training script for all datasets.
2.  **`evaluate_paper_metrics.py`**: Unified, paper-aligned evaluation script.
3.  **`load_fev_datasets.py`**: Unified data loader for all benchmark datasets.
4.  **`adaptive_attention.py`**: The core implementation of the Adaptive Attention mechanism.

---

## 📈 How to Train

To train the Adaptive model on any of the supported datasets (ETTm1, ETTh1, electricity, epf_de, hospital, rossmann):

```bash
# Example: Training on ETTm1
python train_universal.py --dataset ETTm1 --steps 10000 --device mps
```

*Note: Use `--device mps` for Apple Silicon GPU, or `--device cuda` for NVIDIA.*

---

## 📊 How to Evaluate

To run the paper-aligned evaluation (SQL Skill Score, MASE Skill Score, MAE) and statistical testing:

```bash
# Example: Evaluating ETTm1
python evaluate_paper_metrics.py --dataset ETTm1 --device mps
```

This will:
- Load the baseline Chronos-2.
- Load your best trained checkpoint from `./checkpoints/`.
- Perform a **paired t-test** across 50 test windows.
- Save result JSONs to `./results/`.

---

## 📁 Project Structure

- `checkpoints/`: Storage for `.pt` model weights.
- `results/`: JSON metrics and PNG visualization plots.
- `logs/`: Training logs for each run.

---

## ✅ Prerequisites

Install dependencies:
```bash
pip install -r Req.txt
```
*(Dependencies include: torch, chronos-forecasting, autogluon.bench, scipy, matplotlib)*

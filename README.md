# Chronos-2 Baseline Experiment

This repository is part of a larger project focused on extending **Chronos-2**.
My task was to first **run Chronos-2 locally** and **verify its functionality** on a benchmark dataset used in the original paper.

## What was done
- Executed the pre-trained **Amazon Chronos-2** model locally
- Tested the model on the **ETT_15T** dataset from `autogluon/fev_datasets`
- Performed basic preprocessing and time-series forecasting
- Compared model predictions with ground truth values

## Setup
Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

pip install -r Req.txt

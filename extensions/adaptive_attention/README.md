# Chronos-2 Adaptive Attention Research

This project implements an **Adaptive Relevance Bias** mechanism for the **Chronos-2** forecasting model. It enables the model to dynamically learn relationships between different variables in a multivariate time series.

## Project Structure
- `adaptive_attention.py`: Core implementation of the `RelevanceBias` and `AdaptiveGroupAttention` modules.
- `train_adaptive.py`: Logic for surgical fine-tuning of the adaptive parameters while keeping the backbone frozen.
- `demonstrate_full_workflow.py`: The main entry point that runs the full evaluation (Baseline -> Untrained Adaptive -> Trained Adaptive).
- `Req.txt`: Python dependencies required for the project.
- `chronos-2juptyter.ipynb`: Original exploration and research notebook.

## Methodology
1. **Architecture**: We surgically inject a relevance bias into the `GroupSelfAttention` layers of the pre-trained Chronos-2 encoder.
2. **Fine-Tuning**: We used a custom dataset of correlated signals to train the adaptive module using an `AdamW` optimizer with a learning rate of `5e-5`.
3. **Stability**: We implemented a `0.1` scaling factor for the bias and gradient clipping to ensure the model maintains its pre-trained forecasting intelligence during adaptation.

## Setup & Execution
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r Req.txt
   ```
3. Run the demonstration:
   ```bash
   python demonstrate_full_workflow.py
   ```

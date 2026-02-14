# Chronos2-NLP

Deep NLP course project built around **Chronos-2** for time-series forecasting and anomaly detection, with two extensions:

1) **Extension A — Forecast-based anomaly detection (Yahoo S5 + residual detectors + LLM explanations)**  
2) **Extension B — Adaptive Attention for Chronos-2**

This README focuses on **reproducibility**: clear entry points, clean outputs, and minimal assumptions.

---

## Repository structure (high-level)

```
Chronos2-NLP/
    src/
        chronos2_nlp/
            data/
                dataset.py
                windowing.py
    configs/
    extensions/
        adaptive_attention/
            adaptive_attention.py
            evaluate_paper_metrics.py
            load_fev_datasets.py
            train_universal.py
        anamoly_detection/
            download_data.py
            extract_yahoo_top_events.py
            run_baseline_all.py
            run_chronos2_baseline.py
            run_llm_explanations.py
            run_rolling_1step.py
            sanity_windows.py
    notebooks/
        00.baseline.ipynb
    README.md
    requirements.txt
    
```

## 1) Setup

### 1.1 Create venv (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
### 1.2 Make sure PYTHONPATH points to src

Most anomaly detection scripts assume:
```
$env:PYTHONPATH="$PWD\src"
```
---

## 2) Baseline: Chronos-2 forecasting

Run the baseline script:
```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/run_chronos2_baseline.py --device cpu --horizon 96 --results_dir results
```
Outputs (example):

* results/ folder with generated tables/figures depending on script behavior.

---

## 3) Extension A: Forecast-based anomaly detection (Detectors + Yahoo S5 + LLM)

### 3.1 Rolling 1-step probabilistic forecasts + detectors

This extension computes rolling 1-step-ahead probabilistic forecasts from Chronos-2:

* quantiles: q10, q50, q90
* prediction interval width: pi_width = q90 - q10
* residual: residual = y_true - q50
* PI-based anomaly score: score_z = |y_true - q50| / (pi_width + eps) plus coverage_out indicator

Then we add three detectors:

* PI detector (`anom_pi`): flags when coverage_out==1 and score_z exceeds a rolling quantile threshold
* Residual z-score (`anom_resid_z`): rolling z-score of residuals with rolling quantile threshold
* Residual MAD (`anom_resid_mad`): robust rolling MAD-based score on residuals with rolling quantile threshold
* Ensemble (`anom_ens`): majority vote (2/3) across [anom_pi, anom_resid_z, anom_resid_mad]

Run on internal domains:
```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/run_rolling_1step.py --device cpu --context 256 --max_steps 600 --stride 1
```
Outputs:

* `results/tables/rolling_1step_<domain>.csv`
* `results/figures/rolling_1step_<domain>.png`

---

### 3.2 Synthetic anomaly injection (sanity check)

We optionally inject synthetic anomalies inside the evaluated window to verify that detectors can recover known anomalies.

Run:
```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/run_rolling_1step.py --device cpu --context 256 --max_steps 600 --stride 1 --inject --n_anoms 10 --seed 0 --debug_inject
```
Outputs:

* `results/tables/rolling_1step_<domain>_INJECT.csv`
* `results/tables/inject_eval_metrics.csv`
* optional injection debug prints in terminal

---

### 3.3 Yahoo S5 dataset (A1Benchmark): download + run

We evaluate on Yahoo S5 A1Benchmark (subset of real_*.csv files).
Each file has ground-truth labels in column is_anomaly (0/1):
timestamp,value,is_anomaly

#### Expected local path

Place files under:

data/raw/yahoo_s5/A1Benchmark/
  real_1.csv
  real_2.csv
  ...

#### Run Yahoo evaluation
```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/run_rolling_1step.py --device cpu --context 256 --max_steps 600 --stride 1 --yahoo --yahoo_dir data/raw/yahoo_s5/A1Benchmark --yahoo_n 20
```
Outputs:

* per-series rolling tables:

  * `results/tables/rolling_1step_yahoo_real_*.csv`
* micro-average metrics:

  * `results/tables/yahoo_a1_micro_metrics.csv`

The rolling table schema includes
(example):
timestamp, t_idx, y_true, q10, q50, q90, pi_width, abs_err, coverage_out, score_z, residual, anom_pi, anom_resid_z, anom_resid_mad, anom_ens, label, series_id

---

### 3.4 Top event extraction (for report + LLM)

To make LLM usage cheap and report-friendly, we rank and select only the strongest anomaly events.

Extract top-K events:

```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/extract_yahoo_top_events.py --detector_col anom_resid_mad --top_k 80
``` 

Typical output:

* `results/tables/yahoo_top_events_anom_resid_mad.csv`
* optionally a shortlist file created by you (e.g., `results/tables/yahoo_llm_shortlist_top20.csv`)

---

### 3.5 LLM explanations for selected events (Groq)

We generate short, structured explanations for selected anomaly events.

#### Set API key (PowerShell)

```
$env:GROQ_API_KEY="YOUR_KEY"
``` 

#### Run LLM explanations on a curated shortlist 
```
$env:PYTHONPATH="$PWD\src"
python extensions/anomaly_detection/run_llm_explanations.py --events_csv results/tables/yahoo_llm_shortlist_top20.csv --detector_col anom_resid_mad --max_events 20 --groq_model llama-3.1-8b-instant
```
Outputs:

* `results/tables/yahoo_llm_explanations.csv`
* optional summary tables (if you generated them): e.g., `results/tables/yahoo_llm_summary.csv`

---

## 4) Extension B: Adaptive Attention 

This extension is located under:

extensions/adaptive_attention/

Goal (high-level):

* surgically inject a relevance bias / adaptive attention module into Chronos-2 attention
* optionally fine-tune only the adaptive parameters while keeping the backbone frozen

Run (example):

cd extensions/adaptive_attention
python demonstrate_full_workflow.py
cd ../..



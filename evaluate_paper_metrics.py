"""
Paper-Aligned Evaluation: Skill Scores (SQL & MASE)

Metrics from Chronos-2 paper:
  - SQL (Scaled Quantile Loss) → Primary metric (Section 5.1, Figure 5a)
  - MASE (Mean Absolute Scaled Error) → Secondary metric (Table 7, Figure 11a)
  - Skill Score = (1 - metric/naive_metric) * 100

For energy domain: seasonal_period=24 (hourly data, daily seasonality)
"""

import torch
import numpy as np
from scipy import stats
import json
import argparse
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from load_fev_datasets import load_fev_dataset, prepare_test_samples


# ============================================================
# Paper-Aligned Metrics
# ============================================================

def quantile_loss(y_true, y_pred, quantile):
    """Quantile loss (pinball loss) for a single quantile level."""
    errors = y_true - y_pred
    return np.where(errors >= 0, quantile * errors, (quantile - 1) * errors).mean()


def scaled_quantile_loss(y_true, quantile_preds, quantiles, y_train, seasonal_period=24):
    """
    Scaled Quantile Loss (SQL) - Paper's PRIMARY metric.
    SQL = QL_model / QL_seasonal_naive
    
    Args:
        y_true: (n_vars, pred_len) actual values
        quantile_preds: (n_vars, n_quantiles, pred_len) quantile forecasts
        quantiles: list of quantile levels (e.g., [0.1, 0.2, ..., 0.9])
        y_train: (n_vars, context_len) historical data for naive baseline
        seasonal_period: seasonality period (24 for hourly)
    """
    n_vars, pred_len = y_true.shape
    
    # Model quantile loss (average across all quantiles and variables)
    model_ql = 0.0
    for q_idx, q in enumerate(quantiles):
        model_ql += quantile_loss(y_true, quantile_preds[:, q_idx, :], q)
    model_ql /= len(quantiles)
    
    # Seasonal naive quantile loss
    # Naive forecast: repeat last seasonal_period from history
    if y_train.shape[1] >= seasonal_period:
        naive_forecast = y_train[:, -seasonal_period:]
        # Tile to cover prediction length
        repeats = (pred_len + seasonal_period - 1) // seasonal_period
        naive_tiled = np.tile(naive_forecast, (1, repeats))[:, :pred_len]
    else:
        naive_tiled = np.tile(y_train.mean(axis=1, keepdims=True), (1, pred_len))
    
    naive_ql = 0.0
    for q in quantiles:
        naive_ql += quantile_loss(y_true, naive_tiled, q)
    naive_ql /= len(quantiles)
    
    # SQL = model_ql / naive_ql
    sql = model_ql / (naive_ql + 1e-8)
    
    return sql, model_ql, naive_ql


def mase(y_true, y_pred, y_train, seasonal_period=24):
    """
    Mean Absolute Scaled Error (MASE) - Paper's SECONDARY metric.
    MASE = MAE_model / MAE_seasonal_naive
    
    Args:
        y_true: (n_vars, pred_len) actual values
        y_pred: (n_vars, pred_len) point forecast (median)
        y_train: (n_vars, context_len) historical data
        seasonal_period: seasonality period (24 for hourly)
    """
    # Model MAE
    mae_model = np.abs(y_true - y_pred).mean()
    
    # Seasonal naive MAE (from training data)
    if y_train.shape[1] > seasonal_period:
        naive_errors = np.abs(y_train[:, seasonal_period:] - y_train[:, :-seasonal_period])
        mae_naive = naive_errors.mean()
    else:
        mae_naive = np.abs(y_train - y_train.mean(axis=1, keepdims=True)).mean()
    
    mase_val = mae_model / (mae_naive + 1e-8)
    
    return mase_val, mae_model, mae_naive


def skill_score(metric_value):
    """Skill Score = (1 - metric) * 100%"""
    return (1 - metric_value) * 100


# ============================================================
# Evaluation with Full Quantile Outputs
# ============================================================

def evaluate_with_quantiles(pipeline, test_samples_with_history, device='mps'):
    """
    Evaluate model and return both point and quantile predictions.
    
    Args:
        test_samples_with_history: list of (context, target, history_for_naive)
    
    Returns:
        dict with MAE, MASE, SQL per sample
    """
    # Chronos-2 quantile levels (9 quantiles)
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = {
        'mae_list': [],
        'mase_list': [],
        'sql_list': [],
        'skill_mase_list': [],
        'skill_sql_list': [],
    }
    
    for i, (context, target, history) in enumerate(test_samples_with_history):
        try:
            # Get quantile forecasts
            forecasts = pipeline.predict(
                [context],
                prediction_length=target.shape[1]
            )
            
            fc_tensor = forecasts[0]  # (n_vars, n_quantiles, pred_len)
            
            # Point forecast (median = quantile index 4)
            median_forecast = fc_tensor[:, 4, :].numpy()
            
            # All quantile forecasts
            quantile_forecasts = fc_tensor.numpy()  # (n_vars, 9, pred_len)
            
            target_np = target.numpy()
            history_np = history.numpy()
            
            # Raw MAE
            mae_val = np.abs(median_forecast - target_np).mean()
            results['mae_list'].append(mae_val)
            
            # MASE
            mase_val, _, _ = mase(target_np, median_forecast, history_np, seasonal_period=24)
            results['mase_list'].append(mase_val)
            results['skill_mase_list'].append(skill_score(mase_val))
            
            # SQL
            sql_val, _, _ = scaled_quantile_loss(
                target_np, quantile_forecasts, quantiles, history_np, seasonal_period=24
            )
            results['sql_list'].append(sql_val)
            results['skill_sql_list'].append(skill_score(sql_val))
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_samples_with_history)} done")
                
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue
    
    # Convert to numpy arrays
    for k in results:
        results[k] = np.array(results[k])
    
    return results


# ============================================================
# Prepare Samples with History for Naive Baseline
# ============================================================

def prepare_samples_with_history(test_data, train_data, context_length=512, prediction_length=96, 
                                  num_samples=50, seasonal_period=24):
    """
    Create test samples that include history for seasonal naive baseline.
    
    Returns:
        List of (context, target, history_for_naive) tuples
    """
    n_vars, total_len = test_data.shape
    window_len = context_length + prediction_length
    
    samples = []
    max_start = total_len - window_len
    
    if max_start <= 0:
        print(f"Warning: Test data too short ({total_len}) for window ({window_len})")
        return []
    
    step = max(1, max_start // num_samples)
    
    for i in range(0, max_start, step):
        if len(samples) >= num_samples:
            break
        
        window = test_data[:, i:i+window_len]
        context = window[:, :context_length]
        target = window[:, context_length:context_length+prediction_length]
        
        # History for naive baseline = context data (used for MASE/SQL scaling)
        history = context.copy()
        
        samples.append((
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(history, dtype=torch.float32)
        ))
    
    return samples


# ============================================================
# Main Evaluation
# ============================================================

def run_evaluation(dataset_name, device='mps'):
    print("=" * 70)
    print(f"PAPER-ALIGNED EVALUATION: {dataset_name.upper()}")
    print("Metrics: Skill Score (SQL) + Skill Score (MASE) + Raw MAE")
    print("=" * 70)
    
    # Determine seasonal period
    seasonal_periods = {
        'ETTm1': 96,       # 15-min data, daily = 96 steps
        'ETTh1': 24,       # hourly data, daily = 24 steps
        'electricity': 24, # hourly data, daily = 24 steps
    }
    seasonal_period = seasonal_periods.get(dataset_name, 24)
    
    # Load data
    print(f"\n[1/4] Loading {dataset_name} dataset...")
    data = load_fev_dataset(dataset_name)
    
    # Determine window lengths
    if dataset_name == 'hospital':
        context_length = 256
        prediction_length = 24
    elif dataset_name == 'rossmann':
        context_length = 128
        prediction_length = 28
    else:
        context_length = 512
        prediction_length = 96

    test_samples = prepare_samples_with_history(
        data['test_data'],
        data['train_data'],
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=50,
        seasonal_period=seasonal_period
    )
    print(f"  {len(test_samples)} test samples")
    print(f"  Context length: {context_length}, Prediction length: {prediction_length}")
    print(f"  Seasonal period: {seasonal_period}")
    
    # Baseline Chronos-2
    print(f"\n[2/4] Evaluating Baseline Chronos-2...")
    baseline_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    baseline_results = evaluate_with_quantiles(baseline_pipeline, test_samples, device)
    
    # Adaptive Attention
    print(f"\n[3/4] Evaluating Adaptive Attention...")
    adaptive_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    adaptive_pipeline = replace_chronos_attention(adaptive_pipeline)
    
    checkpoint_path = f'./checkpoints/{dataset_name}_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    adaptive_pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    
    adaptive_results = evaluate_with_quantiles(adaptive_pipeline, test_samples, device)
    
    # Statistical tests
    print(f"\n[4/4] Computing statistics...")
    
    # Paired t-tests
    t_mae, p_mae = stats.ttest_rel(adaptive_results['mae_list'], baseline_results['mae_list'], alternative='less')
    t_mase, p_mase = stats.ttest_rel(adaptive_results['mase_list'], baseline_results['mase_list'], alternative='less')
    t_sql, p_sql = stats.ttest_rel(adaptive_results['sql_list'], baseline_results['sql_list'], alternative='less')
    
    # Effect sizes
    diff_mae = adaptive_results['mae_list'] - baseline_results['mae_list']
    d_mae = np.mean(diff_mae) / (np.std(diff_mae) + 1e-8)
    
    diff_mase = adaptive_results['mase_list'] - baseline_results['mase_list']
    d_mase = np.mean(diff_mase) / (np.std(diff_mase) + 1e-8)
    
    diff_sql = adaptive_results['sql_list'] - baseline_results['sql_list']
    d_sql = np.mean(diff_sql) / (np.std(diff_sql) + 1e-8)
    
    # Compile results
    final_results = {
        'dataset': dataset_name,
        'seasonal_period': seasonal_period,
        'num_samples': len(test_samples),
        'baseline': {
            'mae': {'mean': float(baseline_results['mae_list'].mean()), 'std': float(baseline_results['mae_list'].std())},
            'mase': {'mean': float(baseline_results['mase_list'].mean()), 'std': float(baseline_results['mase_list'].std())},
            'sql': {'mean': float(baseline_results['sql_list'].mean()), 'std': float(baseline_results['sql_list'].std())},
            'skill_mase': {'mean': float(baseline_results['skill_mase_list'].mean()), 'std': float(baseline_results['skill_mase_list'].std())},
            'skill_sql': {'mean': float(baseline_results['skill_sql_list'].mean()), 'std': float(baseline_results['skill_sql_list'].std())},
        },
        'adaptive': {
            'mae': {'mean': float(adaptive_results['mae_list'].mean()), 'std': float(adaptive_results['mae_list'].std())},
            'mase': {'mean': float(adaptive_results['mase_list'].mean()), 'std': float(adaptive_results['mase_list'].std())},
            'sql': {'mean': float(adaptive_results['sql_list'].mean()), 'std': float(adaptive_results['sql_list'].std())},
            'skill_mase': {'mean': float(adaptive_results['skill_mase_list'].mean()), 'std': float(adaptive_results['skill_mase_list'].std())},
            'skill_sql': {'mean': float(adaptive_results['skill_sql_list'].mean()), 'std': float(adaptive_results['skill_sql_list'].std())},
        },
        'statistical_tests': {
            'mae':  {'t_stat': float(t_mae), 'p_value': float(p_mae), 'cohens_d': float(d_mae)},
            'mase': {'t_stat': float(t_mase), 'p_value': float(p_mase), 'cohens_d': float(d_mase)},
            'sql':  {'t_stat': float(t_sql), 'p_value': float(p_sql), 'cohens_d': float(d_sql)},
        },
        'improvements': {
            'mae_pct': float((baseline_results['mae_list'].mean() - adaptive_results['mae_list'].mean()) / baseline_results['mae_list'].mean() * 100),
            'skill_mase_diff_pp': float(adaptive_results['skill_mase_list'].mean() - baseline_results['skill_mase_list'].mean()),
            'skill_sql_diff_pp': float(adaptive_results['skill_sql_list'].mean() - baseline_results['skill_sql_list'].mean()),
        }
    }
    
    # Save
    save_path = f'./results/{dataset_name}_paper_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"RESULTS: {dataset_name.upper()} (Paper-Aligned Metrics)")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Baseline':<20} {'Adaptive':<20} {'Improvement':<15} {'p-value':<10}")
    print("-" * 90)
    
    # Raw MAE
    b_mae = final_results['baseline']['mae']['mean']
    a_mae = final_results['adaptive']['mae']['mean']
    imp_mae = final_results['improvements']['mae_pct']
    print(f"{'MAE (raw)':<25} {b_mae:<20.4f} {a_mae:<20.4f} {imp_mae:>+.2f}%{'':>7} {p_mae:<10.4f}")
    
    # MASE
    b_mase = final_results['baseline']['mase']['mean']
    a_mase = final_results['adaptive']['mase']['mean']
    print(f"{'MASE':<25} {b_mase:<20.4f} {a_mase:<20.4f} {'':>15} {p_mase:<10.4f}")
    
    # Skill Score (MASE)
    b_ss_mase = final_results['baseline']['skill_mase']['mean']
    a_ss_mase = final_results['adaptive']['skill_mase']['mean']
    diff_ss_mase = final_results['improvements']['skill_mase_diff_pp']
    print(f"{'Skill Score (MASE) %':<25} {b_ss_mase:<20.2f} {a_ss_mase:<20.2f} {diff_ss_mase:>+.2f} pp{'':>4} {p_mase:<10.4f}")
    
    # SQL
    b_sql = final_results['baseline']['sql']['mean']
    a_sql = final_results['adaptive']['sql']['mean']
    print(f"{'SQL':<25} {b_sql:<20.4f} {a_sql:<20.4f} {'':>15} {p_sql:<10.4f}")
    
    # Skill Score (SQL) - PRIMARY
    b_ss_sql = final_results['baseline']['skill_sql']['mean']
    a_ss_sql = final_results['adaptive']['skill_sql']['mean']
    diff_ss_sql = final_results['improvements']['skill_sql_diff_pp']
    print(f"{'⭐ Skill Score (SQL) %':<25} {b_ss_sql:<20.2f} {a_ss_sql:<20.2f} {diff_ss_sql:>+.2f} pp{'':>4} {p_sql:<10.4f}")
    
    print("-" * 90)
    
    # Significance summary
    print(f"\nPaper comparison (energy domain, Figure 5a):")
    print(f"  Paper's Chronos-2 ICL improvement: +6.5 pp (SQL skill score)")
    print(f"  Our Adaptive Attention improvement: {diff_ss_sql:+.2f} pp (SQL skill score)")
    
    for metric_name, p_val in [("MAE", p_mae), ("MASE", p_mase), ("SQL", p_sql)]:
        if p_val < 0.01:
            print(f"  {metric_name}: ✅ HIGHLY SIGNIFICANT (p={p_val:.4f})")
        elif p_val < 0.05:
            print(f"  {metric_name}: ✅ SIGNIFICANT (p={p_val:.4f})")
        else:
            print(f"  {metric_name}: ⚠️  Not significant (p={p_val:.4f})")
    
    print(f"\n✓ Results saved to: {save_path}")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-aligned evaluation with Skill Scores")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ETTm1', 'ETTh1', 'electricity', 'hospital', 'epf_de', 'rossmann'],
                        help='Dataset to evaluate')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device (mps/cpu/cuda)')
    
    args = parser.parse_args()
    run_evaluation(args.dataset, args.device)

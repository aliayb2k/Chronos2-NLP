"""
Quick ETTh1 evaluation using the working API from evaluate_statistical.py
"""

import torch
import numpy as np
from scipy import stats
import json
import os
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from load_fev_datasets import load_fev_dataset, prepare_test_samples


def evaluate_model(pipeline, test_samples, device='mps'):
    """Evaluate model using pipeline.predict() - the working API"""
    print(f"Evaluating on {len(test_samples)} samples...")
    
    mae_list = []
    
    for i, (context, target) in enumerate(test_samples):
        try:
            context_list = [context]
            
            # Use pipeline.predict() - this works!
            forecasts = pipeline.predict(
                context_list,
                prediction_length=target.shape[1]
            )
            
            fc_tensor = forecasts[0]  # (n_vars, quantiles, pred_len)
            median_idx = 4  # Median quantile
            forecast_median = fc_tensor[:, median_idx, :]  # (n_vars, pred_len)
            
            # Compute MAE
            mae = torch.abs(forecast_median - target).mean().item()
            mae_list.append(mae)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_samples)} done")
                
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue
    
    return np.array(mae_list)


# Load data
print("="*70)
print("ETTH1 EVALUATION")
print("="*70)

print("\n[1/3] Loading ETTh1 test set...")
data = load_fev_dataset('ETTh1')
test_samples = prepare_test_samples(
    data['test_data'],
    context_length=512,
    prediction_length=96,
    num_samples=50
)
print(f"  {len(test_samples)} test samples")

# Baseline
print("\n[2/3] Evaluating Baseline Chronos-2...")
baseline_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map='mps',
    torch_dtype=torch.float32
)
baseline_maes = evaluate_model(baseline_pipeline, test_samples)

# Adaptive
print("\n[3/3] Evaluating Adaptive Attention...")
adaptive_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map='mps',
    torch_dtype=torch.float32
)
adaptive_pipeline = replace_chronos_attention(adaptive_pipeline)

# Load trained checkpoint
checkpoint = torch.load('./checkpoints/ETTh1_best.pt', map_location='mps')
adaptive_pipeline.model.load_state_dict(checkpoint['model_state_dict'])

adaptive_maes = evaluate_model(adaptive_pipeline, test_samples)

# Statistics
print("\n[4/4] Computing statistics...")
t_stat, p_val = stats.ttest_rel(adaptive_maes, baseline_maes, alternative='less')
diff = adaptive_maes - baseline_maes
cohens_d = np.mean(diff) / np.std(diff)
improvement = (baseline_maes.mean() - adaptive_maes.mean()) / baseline_maes.mean() * 100

results = {
    'baseline': {
        'mean': float(baseline_maes.mean()),
        'std': float(baseline_maes.std())
    },
    'adaptive': {
        'mean': float(adaptive_maes.mean()),
        'std': float(adaptive_maes.std())
    },
    'comparison': {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'cohens_d': float(cohens_d),
        'improvement_pct': float(improvement)
    }
}

# Save
with open('./results/ETTh1_evaluation.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nBaseline MAE: {results['baseline']['mean']:.4f} ± {results['baseline']['std']:.4f}")
print(f"Adaptive MAE: {results['adaptive']['mean']:.4f} ± {results['adaptive']['std']:.4f}")
print(f"\nImprovement: {results['comparison']['improvement_pct']:.2f}%")
print(f"p-value: {results['comparison']['p_value']:.4f}")
print(f"Cohen's d: {results['comparison']['cohens_d']:.3f}")

if p_val < 0.01:
    print("\n✓ HIGHLY SIGNIFICANT (p < 0.01)")
elif p_val < 0.05:
    print("\n✓ SIGNIFICANT (p < 0.05)")
else:
    print("\n⚠ Not significant (p >= 0.05)")

print(f"\n✓ Results saved to: ./results/ETTh1_evaluation.json")

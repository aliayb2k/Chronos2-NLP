"""
Electricity evaluation WITH SCALING (standardization)
Normalizes data to mean=0, std=1 before evaluation
"""

import torch
import numpy as np
from scipy import stats
import json
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from load_fev_datasets import load_fev_dataset


def scale_data(data):
    """Standardize data to mean=0, std=1 per variable"""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    scaled = (data - mean) / (std + 1e-8)
    return scaled, mean, std


def inverse_scale(scaled_data, mean, std):
    """Convert scaled data back to original scale"""
    return scaled_data * std + mean


def prepare_scaled_samples(test_data, context_length=512, prediction_length=96, num_samples=50):
    """Create test samples with scaling applied"""
    n_vars, total_len = test_data.shape
    window_len = context_length + prediction_length
    
    samples = []
    max_start = total_len - window_len
    
    if max_start <= 0:
        return []
    
    step = max(1, max_start // num_samples)
    
    for i in range(0, max_start, step):
        if len(samples) >= num_samples:
            break
        
        window = test_data[:, i:i+window_len]
        
        # Scale the window
        scaled_window, mean, std = scale_data(window)
        
        context = scaled_window[:, :context_length]
        target_scaled = scaled_window[:, context_length:context_length+prediction_length]
        target_raw = window[:, context_length:context_length+prediction_length]
        
        samples.append((
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target_scaled, dtype=torch.float32),
            torch.tensor(target_raw, dtype=torch.float32),
            mean, std
        ))
    
    return samples


def evaluate_model_scaled(pipeline, test_samples, device='mps'):
    """Evaluate model with scaling - predictions and errors in ORIGINAL scale"""
    print(f"Evaluating on {len(test_samples)} samples...")
    
    mae_list = []
    
    for i, (context, target_scaled, target_raw, mean, std) in enumerate(test_samples):
        try:
            context_list = [context]
            
            # Predict on SCALED data
            forecasts = pipeline.predict(
                context_list,
                prediction_length=target_scaled.shape[1]
            )
            
            fc_tensor = forecasts[0]
            median_idx = 4
            forecast_scaled = fc_tensor[:, median_idx, :]
            
            # Convert prediction back to ORIGINAL scale
            forecast_raw = inverse_scale(
                forecast_scaled.numpy(), 
                mean, 
                std
            )
            
            # Compute MAE in ORIGINAL scale
            mae = np.abs(forecast_raw - target_raw.numpy()).mean()
            mae_list.append(mae)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_samples)} done")
                
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue
    
    return np.array(mae_list)


# Load data
print("="*70)
print("ELECTRICITY (ENTSO-E) EVALUATION - WITH SCALING")
print("="*70)

print("\n[1/3] Loading Electricity test set...")
data = load_fev_dataset('electricity')
test_samples = prepare_scaled_samples(
    data['test_data'],
    context_length=512,
    prediction_length=96,
    num_samples=50
)
print(f"  {len(test_samples)} test samples (with scaling applied)")

# Baseline
print("\n[2/3] Evaluating Baseline Chronos-2...")
baseline_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map='mps',
    torch_dtype=torch.float32
)
baseline_maes = evaluate_model_scaled(baseline_pipeline, test_samples)

# Adaptive
print("\n[3/3] Evaluating Adaptive Attention...")
adaptive_pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map='mps',
    torch_dtype=torch.float32
)
adaptive_pipeline = replace_chronos_attention(adaptive_pipeline)

checkpoint = torch.load('./checkpoints/electricity_best.pt', map_location='mps')
adaptive_pipeline.model.load_state_dict(checkpoint['model_state_dict'])

adaptive_maes = evaluate_model_scaled(adaptive_pipeline, test_samples)

# Statistics
print("\n[4/4] Computing statistics...")
t_stat, p_val = stats.ttest_rel(adaptive_maes, baseline_maes, alternative='less')
diff = adaptive_maes - baseline_maes
cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
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
    },
    'note': 'Evaluation with scaling: data normalized to mean=0, std=1 before prediction, MAE computed in original scale'
}

# Save
with open('./results/electricity_evaluation_scaled.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print
print("\n" + "="*70)
print("RESULTS (WITH SCALING)")
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

print(f"\n✓ Results saved to: ./results/electricity_evaluation_scaled.json")

# Compare with unscaled
print("\n" + "="*70)
print("COMPARISON: SCALED vs UNSCALED")
print("="*70)
try:
    with open('./results/electricity_evaluation.json', 'r') as f:
        unscaled = json.load(f)
    
    print("\nUNSCALED (original):")
    print(f"  Baseline: {unscaled['baseline']['mean']:.4f}")
    print(f"  Adaptive: {unscaled['adaptive']['mean']:.4f}")
    print(f"  Improvement: {unscaled['comparison']['improvement_pct']:.2f}%")
    
    print("\nSCALED (this run):")
    print(f"  Baseline: {results['baseline']['mean']:.4f}")
    print(f"  Adaptive: {results['adaptive']['mean']:.4f}")
    print(f"  Improvement: {results['comparison']['improvement_pct']:.2f}%")
    
except:
    print("(Could not load unscaled results for comparison)")

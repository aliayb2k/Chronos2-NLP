"""
Statistical evaluation script for comparing Baseline vs Adaptive Attention.

This script evaluates both models on the test set and performs statistical significance testing.
"""

import torch
import numpy as np
from scipy import stats
import json
import os
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from prepare_ett_data import load_ett_multivariate, create_windowed_samples


def evaluate_model_on_samples(pipeline, test_samples, device='cpu', model_name="Model"):
    """
    Evaluate model on all test samples and collect MAE scores.
    
    Args:
        pipeline: Chronos2Pipeline
        test_samples: List of (context, target) tuples
        device: Device to run on
        model_name: Name for logging
    
    Returns:
        List of MAE scores (one per sample)
    """
    print(f"\nEvaluating {model_name} on {len(test_samples)} test samples...")
    
    mae_list = []
    model = pipeline.model
    model.eval()
    
    with torch.no_grad():
        for i, (context, target) in enumerate(test_samples):
            try:
                context_list = [context]
                
                # Predict
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
                    print(f"  Processed {i+1}/{len(test_samples)} samples")
                    
            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue
    
    model.train()
    print(f"✓ {model_name} evaluation complete")
    return mae_list


def perform_statistical_test(baseline_maes, adaptive_maes):
    """
    Perform paired t-test and compute statistical metrics.
    
    Returns:
        Dictionary with test results
    """
    baseline_arr = np.array(baseline_maes)
    adaptive_arr = np.array(adaptive_maes)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(adaptive_arr, baseline_arr, alternative='less')
    
    # Effect size (Cohen's d for paired samples)
    diff = adaptive_arr - baseline_arr
    d = np.mean(diff) / np.std(diff)
    
    # Confidence interval for mean difference
    ci = stats.t.interval(
        0.95,
        len(diff) - 1,
        loc=np.mean(diff),
        scale=stats.sem(diff)
    )
    
    # Summary statistics
    results = {
        'baseline': {
            'mean': float(np.mean(baseline_arr)),
            'std': float(np.std(baseline_arr)),
            'median': float(np.median(baseline_arr)),
            'min': float(np.min(baseline_arr)),
            'max': float(np.max(baseline_arr))
        },
        'adaptive': {
            'mean': float(np.mean(adaptive_arr)),
            'std': float(np.std(adaptive_arr)),
            'median': float(np.median(adaptive_arr)),
            'min': float(np.min(adaptive_arr)),
            'max': float(np.max(adaptive_arr))
        },
        'statistical_test': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohen_d': float(d),
            'ci_95_lower': float(ci[0]),
            'ci_95_upper': float(ci[1]),
            'mean_difference': float(np.mean(diff)),
            'improvement_pct': float((np.mean(diff) / np.mean(baseline_arr)) * 100)
        },
        'interpretation': interpret_results(p_value, np.mean(diff))
    }
    
    return results


def interpret_results(p_value, mean_diff):
    """Interpret statistical test results."""
    if p_value < 0.01:
        significance = "highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "significant (p < 0.05)"
    else:
        significance = "not significant (p >= 0.05)"
    
    if mean_diff < 0:
        direction = "Adaptive performs BETTER than baseline"
    elif mean_diff > 0:
        direction = "Adaptive performs WORSE than baseline"
    else:
        direction = "No difference"
    
    return f"{direction} - {significance}"


def run_evaluation(
    checkpoint_path=None,
    output_dir='./results',
    n_test_samples=50,
    device=None
):
    """
    Run complete evaluation: baseline vs adaptive.
    
    Args:
        checkpoint_path: Path to trained adaptive model checkpoint (or None for untrained)
        output_dir: Directory to save results
        n_test_samples: Number of test samples to use
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("STATISTICAL EVALUATION: BASELINE VS ADAPTIVE ATTENTION")
    print("=" * 70)
    
    # Load test data
    print("\n[1/4] Loading ETTm1 test set...")
    _, _, test_data = load_ett_multivariate()
    test_samples = create_windowed_samples(test_data, context_length=96, prediction_length=96)
    test_samples = test_samples[:n_test_samples]
    print(f"  Using {len(test_samples)} test samples")
    
    # Evaluate baseline
    print("\n[2/4] Evaluating baseline Chronos-2...")
    baseline_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    baseline_maes = evaluate_model_on_samples(baseline_pipeline, test_samples, device, "Baseline")
    
    # Evaluate adaptive
    print("\n[3/4] Evaluating adaptive attention...")
    adaptive_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    adaptive_pipeline = replace_chronos_attention(adaptive_pipeline)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        adaptive_pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  Using untrained adaptive attention (zero-initialized)")
    
    adaptive_maes = evaluate_model_on_samples(adaptive_pipeline, test_samples, device, "Adaptive")
    
    # Statistical analysis
    print("\n[4/4] Performing statistical analysis...")
    results = perform_statistical_test(baseline_maes, adaptive_maes)
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nBaseline Chronos-2:")
    print(f"  MAE: {results['baseline']['mean']:.4f} ± {results['baseline']['std']:.4f}")
    
    print("\nAdaptive Attention:")
    print(f"  MAE: {results['adaptive']['mean']:.4f} ± {results['adaptive']['std']:.4f}")
    
    print("\nStatistical Test:")
    print(f"  Mean Difference: {results['statistical_test']['mean_difference']:.4f}")
    print(f"  95% CI: [{results['statistical_test']['ci_95_lower']:.4f}, {results['statistical_test']['ci_95_upper']:.4f}]")
    print(f"  t-statistic: {results['statistical_test']['t_statistic']:.3f}")
    print(f"  p-value: {results['statistical_test']['p_value']:.4f}")
    print(f"  Cohen's d: {results['statistical_test']['cohen_d']:.3f}")
    
    print(f"\n{results['interpretation']}")
    
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    # Run evaluation with best trained model
    results = run_evaluation(
        checkpoint_path='./checkpoints/best_model.pt',
        n_test_samples=50
    )

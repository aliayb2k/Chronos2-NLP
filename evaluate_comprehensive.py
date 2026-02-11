"""
Comprehensive 3-way evaluation script for all datasets.

Compares three configurations:
1. Univariate baseline (each variable independently)
2. Chronos-2 group attention baseline  
3. Adaptive attention (your contribution)

Usage:
    python evaluate_comprehensive.py --dataset ETTm1
    python evaluate_comprehensive.py --dataset all  # Run on all 5 datasets
"""

import argparse
import os
import json
import numpy as np
import torch
from scipy import stats
from chronos import ChronosPipeline
from adaptive_attention import replace_chronos_attention
from load_fev_datasets import load_fev_dataset, prepare_test_samples


def evaluate_univariate(pipeline, test_samples, prediction_length, device='cpu'):
    """
    Mode 1: Univariate baseline.
    Predict each variable independently (no group attention).
    """
    print("\nEvaluating Univariate Baseline...")
    model = pipeline.model
    model.eval()
    
    all_maes = []
    
    with torch.no_grad():
        for i, (context, target) in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_samples)} samples")
            
            n_vars = context.shape[0]
            context = context.to(device)
            target = target.to(device)
            
            # Predict each variable separately
            var_maes = []
            for v in range(n_vars):
                # Single variable context
                ctx_single = context[v:v+1, :]  # (1, seq_len)
                tgt_single = target[v:v+1, :]    # (1, pred_len)
                
                # Group IDs: single series (no grouping)
                group_ids = torch.zeros(1, dtype=torch.long).to(device)
                
                num_patches = (prediction_length + 15) // 16
                
                try:
                    output = model(
                        context=ctx_single,
                        future_target=tgt_single,
                        group_ids=group_ids,
                        num_output_patches=num_patches
                    )
                    
                    # Extract median forecast
                    forecast = output.prediction_outputs[:, :, 4]  # Median (0.5 quantile)
                    mae = torch.abs(forecast - tgt_single).mean().item()
                    var_maes.append(mae)
                    
                except Exception as e:
                    print(f"    Error on variable {v}: {e}")
                    continue
            
            # Average across variables for this sample
            if len(var_maes) > 0:
                all_maes.append(np.mean(var_maes))
    
    print(f"✓ Univariate evaluation complete")
    return np.array(all_maes)


def evaluate_baseline(pipeline, test_samples, prediction_length, device='cpu'):
    """
    Mode 2: Chronos-2 group attention baseline.
    Standard multivariate mode (no adaptive attention).
    """
    print("\nEvaluating Baseline Chronos-2 Group Attention...")
    model = pipeline.model
    model.eval()
    
    all_maes = []
    
    with torch.no_grad():
        for i, (context, target) in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_samples)} samples")
            
            n_vars = context.shape[0]
            context = context.to(device)
            target = target.to(device)
            
            # All variables in same group
            group_ids = torch.zeros(n_vars, dtype=torch.long).to(device)
            
            num_patches = (prediction_length + 15) // 16
            
            try:
                output = model(
                    context=context,
                    future_target=target,
                    group_ids=group_ids,
                    num_output_patches=num_patches
                )
                
                # Extract median forecast
                forecast = output.prediction_outputs[:, :, 4]  # Median
                mae = torch.abs(forecast - target).mean().item()
                all_maes.append(mae)
                
            except Exception as e:
                print(f"    Error on sample {i}: {e}")
                continue
    
    print(f"✓ Baseline evaluation complete")
    return np.array(all_maes)


def evaluate_adaptive(pipeline, checkpoint_path, test_samples, prediction_length, device='cpu'):
    """
    Mode 3: Adaptive attention (your contribution).
    Group attention + RelevanceBias.
    """
    print("\nEvaluating Adaptive Attention...")
    
    # Replace with adaptive attention
    pipeline = replace_chronos_attention(pipeline)
    
    # Load trained checkpoint
    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    
    model = pipeline.model
    model.eval()
    
    all_maes = []
    
    with torch.no_grad():
        for i, (context, target) in enumerate(test_samples):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_samples)} samples")
            
            n_vars = context.shape[0]
            context = context.to(device)
            target = target.to(device)
            
            group_ids = torch.zeros(n_vars, dtype=torch.long).to(device)
            num_patches = (prediction_length + 15) // 16
            
            try:
                output = model(
                    context=context,
                    future_target=target,
                    group_ids=group_ids,
                    num_output_patches=num_patches
                )
                
                forecast = output.prediction_outputs[:, :, 4]
                mae = torch.abs(forecast - target).mean().item()
                all_maes.append(mae)
                
            except Exception as e:
                print(f"    Error on sample {i}: {e}")
                continue
    
    print(f"✓ Adaptive evaluation complete")
    return np.array(all_maes)


def compute_statistics(univariate_maes, baseline_maes, adaptive_maes):
    """Compute statistical comparisons between all three modes."""
    
    results = {}
    
    # Summary statistics
    results['univariate'] = {
        'mean': float(np.mean(univariate_maes)),
        'std': float(np.std(univariate_maes)),
        'median': float(np.median(univariate_maes))
    }
    
    results['baseline'] = {
        'mean': float(np.mean(baseline_maes)),
        'std': float(np.std(baseline_maes)),
        'median': float(np.median(baseline_maes))
    }
    
    results['adaptive'] = {
        'mean': float(np.mean(adaptive_maes)),
        'std': float(np.std(adaptive_maes)),
        'median': float(np.median(adaptive_maes))
    }
    
    # Paired t-tests
    # 1. Baseline vs Univariate
    t_stat_bu, p_val_bu = stats.ttest_rel(baseline_maes, univariate_maes)
    improvement_bu = (univariate_maes.mean() - baseline_maes.mean()) / univariate_maes.mean() * 100
    
    # 2. Adaptive vs Baseline 
    t_stat_ab, p_val_ab = stats.ttest_rel(adaptive_maes, baseline_maes)
    improvement_ab = (baseline_maes.mean() - adaptive_maes.mean()) / baseline_maes.mean() * 100
    
    # 3. Adaptive vs Univariate
    t_stat_au, p_val_au = stats.ttest_rel(adaptive_maes, univariate_maes)
    improvement_au = (univariate_maes.mean() - adaptive_maes.mean()) / univariate_maes.mean() * 100
    
    # Cohen's d effect sizes
    def cohens_d(a, b):
        diff = a - b
        return np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
    
    results['comparisons'] = {
        'baseline_vs_univariate': {
            't_statistic': float(t_stat_bu),
            'p_value': float(p_val_bu),
            'improvement_pct': float(improvement_bu),
            'cohens_d': float(cohens_d(baseline_maes, univariate_maes))
        },
        'adaptive_vs_baseline': {
            't_statistic': float(t_stat_ab),
            'p_value': float(p_val_ab),
            'improvement_pct': float(improvement_ab),
            'cohens_d': float(cohens_d(adaptive_maes, baseline_maes))
        },
        'adaptive_vs_univariate': {
            't_statistic': float(t_stat_au),
            'p_value': float(p_val_au),
            'improvement_pct': float(improvement_au),
            'cohens_d': float(cohens_d(adaptive_maes, univariate_maes))
        }
    }
    
    return results


def evaluate_dataset(dataset_name, device='mps'):
    """Run comprehensive 3-way evaluation on a single dataset."""
    
    print("=" * 70)
    print(f"COMPREHENSIVE EVALUATION: {dataset_name.upper()}")
    print("=" * 70)
    print()
    
    # Load data
    print(f"[1/4] Loading {dataset_name} test set...")
    data = load_fev_dataset(dataset_name)
    test_samples = prepare_test_samples(
        data['test_data'],
        context_length=512,
        prediction_length=data['prediction_length'],
        num_samples=50
    )
    print(f"  Using {len(test_samples)} test samples")
    
    # Load pipeline
    print("\n[2/4] Loading Chronos-2 pipeline...")
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    
    # Mode 1: Univariate
    print("\n[3/4] Running evaluations...")
    univariate_maes = evaluate_univariate(pipeline, test_samples, data['prediction_length'], device)
    
    # Mode 2: Baseline
    baseline_maes = evaluate_baseline(pipeline, test_samples, data['prediction_length'], device)
    
    # Mode 3: Adaptive
    checkpoint_path = f"./checkpoints/{dataset_name}_best.pt"
    adaptive_maes = evaluate_adaptive(pipeline, checkpoint_path, test_samples, data['prediction_length'], device)
    
    # Statistics
    print("\n[4/4] Computing statistics...")
    results = compute_statistics(univariate_maes, baseline_maes, adaptive_maes)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nUnivariate Baseline:")
    print(f"  MAE: {results['univariate']['mean']:.4f} ± {results['univariate']['std']:.4f}")
    print(f"\nChronos-2 Group Attention:")
    print(f"  MAE: {results['baseline']['mean']:.4f} ± {results['baseline']['std']:.4f}")
    print(f"\nAdaptive Attention:")
    print(f"  MAE: {results['adaptive']['mean']:.4f} ± {results['adaptive']['std']:.4f}")
    
    print(f"\nComparisons:")
    comp = results['comparisons']
    print(f"\nBaseline vs Univariate:")
    print(f"  Improvement: {comp['baseline_vs_univariate']['improvement_pct']:.2f}%")
    print(f"  p-value: {comp['baseline_vs_univariate']['p_value']:.4f}")
    
    print(f"\nAdaptive vs Baseline:")
    print(f"  Improvement: {comp['adaptive_vs_baseline']['improvement_pct']:.2f}%")
    print(f"  p-value: {comp['adaptive_vs_baseline']['p_value']:.4f}")
    print(f"  Cohen's d: {comp['adaptive_vs_baseline']['cohens_d']:.3f}")
    
    # Save results
    output_path = f"./results/{dataset_name}_comprehensive.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive 3-way evaluation")
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['ETTm1', 'ETTh1', 'electricity', 'traffic', 'weather']
    else:
        datasets = [args.dataset]
    
    all_results = {}
    for ds in datasets:
        results = evaluate_dataset(ds, device=args.device)
        all_results[ds] = results
    
    # Save aggregated results
    if len(datasets) > 1:
        with open('./results/all_datasets_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\n✓ Aggregated results saved to: ./results/all_datasets_summary.json")

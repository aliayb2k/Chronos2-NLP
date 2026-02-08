
import torch
import numpy as np
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from train_adaptive import train_model

def get_test_sample(context_len=50, prediction_len=20):
    """
    Creates a single multivariate time series (2 variates) where:
    Var 0: Sine wave
    Var 1: 0.8 * Var 0 (Highly correlated)
    
    We want to see if the model can use Var 0 to better predict Var 1.
    """
    t = torch.linspace(0, 10, context_len + prediction_len)
    phase = 0.0 # Fixed for test
    freq = 2.0
    
    signal = torch.sin(t * freq + phase)
    
    v0 = signal 
    v1 = v0 * 0.8 + 0.2
    
    # Combined: (2, total_len)
    combined = torch.stack([v0, v1])
    
    # Context
    context = combined[:, :context_len] # (2, 50)
    target = combined[:, context_len:]  # (2, 20)
    
    return context, target

def evaluate(pipeline, context, target, name="Model"):
    """
    Runs prediction and computes MAE against the known target.
    """
    torch.manual_seed(42)
    
    # Input to predict: (1, 2, context_len) -> batch=1, n_vars=2
    # But pipeline.predict expects list or tensor.
    # If we pass (2, context_len), it might treat as 2 univariate or 1 multivariate depending on config/input.
    # Best way for multivariate: List of dicts or list of tensors where each element is (n_vars, len)
    
    # Let's pass a list containing one element of shape (2, context_len)
    input_data = [context] 
    
    forecasts = pipeline.predict(
        input_data,
        prediction_length=target.shape[1],
    )
    
    # Forecasts: List of [ (2, quantiles, pred_len) ]
    # We want mean prediction
    fc_tensor = forecasts[0] # (2, quantiles, 20)
    
    # Mean over quantiles seems to be the way to get point forecast 
    # if we don't have explicit mean output?
    # Chronos2Pipeline returns quantiles. The median (0.5) is usually the point forecast.
    # Inspecting pipeline.py: `predict` returns list of tensors.
    # To get "mean", we can average quantiles or pick median.
    
    # Let's use median (index 4 in default [0.1...0.9])
    # Default quantiles: 0.1, ..., 0.9. Median is at index 4.
    median_idx = 4
    forecast_median = fc_tensor[:, median_idx, :] # (2, 20)
    
    # Compute MAE vs Target (2, 20)
    # We care mostly about Var 1 (index 1) which is the dependent variable?
    # Let's calculate overall MAE.
    mae = torch.abs(forecast_median - target).mean().item()
    
    print(f"[{name}] MAE: {mae:.6f}")
    return mae, forecast_median

def run_demo():
    print("="*60)
    print("CHRONOS-2 ADAPTIVE ATTENTION DEMONSTRATION")
    print("="*60)
    
    # 0. Data
    print("\n[0] Generating Test Data (Correlated Sine Waves)...")
    context, target = get_test_sample()
    print(f"    Context shape: {context.shape}")
    print(f"    Target shape:  {target.shape}")
    
    # 1. Baseline
    print("\n[1] Loading Baseline Chronos-2...")
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cpu",
            torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("    Running Baseline Prediction...")
    baseline_mae, _ = evaluate(pipeline, context, target, name="Baseline")
    
    # 2. Add Adaptive Attention (Untrained)
    print("\n[2] Injecting Adaptive Attention (Untrained)...")
    pipeline = replace_chronos_attention(pipeline)
    
    print("    Running Untrained Adaptive Prediction...")
    untrained_mae, _ = evaluate(pipeline, context, target, name="Untrained")
    
    # Check simple difference
    if abs(untrained_mae - baseline_mae) < 1e-6:
        print("    WARNING: No difference detected. Is the module active?")
    else:
        print("    Difference detected (Random Init).")
        
    # 3. Fine-Tune
    print("\n[3] Fine-Tuning Adaptive Module (Optimized)...")
    # Train longer with lower learning rate for stability
    pipeline = train_model(pipeline, steps=500, learning_rate=5e-5) 
    
    # 4. Trained Evaluation
    print("\n[4] Running Trained Adaptive Prediction...")
    trained_mae, _ = evaluate(pipeline, context, target, name="Trained")
    
    # 5. Summary
    summary = "\n" + "="*60 + "\n"
    summary += "FINAL RESULTS SUMMARY\n"
    summary += "="*60 + "\n"
    summary += f"{'Configuration':<20} | {'MAE':<10} | {'vs Baseline'}\n"
    summary += "-" * 50 + "\n"
    summary += f"{'Baseline':<20} | {baseline_mae:.6f} | {'-':<10}\n"
    summary += f"{'Adaptive (Random)':<20} | {untrained_mae:.6f} | {untrained_mae-baseline_mae:+.6f}\n"
    summary += f"{'Adaptive (Trained)':<20} | {trained_mae:.6f} | {trained_mae-baseline_mae:+.6f}\n"
    summary += "-" * 50 + "\n"
    
    if trained_mae < baseline_mae:
        summary += "\nSUCCESS: Fine-tuning IMPROVED performance!\n"
    elif trained_mae < untrained_mae:
        summary += "\nPARTIAL SUCCESS: Fine-tuning improved over random init, but not baseline yet.\n"
    else:
        summary += "\nNOTE: Training did not improve performance. Needs more data/steps.\n"
        
    print(summary)
    
    # Write to persistence file
    with open("final_results_confirmed.txt", "w") as f:
        f.write(summary)

if __name__ == "__main__":
    run_demo()

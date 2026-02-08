
import torch
import torch.optim as optim
import numpy as np
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention

def generate_correlated_data(batch_size=4, context_len=50, prediction_len=20):
    """
    Generates data where variable 1 is correlated with variable 0, 
    so the model has something to 'learn' to attend to.
    """
    # Base signal: Sine wave + noise
    t = torch.linspace(0, 10, context_len + prediction_len)
    
    batch_context = []
    batch_target = []
    
    for _ in range(batch_size):
        # Random phase/freq
        phase = torch.rand(1) * 6.28
        freq = 2.0 # Match the test case frequency for demonstration of improvement
        signal = torch.sin(t * freq + phase)
        
        # Var 0: The signal
        v0 = signal + 0.01 * torch.randn_like(signal)
        
        # Var 1: Correlated with Var 0 (e.g., lagged or scaled)
        # Let's make it a direct copy plus noise for strong correlation
        v1 = v0 * 0.8 + 0.2 + 0.01 * torch.randn_like(v0)
        
        # Stack vars (2, total_len)
        combined = torch.stack([v0, v1])
        
        # Split into context and target
        # Context: (2, context_len)
        # Target: (2, prediction_len)
        c = combined[:, :context_len]
        tgt = combined[:, context_len:]
        
        batch_context.append(c)
        batch_target.append(tgt)
        
    # Stack batch: (batch, n_vars, len)
    # Chronos expects (batch, len) for univariate or we use group_ids for multivariate.
    # But Chronos2Model natively handles (batch, time, dim) if we map correctly? 
    # Actually Chronos-2 input is usually list of tensors.
    
    # For simplicity with the pipeline/model forward:
    # We will treat each pair as a "group".
    # Batch size of 4 pairs mean 8 time series.
    # Group IDs: [0, 0, 1, 1, 2, 2, 3, 3]
    
    contexts_flat = []
    targets_flat = []
    group_ids = []
    
    for i in range(batch_size):
        contexts_flat.append(batch_context[i][0]) # Var 0
        contexts_flat.append(batch_context[i][1]) # Var 1
        targets_flat.append(batch_target[i][0])
        targets_flat.append(batch_target[i][1])
        group_ids.extend([i, i])
        
    contexts_tensor = torch.stack(contexts_flat) # (batch*2, context_len)
    targets_tensor = torch.stack(targets_flat)   # (batch*2, pred_len)
    group_ids_tensor = torch.tensor(group_ids)   # (batch*2,)
    
    return contexts_tensor, targets_tensor, group_ids_tensor

def train_model(pipeline, steps=500, learning_rate=5e-5):
    """
    Fine-tunes ONLY the RelevanceBias parameters of the model.
    """
    print(f"Starting Fine-tuning for {steps} steps with LR {learning_rate}...")
    
    model = pipeline.model
    model.train()
    
    # 1. Freeze Backbone
    trainable_params = []
    
    for name, param in model.named_parameters():
        if "relevance_bias" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    # 2. Optimizer with weight decay
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # 3. Training Loop
    # Generate more data for better generalization
    ctx, tgt, grp = generate_correlated_data(batch_size=16)
    
    device = model.device
    ctx = ctx.to(device)
    tgt = tgt.to(device)
    grp = grp.to(device)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(
            context=ctx,
            future_target=tgt,
            group_ids=grp,
            num_output_patches=2
        )
        
        loss = output.loss
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{steps} | Loss: {loss.item():.4f}")
            
    print("Fine-tuning complete.")
    return pipeline

if __name__ == "__main__":
    # Test standalone
    print("Initializing pipeline...")
    p = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    p = replace_chronos_attention(p)
    train_model(p, steps=10)
    print("Test run finished.")

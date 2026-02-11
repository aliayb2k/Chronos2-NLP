"""
Training script for Adaptive Attention Extension on ETTm1 dataset.

This script trains the RelevanceBias module while keeping the Chronos-2 backbone frozen.
"""

import torch
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from prepare_ett_data import load_ett_multivariate, prepare_training_batches, create_windowed_samples


def validate(pipeline, val_samples, device='cpu'):
    """
    Compute validation loss on validation set.
    Matches training data format exactly.
    """
    model = pipeline.model
    model.eval()
    
    total_loss = 0.0
    num_samples = min(len(val_samples), 15)  # Balanced: accurate yet fast
    
    with torch.no_grad():
        for i in range(num_samples):
            context, target = val_samples[i]  # (7, 96) each
            
            # Match training format: treat each variable as separate series
            n_vars = context.shape[0]
            
            # Flatten into batch dimension
            ctx_batch = context  # (7, 96)
            tgt_batch = target    # (7, 96)
            
            # Group IDs: all from same sample (group 0)
            group_ids = torch.zeros(n_vars, dtype=torch.long).to(device)
            
            # Move to device
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Calculate required patches
            pred_len = tgt_batch.shape[1]
            num_patches = (pred_len + 15) // 16  # 96 -> 6
            
            try:
                output = model(
                    context=ctx_batch,
                    future_target=tgt_batch,
                    group_ids=group_ids,
                    num_output_patches=num_patches
                )
                total_loss += output.loss.item()
            except Exception as e:
                print(f"  Val error {i}: {e}")
                continue
    
    model.train()
    return total_loss / num_samples if num_samples > 0 else float('inf')


def train_on_ett(
    output_dir='./results',
    checkpoint_dir='./checkpoints',
    max_steps=10000,
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    val_every=100,
    early_stopping_patience=20,
    device=None
):
    """
    Train adaptive attention module on ETTm1 dataset.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Auto-detect device (prioritize MPS for Apple Silicon)
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
            print("  Using Apple GPU (MPS) acceleration!")
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print("=" * 70)
    print("ADAPTIVE ATTENTION TRAINING ON ETTm1")
    print("=" * 70)
    print(f"\\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Validation every: {val_every} steps")
    print(f"  Early stopping patience: {early_stopping_patience}")
    
    # Load data
    print(f"\\n[1/5] Loading ETTm1 dataset...")
    train_data, val_data, test_data = load_ett_multivariate()
    
    val_samples = create_windowed_samples(val_data, context_length=96, prediction_length=96)
    print(f"  Validation samples: {len(val_samples)}")
    
    # Prepare training batches
    print(f"\\n[2/5] Preparing training batches...")
    batches = prepare_training_batches(
        train_data,
        batch_size=batch_size,
        context_length=96,
        prediction_length=96
    )
    print(f"  Total batches: {len(batches)}")
    
    # Load model
    print(f"\\n[3/5] Loading Chronos-2 model...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32
    )
    
    # Inject adaptive attention
    print(f"\\n[4/5] Injecting adaptive attention...")
    pipeline = replace_chronos_attention(pipeline)
    
    model = pipeline.model
    model.train()
    
    # Freeze backbone, train only relevance_bias
    trainable_params = []
    frozen_params = 0
    for name, param in model.named_parameters():
        if "relevance_bias" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {trainable_count:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    print(f"\\n[5/5] Starting training...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_log = {
        'steps': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    step = 0
    epoch = 0
    
    while step < max_steps:
        epoch += 1
        print(f"\\nEpoch {epoch}")
        
        # Shuffle batches
        batch_indices = torch.randperm(len(batches))
        
        for batch_idx in batch_indices:
            if step >= max_steps:
                break
            
            ctx, tgt, grp = batches[batch_idx.item()]
            ctx = ctx.to(device)
            tgt = tgt.to(device)
            grp = grp.to(device)
            
            optimizer.zero_grad()
            
            # Calculate required patches (Chronos-2 uses patch_size=16)
            pred_len = tgt.shape[1]
            num_patches = (pred_len + 15) // 16  # Ceiling division: 96 -> 6 patches
            
            try:
                # Forward pass
                output = model(
                    context=ctx,
                    future_target=tgt,
                    group_ids=grp,
                    num_output_patches=num_patches
                )
                
                loss = output.loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                
                # Log
                training_log['steps'].append(step)
                training_log['train_loss'].append(loss.item())
                training_log['learning_rate'].append(learning_rate)
                
                if (step + 1) % 50 == 0:
                    print(f"  Step {step+1}/{max_steps} | Loss: {loss.item():.4f}")
                
                # Validation
                if (step + 1) % val_every == 0:
                    print(f"\\n  Validating at step {step+1}...")
                    val_loss = validate(pipeline, val_samples, device)
                    training_log['val_loss'].append(val_loss)
                    
                    print(f"  Validation Loss: {val_loss:.4f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best checkpoint
                        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, checkpoint_path)
                        print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
                    else:
                        patience_counter += 1
                        print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
                        
                        if patience_counter >= early_stopping_patience:
                            print(f"\\n  Early stopping triggered at step {step+1}")
                            break
                
                step += 1
                
            except Exception as e:
                print(f"  Training error at step {step}: {e}")
                step += 1
                continue
        
        if patience_counter >= early_stopping_patience:
            break
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint_path)
    
    # Save training log
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps: {step}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"  Training log: {log_path}")
    
    return pipeline, training_log


if __name__ == "__main__":
    # Run training
    pipeline, log = train_on_ett(
        max_steps=10000,
        batch_size=32,
        learning_rate=5e-5,
        val_every=100,
        early_stopping_patience=20
    )
    
    print("\\n✓ Training script completed successfully!")

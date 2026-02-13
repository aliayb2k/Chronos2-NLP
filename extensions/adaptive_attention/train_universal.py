"""
Universal training script for adaptive attention on any fev-bench dataset.
Supports ETTm1, ETTh1, Electricity, Traffic, and Weather.

Usage:
    python train_universal.py --dataset ETTm1 --pred_len 96
    python train_universal.py --dataset electricity --pred_len 96 --max_steps 5000
"""

import argparse
import os
import json
import torch
import torch.optim as optim
from chronos import Chronos2Pipeline
from adaptive_attention import replace_chronos_attention
from load_fev_datasets import load_fev_dataset, prepare_test_samples


def validate(pipeline, val_samples, device='cpu'):
    """Compute validation loss on validation set."""
    model = pipeline.model
    model.eval()
    
    total_loss = 0.0
    num_samples = min(len(val_samples), 15)
    
    with torch.no_grad():
        for i in range(num_samples):
            context, target = val_samples[i]
            n_vars = context.shape[0]
            
            group_ids = torch.zeros(n_vars, dtype=torch.long).to(device)
           
            context = context.to(device)
            target = target.to(device)
            
            pred_len = target.shape[1]
            num_patches = (pred_len + 15) // 16
            
            try:
                output = model(
                    context=context,
                    future_target=target,
                    group_ids=group_ids,
                    num_output_patches=num_patches
                )
                total_loss += output.loss.item()
            except Exception as e:
                print(f"  Val error {i}: {e}")
                continue
    
    model.train()
    return total_loss / num_samples if num_samples > 0 else float('inf')


def train_on_dataset(
    dataset_name='ETTm1',
    prediction_length=96,
    context_length=512,
    max_steps=10000,
    batch_size=32,
    learning_rate=5e-5,
    weight_decay=0.01,
    validation_every=100,
    patience=10000,  # Effectively disabled - will train all steps
    device=None,
    output_dir='./results',
    checkpoint_dir='./checkpoints'
):
    """Train adaptive attention on specified dataset."""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
            print("  Using Apple GPU (MPS) acceleration!")
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print("=" * 70)
    print(f"ADAPTIVE ATTENTION TRAINING ON {dataset_name.upper()}")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Context length: {context_length}")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Validation every: {validation_every} steps")
    print(f"  Early stopping patience: {patience}")
    print()
    
    # [1/5] Load dataset
    print(f"[1/5] Loading {dataset_name} dataset...")
    data = load_fev_dataset(dataset_name, context_length, prediction_length)
    
    print(f"  Variables: {data['n_variables']}")
    print(f"  Train shape: {data['train_data'].shape}")
    print(f"  Val shape: {data['val_data'].shape}")  
    print(f"  Test shape: {data['test_data'].shape}")
    print()
    
    # [2/5] Prepare training/validation samples
    print("[2/5] Preparing training batches...")
    train_samples = prepare_test_samples(
        data['train_data'],
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=1000
    )
    
    val_samples = prepare_test_samples(
        data['val_data'],
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=50
    )
    
    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print()
    
    # [3/5] Load Chronos-2
    print("[3/5] Loading Chronos-2 model...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.float32


    )
    print()
    
    # [4/5] Inject adaptive attention
    print("[4/5] Injecting adaptive attention...")
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
    print()
    
    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    # [5/5] Training loop
    print("[5/5] Starting training...")
    print("=" * 70)
    print()
    
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
        print(f"Epoch {epoch}")
        
        # Shuffle samples
        indices = torch.randperm(len(train_samples))
        
        for idx in indices:
            if step >= max_steps:
                break
            
            context, target = train_samples[idx.item()]
            n_vars = context.shape[0]
            
            context = context.to(device)
            target = target.to(device)
            group_ids = torch.zeros(n_vars, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            pred_len = target.shape[1]
            num_patches = (pred_len + 15) // 16
            
            try:
                output = model(
                    context=context,
                    future_target=target,
                    group_ids=group_ids,
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
                if (step + 1) % validation_every == 0:
                    print(f"\n  Validating at step {step+1}...")
                    val_loss = validate(pipeline, val_samples, device)
                    training_log['val_loss'].append(val_loss)
                    
                    print(f"  Validation Loss: {val_loss:.4f}")
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save best checkpoint
                        checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_best.pt')
                        torch.save({
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                        }, checkpoint_path)
                        print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
                    else:
                        patience_counter += 1
                        print(f"  No improvement ({patience_counter}/{patience})")
                        
                        if patience_counter >= patience:
                            print(f"\n  Early stopping triggered at step {step+1}")
                            break
                
                step += 1
                
            except Exception as e:
                print(f"  Training error at step {step}: {e}")
                step += 1
                continue
        
        if patience_counter >= patience:
            break
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f'{dataset_name}_final.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_checkpoint_path)
    
    # Save training log
    log_path = os.path.join(output_dir, f'{dataset_name}_training_log.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total steps: {step}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model: {checkpoint_path}")
    print(f"  Training log: {log_path}")
    
    return pipeline, training_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train adaptive attention on fev-bench datasets")
    
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ETTm1', 'ETTh1', 'electricity', 'hospital', 'epf_de', 'rossmann'],
                        help='Dataset to train on')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon length')
    parser.add_argument('--context_len', type=int, default=512,
                        help='Context window length')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum training steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps', None],
                        help='Device to use (auto-detect if None)')
    
    args = parser.parse_args()
    
    train_on_dataset(
        dataset_name=args.dataset,
        prediction_length=args.pred_len,
        context_length=args.context_len,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    print("\n✓ Training script completed successfully!")

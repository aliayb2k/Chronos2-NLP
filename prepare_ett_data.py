"""
Data preparation module for ETTm1 multivariate time series dataset.

This module loads the ETTm1 dataset from Hugging Face and prepares it for
training the adaptive attention mechanism.
"""

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from typing import Tuple, List


def load_ett_multivariate(
    context_length: int = 96,
    prediction_length: int = 96,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load ETTm1 dataset and prepare for multivariate training.
    
    Args:
        context_length: Number of historical timesteps to use as context
        prediction_length: Number of future timesteps to predict
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation (test = 1 - train - val)
    
    Returns:
        train_data: Training data tensor (n_samples, n_vars, sequence_len)
        val_data: Validation data tensor
        test_data: Test data tensor
    """
    print("Loading ETTm1 dataset from Hugging Face...")
    
    # Load dataset
    dataset = load_dataset("autogluon/fev_datasets", "ETT_15T")
    df_raw = dataset["train"].to_pandas()
    
    # Filter for ETTm1 only
    df_raw = df_raw[df_raw["id"] == "ETTm1"].copy()
    
    # Variable columns (7 variables)
    variables = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    
    # Extract and convert to proper format
    print("Processing multivariate time series...")
    
    # Initialize list to hold series for each variable
    series_list = []
    
    for var in variables:
        # Extract series (should be numpy array)
        series = df_raw[var].iloc[0]
        
        # Convert to numpy if needed
        if isinstance(series, list):
            series = np.array(series)
        
        series_list.append(series)
    
    # Stack into (n_vars, total_length) array
    multivariate_data = np.stack(series_list, axis=0)  # Shape: (7, 69680)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    print(f"Variables: {variables}")
    
    # Calculate split indices
    total_length = multivariate_data.shape[1]
    train_end = int(total_length * train_ratio)
    val_end = int(total_length * (train_ratio + val_ratio))
    
    print(f"\nData splits:")
    print(f"  Train: 0 to {train_end} ({train_end} timesteps)")
    print(f"  Val: {train_end} to {val_end} ({val_end - train_end} timesteps)")
    print(f"  Test: {val_end} to {total_length} ({total_length - val_end} timesteps)")
    
    # Split data
    train_raw = multivariate_data[:, :train_end]
    val_raw = multivariate_data[:, train_end:val_end]
    test_raw = multivariate_data[:, val_end:]
    
    # Convert to torch tensors
    train_data = torch.from_numpy(train_raw).float()
    val_data = torch.from_numpy(val_raw).float()
    test_data = torch.from_numpy(test_raw).float()
    
    return train_data, val_data, test_data


def create_windowed_samples(
    data: torch.Tensor,
    context_length: int = 96,
    prediction_length: int = 96,
    stride: int = None
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create sliding window samples for training/evaluation.
    
    Args:
        data: Input tensor (n_vars, total_length)
        context_length: Length of context window
        prediction_length: Length of prediction window
        stride: Step size between windows (default: prediction_length for non-overlapping)
    
    Returns:
        List of (context, target) tuples
    """
    if stride is None:
        stride = prediction_length  # Non-overlapping windows by default
    
    n_vars, total_length = data.shape
    window_size = context_length + prediction_length
    samples = []
    
    for start_idx in range(0, total_length - window_size + 1, stride):
        context = data[:, start_idx:start_idx + context_length]
        target = data[:, start_idx + context_length:start_idx + window_size]
        samples.append((context, target))
    
    return samples


def prepare_training_batches(
    train_data: torch.Tensor,
    batch_size: int = 32,
    context_length: int = 96,
    prediction_length: int = 96
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Prepare batched training data for Chronos-2 format.
    
    Returns:
        List of (context_batch, target_batch, group_ids) tuples
    """
    # Create windowed samples
    samples = create_windowed_samples(
        train_data,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=prediction_length // 2  # 50% overlap for more training data
    )
    
    print(f"Created {len(samples)} training windows")
    
    # Group into batches
    batches = []
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        
        # Flatten: each variable becomes a separate series in the batch
        contexts = []
        targets = []
        group_ids = []
        
        for sample_idx, (context, target) in enumerate(batch_samples):
            n_vars = context.shape[0]
            for var_idx in range(n_vars):
                contexts.append(context[var_idx])
                targets.append(target[var_idx])
                group_ids.append(sample_idx)
        
        # Stack into tensors
        context_batch = torch.stack(contexts)  # (batch_size * n_vars, context_len)
        target_batch = torch.stack(targets)    # (batch_size * n_vars, pred_len)
        group_ids_tensor = torch.tensor(group_ids)
        
        batches.append((context_batch, target_batch, group_ids_tensor))
    
    return batches


if __name__ == "__main__":
    # Test the data loader
    print("=" * 60)
    print("Testing ETTm1 Data Loader")
    print("=" * 60)
    
    # Load data
    train, val, test = load_ett_multivariate()
    
    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape: {val.shape}")
    print(f"Test shape: {test.shape}")
    
    # Create test samples
    test_samples = create_windowed_samples(test, context_length=96, prediction_length=96)
    print(f"\nCreated {len(test_samples)} non-overlapping test samples")
    
    # Show sample
    context, target = test_samples[0]
    print(f"\nSample context shape: {context.shape}")
    print(f"Sample target shape: {target.shape}")
    
    # Test batch creation
    print("\n" + "=" * 60)
    print("Testing Batch Creation")
    print("=" * 60)
    
    batches = prepare_training_batches(train, batch_size=4)
    print(f"Created {len(batches)} training batches")
    
    ctx, tgt, grp = batches[0]
    print(f"\nFirst batch:")
    print(f"  Context shape: {ctx.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Group IDs shape: {grp.shape}")
    print(f"  Unique groups: {torch.unique(grp).tolist()}")
    
    print("\n✓ Data preparation successful!")

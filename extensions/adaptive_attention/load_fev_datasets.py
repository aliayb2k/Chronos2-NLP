"""
Unified data loader for fev-bench datasets.
Loads ETTm1, ETTh1, Electricity, Traffic, and Weather datasets.
Uses autogluon/fev_datasets (same as prepare_ett_data.py)
"""

import torch
import numpy as np
from datasets import load_dataset

def load_fev_dataset(dataset_name, context_length=512, prediction_length=96):
    """
    Load a dataset from fev-bench.
    
    Args:
        dataset_name: Name of dataset (ETTm1, ETTh1, electricity, traffic, weather)
        context_length: Context window length
        prediction_length: Forecast horizon
    
    Returns:
        Dictionary with train/val/test data and metadata
    """
    
    if dataset_name == 'ETTm1':
        return _load_ett_dataset('ETTm1', 'ETT_15T', '15min', prediction_length)
    elif dataset_name == 'ETTh1':
        return _load_ett_dataset('ETTh1', 'ETT_1H', 'H', prediction_length)
    elif dataset_name == 'electricity':
        return _load_electricity(context_length, prediction_length)
    elif dataset_name == 'traffic':
        return _load_traffic(context_length, prediction_length)
    elif dataset_name == 'weather':
        return _load_weather(context_length, prediction_length)
    elif dataset_name == 'hospital':
        return _load_hospital(context_length, prediction_length)
    elif dataset_name == 'epf_de':
        return _load_epf_de(context_length, prediction_length)
    elif dataset_name == 'rossmann':
        return _load_rossmann(context_length, prediction_length)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_ett_dataset(dataset_id, subset_name, freq, prediction_length):
    """Generic ETT loader for ETTm1 and ETTh1"""
    print(f"Loading {dataset_id} from autogluon/fev_datasets...")
    
    # Load from HuggingFace
    dataset = load_dataset("autogluon/fev_datasets", subset_name)
    df_raw = dataset["train"].to_pandas()
    
    # Filter for specific ETT dataset
    df_raw = df_raw[df_raw["id"] == dataset_id].copy()
    
    # Variable columns (7 variables)
    variables = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    
    # Extract multivariate series
    print("Processing multivariate time series...")
    series_list = []
    
    for var in variables:
        series = df_raw[var].iloc[0]
        if isinstance(series, list):
            series = np.array(series)
        series_list.append(series)
    
    # Stack to (n_vars, timesteps)
    multivariate_data = np.stack(series_list, axis=0)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    print(f"Variables: {variables}")
    
    # 60/20/20 split
    n_vars, total_len = multivariate_data.shape
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"\nData splits:")
    print(f"  Train: 0 to {train_end} ({train_end} timesteps)")
    print(f"  Val: {train_end} to {val_end} ({val_end - train_end} timesteps)")
    print(f"  Test: {val_end} to {total_len} ({total_len - val_end} timesteps)")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': variables,
        'freq': freq,
        'prediction_length': prediction_length
    }


def _load_electricity(context_length, prediction_length):
    """Load ENTSO-E Electricity dataset (6 European zones, hourly)"""
    print("Loading ENTSO-E Electricity from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "entsoe_1H")
    df_raw = dataset["train"].to_pandas()
    
    if len(df_raw) == 0:
        raise ValueError("Electricity dataset is empty")
    
    print(f"Found {len(df_raw)} electricity zones")
    
    series_list = []
    zone_names = []
    
    for idx in range(len(df_raw)):
        zone_id = df_raw['id'].iloc[idx]
        target_series = df_raw['target'].iloc[idx]
        
        if isinstance(target_series, list):
            target_series = np.array(target_series)
        
        series_list.append(target_series)
        zone_names.append(str(zone_id))
    
    multivariate_data = np.stack(series_list, axis=0)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    print(f"Variables: {zone_names}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': zone_names,
        'freq': 'H',
        'prediction_length': prediction_length
    }


def _load_traffic(context_length, prediction_length):
    """Load Traffic dataset (862 sensors)"""
    print("Loading Traffic from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "traffic")
    df_raw = dataset["train"].to_pandas()
    
    if len(df_raw) == 0:
        raise ValueError("Traffic dataset is empty")
    
    exclude_cols = {'id', 'timestamp', 'item_id'}
    var_columns = [col for col in df_raw.columns if col not in exclude_cols]
    
    print(f"Found {len(var_columns)} traffic sensors")
    
    series_list = []
    for col in var_columns:
        series = df_raw[col].iloc[0]
        if isinstance(series, list):
            series = np.array(series)
        series_list.append(series)
    
    multivariate_data = np.stack(series_list, axis=0)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': var_columns,
        'freq': 'H',
        'prediction_length': 24  # Traffic standard: 24-hour horizon
    }


def _load_weather(context_length, prediction_length):
    """Load Weather dataset (21 meteorological variables)"""
    print("Loading Weather from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "weather")
    df_raw = dataset["train"].to_pandas()
    
    if len(df_raw) == 0:
        raise ValueError("Weather dataset is empty")
    
    exclude_cols = {'id', 'timestamp', 'item_id', 'date'}
    var_columns = [col for col in df_raw.columns if col not in exclude_cols]
    
    print(f"Found {len(var_columns)} weather variables")
    
    series_list = []
    for col in var_columns:
        series = df_raw[col].iloc[0]
        if isinstance(series, list):
            series = np.array(series)
        series_list.append(series)
    
    multivariate_data = np.stack(series_list, axis=0)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': var_columns,
        'freq': '10min',
        'prediction_length': prediction_length
    }


def _load_hospital(context_length, prediction_length):
    """Load Hospital Admissions dataset (8 Saudi hospitals, daily)"""
    print("Loading Hospital Admissions from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "hospital_admissions_1D")
    df_raw = dataset["train"].to_pandas()
    
    if len(df_raw) == 0:
        raise ValueError("Hospital Admissions dataset is empty")
    
    print(f"Found {len(df_raw)} hospitals")
    
    series_list = []
    hospital_names = []
    
    for idx in range(len(df_raw)):
        hospital_id = df_raw['id'].iloc[idx]
        target_series = df_raw['target'].iloc[idx]
        
        if isinstance(target_series, list):
            target_series = np.array(target_series)
        
        series_list.append(target_series)
        hospital_names.append(str(hospital_id))
    
    # Truncate all series to minimum length (they may differ slightly)
    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]
    
    multivariate_data = np.stack(series_list, axis=0)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    print(f"Hospitals: {hospital_names}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': hospital_names,
        'freq': 'D',
        'prediction_length': prediction_length
    }


def prepare_test_samples(test_data, context_length=512, prediction_length=96, num_samples=50):
    """
    Create test samples for evaluation.
    
    Args:
        test_data: (n_vars, timesteps) numpy array
        context_length: Context window size
        prediction_length: Prediction horizon
        num_samples: Number of samples to create
    
    Returns:
        List of (context, target) tuples, each torch.Tensor of shape (n_vars, length)
    """
    n_vars, total_len = test_data.shape
    window_len = context_length + prediction_length
    
    samples = []
    max_start = total_len - window_len
    
    if max_start <= 0:
        print(f"Warning: Test data too short ({total_len}) for window ({window_len})")
        return []
    
    # Evenly spaced samples
    step = max(1, max_start // num_samples)
    
    for i in range(0, max_start, step):
        if len(samples) >= num_samples:
            break
        
        window = test_data[:, i:i+window_len]
        context = window[:, :context_length]
        target = window[:, context_length:context_length+prediction_length]
        
        samples.append((
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        ))
    
    return samples


def _load_epf_de(context_length, prediction_length):
    """Load EPF-DE (Electricity Price Forecasting - Germany) with covariates"""
    print("Loading EPF-DE from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "epf_de")
    df_raw = dataset["train"].to_pandas()
    
    if len(df_raw) == 0:
        raise ValueError("EPF-DE dataset is empty")
    
    # Target and covariate columns
    var_columns = ['target', 'Ampirion Load Forecast', 'PV+Wind Forecast']
    print(f"Variables: {var_columns}")
    
    series_list = []
    for col in var_columns:
        series = df_raw[col].iloc[0]
        if isinstance(series, list):
            series = np.array(series)
        series_list.append(series)
    
    multivariate_data = np.stack(series_list, axis=0) # (3, 52416)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': var_columns,
        'freq': 'H',
        'prediction_length': prediction_length
    }


def _load_rossmann(context_length, prediction_length):
    """Load Rossmann (Daily Sales) dataset - selecting first 10 stores"""
    print("Loading Rossmann from autogluon/fev_datasets...")
    
    dataset = load_dataset("autogluon/fev_datasets", "rossmann_1D")
    df_raw = dataset["train"].to_pandas()
    
    # Selecting first 10 stores to keep attention matrix small
    n_stores = 10
    df_subset = df_raw.iloc[:n_stores]
    
    series_list = []
    store_ids = []
    
    for idx in range(len(df_subset)):
        sid = df_subset['id'].iloc[idx]
        sales = df_subset['Sales'].iloc[idx]
        if isinstance(sales, list):
            sales = np.array(sales)
        series_list.append(sales)
        store_ids.append(f"Store_{sid}")
    
    # Truncate to min length
    min_len = min(len(s) for s in series_list)
    series_list = [s[:min_len] for s in series_list]
    
    multivariate_data = np.stack(series_list, axis=0) # (10, 942)
    n_vars, total_len = multivariate_data.shape
    
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)
    
    print(f"Multivariate data shape: {multivariate_data.shape}")
    
    return {
        'train_data': multivariate_data[:, :train_end],
        'val_data': multivariate_data[:, train_end:val_end],
        'test_data': multivariate_data[:, val_end:],
        'n_variables': n_vars,
        'variable_names': store_ids,
        'freq': 'D',
        'prediction_length': prediction_length
    }


if __name__ == "__main__":
    # Test all datasets
    datasets = ['ETTm1', 'ETTh1', 'electricity', 'traffic', 'weather']
    
    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Testing {ds_name}")
        print('='*70)
        
        try:
            data = load_fev_dataset(ds_name)
            print(f"✓ Loaded successfully!")
            print(f"  Variables: {data['n_variables']}")
            print(f"  Train shape: {data['train_data'].shape}")
            print(f"  Val shape: {data['val_data'].shape}")
            print(f"  Test shape: {data['test_data'].shape}")
            print(f"  Frequency: {data['freq']}")
            print(f"  Prediction length: {data['prediction_length']}")
            
            # Test sample creation
            samples = prepare_test_samples(
                data['test_data'],
                context_length=512,
                prediction_length=data['prediction_length'],
                num_samples=5
            )
            print(f"  Test samples created: {len(samples)}")
            if len(samples) > 0:
                ctx, tgt = samples[0]
                print(f"  Sample shapes: context={ctx.shape}, target={tgt.shape}")
            
        except Exception as e:
            print(f"✗ Error loading {ds_name}: {e}")
            import traceback
            traceback.print_exc()

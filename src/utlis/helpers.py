import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import yaml
from datetime import datetime

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_directory(directory: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    Returns:
        Path object for the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_results(results: Dict, output_dir: str, filename: str) -> None:
    """
    Save analysis results to JSON file.
    
    Args:
        results: Dictionary of results to save
        output_dir: Output directory
        filename: Name of output file
    """
    output_path = ensure_directory(output_dir) / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    Returns:
        True if validation passes, raises ValueError otherwise
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for array of values.
    
    Args:
        data: Numpy array of values
    Returns:
        Dictionary of statistics
    """
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }

def smooth_data(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to data.
    
    Args:
        data: Input data array
        window_size: Size of moving average window
    Returns:
        Smoothed data array
    """
    return pd.Series(data).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp for file naming.
    
    Args:
        timestamp: Optional datetime object, uses current time if None
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime('%Y%m%d_%H%M%S')

def calculate_cycle_metrics(charge_data: np.ndarray, 
                          discharge_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for a single charge-discharge cycle.
    
    Args:
        charge_data: Array of charging measurements
        discharge_data: Array of discharging measurements
    Returns:
        Dictionary of cycle metrics
    """
    return {
        'coulombic_efficiency': abs(np.sum(discharge_data) / np.sum(charge_data)),
        'charge_time': len(charge_data),
        'discharge_time': len(discharge_data),
        'total_time': len(charge_data) + len(discharge_data)
    }

def detect_anomalies(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect anomalies using z-score method.
    
    Args:
        data: Input data array
        threshold: Z-score threshold for anomaly detection
    Returns:
        Boolean array indicating anomalies
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

def export_to_csv(df: pd.DataFrame, output_path: str, 
                 include_timestamp: bool = True) -> None:
    """
    Export DataFrame to CSV with optional timestamp in filename.
    
    Args:
        df: DataFrame to export
        output_path: Path for output file
        include_timestamp: Whether to include timestamp in filename
    """
    if include_timestamp:
        path = Path(output_path)
        timestamped_path = path.parent / f"{path.stem}_{format_timestamp()}{path.suffix}"
        df.to_csv(timestamped_path, index=False)
    else:
        df.to_csv(output_path, index=False)

def parse_time_string(time_str: str) -> pd.Timestamp:
    """
    Parse time string to timestamp, handling multiple formats.
    
    Args:
        time_str: Time string to parse
    Returns:
        Pandas Timestamp object
    """
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(time_str, format=fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse time string: {time_str}")

def interpolate_missing(df: pd.DataFrame, 
                       columns: List[str],
                       method: str = 'linear') -> pd.DataFrame:
    """
    Interpolate missing values in specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to interpolate
        method: Interpolation method
    Returns:
        DataFrame with interpolated values
    """
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].interpolate(method=method)
    return result
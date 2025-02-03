import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter

class BatteryDataPreprocessor:
    """
    Preprocessor for battery cycling data.
    Handles cleaning, normalization, and feature extraction.
    """
    
    def __init__(self, voltage_range: Tuple[float, float] = (2.5, 4.2)):
        self.voltage_limits = voltage_range
        self.capacity_threshold = 0.1  # Minimum capacity threshold
        self.outlier_std_threshold = 3.0  # Standard deviations for outlier detection
        
    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline for raw battery data.
        
        Args:
            df: Raw data DataFrame with columns: cycle, voltage, current, temperature
        Returns:
            Processed DataFrame with additional features
        """
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Basic cleaning
        processed_df = self._remove_invalid_data(processed_df)
        processed_df = self._handle_missing_values(processed_df)
        
        # Feature extraction
        processed_df = self._calculate_capacity(processed_df)
        processed_df = self._add_cycle_features(processed_df)
        processed_df = self._calculate_resistance(processed_df)
        
        # Normalize features
        processed_df = self._normalize_features(processed_df)
        
        return processed_df
    
    def _remove_invalid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove data points outside valid ranges."""
        mask = (
            (df['voltage'].between(*self.voltage_limits)) &
            (df['capacity'] > self.capacity_threshold) &
            (df['temperature'].between(-20, 60))  # Reasonable temperature range
        )
        return df[mask].copy()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # For critical columns, remove rows with missing values
        critical_columns = ['cycle', 'voltage', 'current']
        df = df.dropna(subset=critical_columns)
        
        # For other columns, interpolate missing values
        df = df.interpolate(method='linear', limit_direction='both')
        
        return df
    
    def _calculate_capacity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate discharge capacity for each cycle."""
        df['capacity'] = df.groupby('cycle').apply(
            lambda x: self._integrate_current(x['current'], x['time'])
        )
        return df
    
    def _add_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cycle-level features."""
        # Calculate cycle-level metrics
        cycle_stats = df.groupby('cycle').agg({
            'voltage': ['mean', 'min', 'max'],
            'temperature': ['mean', 'max', 'std'],
            'current': ['mean', 'min', 'max']
        })
        
        # Flatten column names
        cycle_stats.columns = ['_'.join(col).strip() for col in cycle_stats.columns.values]
        
        # Merge back to original dataframe
        df = df.merge(cycle_stats, left_on='cycle', right_index=True)
        
        return df
    
    def _calculate_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate internal resistance using voltage and current data."""
        # Calculate resistance during discharge
        discharge_mask = df['current'] < 0
        df.loc[discharge_mask, 'resistance'] = (
            -df.loc[discharge_mask, 'voltage'] / df.loc[discharge_mask, 'current']
        )
        
        # Smooth resistance values
        df['resistance'] = savgol_filter(
            df['resistance'].fillna(method='ffill'), 
            window_length=21, 
            polyorder=3
        )
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features to [0,1] range."""
        features_to_normalize = [
            'voltage', 'current', 'temperature', 'capacity', 'resistance'
        ]
        
        for feature in features_to_normalize:
            if feature in df.columns:
                df[f'{feature}_normalized'] = (
                    df[feature] - df[feature].min()
                ) / (df[feature].max() - df[feature].min())
        
        return df
    
    @staticmethod
    def _integrate_current(current: pd.Series, time: pd.Series) -> float:
        """
        Calculate capacity by integrating current over time.
        
        Args:
            current: Current measurements
            time: Time stamps
        Returns:
            Integrated capacity
        """
        # Convert to numpy arrays for faster computation
        current_array = current.to_numpy()
        time_array = time.to_numpy()
        
        # Calculate time differences
        dt = np.diff(time_array)
        
        # Calculate capacity using trapezoidal integration
        capacity = np.trapz(current_array[:-1], dx=dt)
        
        return abs(capacity)
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Detect outliers in specified columns using z-score method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers
        Returns:
            DataFrame with outlier indicators
        """
        outlier_mask = pd.DataFrame(index=df.index)
        
        for column in columns:
            if column in df.columns:
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outlier_mask[f'{column}_outlier'] = z_scores > self.outlier_std_threshold
        
        return pd.concat([df, outlier_mask], axis=1)
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract key features from processed data.
        
        Args:
            df: Processed DataFrame
        Returns:
            Dictionary of extracted features
        """
        features = {
            'mean_capacity': df['capacity'].mean(),
            'capacity_fade_rate': self._calculate_fade_rate(df['capacity']),
            'mean_temperature': df['temperature'].mean(),
            'temperature_variance': df['temperature'].var(),
            'mean_resistance': df['resistance'].mean(),
            'resistance_increase': self._calculate_resistance_increase(df['resistance']),
            'cycle_count': df['cycle'].nunique()
        }
        
        return features
    
    @staticmethod
    def _calculate_fade_rate(capacity: pd.Series) -> float:
        """Calculate capacity fade rate."""
        # Fit linear regression to capacity vs cycle number
        cycles = np.arange(len(capacity))
        coefficients = np.polyfit(cycles, capacity, deg=1)
        return abs(coefficients[0])  # Return absolute fade rate
    
    @staticmethod
    def _calculate_resistance_increase(resistance: pd.Series) -> float:
        """Calculate resistance increase rate."""
        initial_resistance = resistance.iloc[0]
        final_resistance = resistance.iloc[-1]
        return (final_resistance - initial_resistance) / initial_resistance
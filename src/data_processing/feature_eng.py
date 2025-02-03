import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple
from scipy.stats import skew, kurtosis

class BatteryFeatureExtractor:
    """
    Extract advanced features from battery cycling data for degradation analysis.
    """
    
    def __init__(self):
        self.feature_names = []
        self.voltage_thresholds = np.linspace(3.0, 4.2, 10)
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all available features from the battery data.
        
        Args:
            df: DataFrame containing battery cycling data
        Returns:
            DataFrame with extracted features
        """
        features = {}
        
        # Capacity-based features
        features.update(self._extract_capacity_features(df))
        
        # Voltage-based features
        features.update(self._extract_voltage_features(df))
        
        # Temperature-based features
        features.update(self._extract_temperature_features(df))
        
        # Differential features
        features.update(self._extract_differential_features(df))
        
        # Combined metrics
        features.update(self._calculate_combined_metrics(df))
        
        return pd.DataFrame([features])
    
    def _extract_capacity_features(self, df: pd.DataFrame) -> Dict:
        """Extract features based on capacity measurements."""
        capacity_data = df['capacity'].values
        
        features = {
            'capacity_mean': np.mean(capacity_data),
            'capacity_std': np.std(capacity_data),
            'capacity_skew': skew(capacity_data),
            'capacity_kurtosis': kurtosis(capacity_data),
            'capacity_range': np.ptp(capacity_data),
            'capacity_q25': np.percentile(capacity_data, 25),
            'capacity_q75': np.percentile(capacity_data, 75),
            'capacity_iqr': np.percentile(capacity_data, 75) - np.percentile(capacity_data, 25)
        }
        
        # Calculate capacity fade rate
        cycle_numbers = df['cycle'].values
        slope, intercept = np.polyfit(cycle_numbers, capacity_data, 1)
        features['capacity_fade_rate'] = abs(slope)
        
        return features
    
    def _extract_voltage_features(self, df: pd.DataFrame) -> Dict:
        """Extract features based on voltage curves."""
        features = {}
        voltage_data = df['voltage'].values
        
        # Basic voltage statistics
        features.update({
            'voltage_mean': np.mean(voltage_data),
            'voltage_std': np.std(voltage_data),
            'voltage_range': np.ptp(voltage_data)
        })
        
        # Voltage curve characteristics
        for threshold in self.voltage_thresholds:
            time_above = np.sum(voltage_data > threshold)
            features[f'time_above_{threshold:.2f}V'] = time_above
        
        # Find voltage peaks and valleys
        peaks, _ = find_peaks(voltage_data)
        valleys, _ = find_peaks(-voltage_data)
        
        features.update({
            'voltage_peak_count': len(peaks),
            'voltage_valley_count': len(valleys),
            'voltage_peak_mean': np.mean(voltage_data[peaks]) if len(peaks) > 0 else 0,
            'voltage_valley_mean': np.mean(voltage_data[valleys]) if len(valleys) > 0 else 0
        })
        
        return features
    
    def _extract_temperature_features(self, df: pd.DataFrame) -> Dict:
        """Extract features based on temperature data."""
        temp_data = df['temperature'].values
        
        features = {
            'temp_mean': np.mean(temp_data),
            'temp_std': np.std(temp_data),
            'temp_max': np.max(temp_data),
            'temp_min': np.min(temp_data),
            'temp_range': np.ptp(temp_data),
            'temp_rms': np.sqrt(np.mean(np.square(temp_data))),
            'temp_above_40': np.sum(temp_data > 40) / len(temp_data),
            'temp_below_10': np.sum(temp_data < 10) / len(temp_data)
        }
        
        # Temperature gradients
        temp_gradient = np.gradient(temp_data)
        features.update({
            'temp_gradient_mean': np.mean(np.abs(temp_gradient)),
            'temp_gradient_max': np.max(np.abs(temp_gradient)),
            'temp_gradient_std': np.std(temp_gradient)
        })
        
        return features
    
    def _extract_differential_features(self, df: pd.DataFrame) -> Dict:
        """Extract differential capacity and voltage features."""
        features = {}
        
        # Calculate dQ/dV (differential capacity)
        voltage_diff = np.diff(df['voltage'].values)
        capacity_diff = np.diff(df['capacity'].values)
        
        # Avoid division by zero
        valid_idx = voltage_diff != 0
        dq_dv = np.zeros_like(voltage_diff)
        dq_dv[valid_idx] = capacity_diff[valid_idx] / voltage_diff[valid_idx]
        
        features.update({
            'dq_dv_mean': np.mean(dq_dv),
            'dq_dv_std': np.std(dq_dv),
            'dq_dv_max': np.max(dq_dv),
            'dq_dv_min': np.min(dq_dv)
        })
        
        # Find peaks in dQ/dV
        peaks, properties = find_peaks(dq_dv, height=0)
        if len(peaks) > 0:
            features.update({
                'dq_dv_peak_count': len(peaks),
                'dq_dv_peak_height_mean': np.mean(properties['peak_heights']),
                'dq_dv_peak_height_std': np.std(properties['peak_heights'])
            })
        
        return features
    
    def _calculate_combined_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate combined metrics from multiple parameters."""
        features = {}
        
        # Energy metrics
        features['energy_efficiency'] = self._calculate_energy_efficiency(df)
        
        # Resistance-related metrics
        if 'resistance' in df.columns:
            resistance_data = df['resistance'].values
            features.update({
                'resistance_mean': np.mean(resistance_data),
                'resistance_increase_rate': (resistance_data[-1] - resistance_data[0]) / len(resistance_data),
                'resistance_std': np.std(resistance_data)
            })
        
        # State of Health indicators
        features['soh_capacity'] = df['capacity'].values[-1] / df['capacity'].values[0]
        
        return features
    
    @staticmethod
    def _calculate_energy_efficiency(df: pd.DataFrame) -> float:
        """Calculate energy efficiency from charge-discharge data."""
        charge_mask = df['current'] > 0
        discharge_mask = df['current'] < 0
        
        charge_energy = np.sum(df.loc[charge_mask, 'voltage'] * df.loc[charge_mask, 'current'])
        discharge_energy = np.sum(df.loc[discharge_mask, 'voltage'] * df.loc[discharge_mask, 'current'])
        
        return abs(discharge_energy / charge_energy) if charge_energy != 0 else 0.0
    
    def get_feature_importance(self, features: pd.DataFrame, target: str) -> Dict[str, float]:
        """
        Calculate feature importance scores.
        
        Args:
            features: DataFrame of extracted features
            target: Name of the target variable
        Returns:
            Dictionary of feature importance scores
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare data
        X = features.drop(columns=[target])
        y = features[target]
        
        # Train a random forest model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance scores
        importance_scores = dict(zip(X.columns, rf.feature_importances_))
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
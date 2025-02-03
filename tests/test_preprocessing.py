import unittest
import pandas as pd
import numpy as np
from src.data_processing.preprocess import BatteryDataPreprocessor

class TestBatteryDataPreprocessor(unittest.TestCase):
    """Test cases for BatteryDataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = BatteryDataPreprocessor(voltage_range=(2.5, 4.2))
        
        # Create sample test data
        cycles = np.repeat(range(1, 11), 10)
        voltage = np.random.uniform(2.5, 4.2, 100)
        current = np.random.uniform(-2, 2, 100)
        temperature = np.random.uniform(20, 40, 100)
        time = np.arange(100)
        
        self.test_df = pd.DataFrame({
            'cycle': cycles,
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'time': time,
            'capacity': np.random.uniform(1, 2, 100)
        })
        
        # Add some invalid data
        self.test_df.loc[0, 'voltage'] = 5.0  # Invalid voltage
        self.test_df.loc[1, 'capacity'] = 0.05  # Below threshold
        self.test_df.loc[2, 'temperature'] = 70  # Invalid temperature
        
    def test_remove_invalid_data(self):
        """Test removal of invalid data points."""
        processed_df = self.preprocessor._remove_invalid_data(self.test_df)
        
        # Check if invalid points were removed
        self.assertTrue(processed_df['voltage'].between(2.5, 4.2).all())
        self.assertTrue(processed_df['capacity'].gt(0.1).all())
        self.assertTrue(processed_df['temperature'].between(-20, 60).all())
        
    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Add some missing values
        test_df = self.test_df.copy()
        test_df.loc[0, 'temperature'] = np.nan
        test_df.loc[1, 'current'] = np.nan
        
        processed_df = self.preprocessor._handle_missing_values(test_df)
        
        # Check if missing values were handled
        self.assertFalse(processed_df['temperature'].isna().any())
        self.assertTrue(len(processed_df) < len(test_df))  # Row with missing current should be removed
        
    def test_calculate_capacity(self):
        """Test capacity calculation."""
        processed_df = self.preprocessor._calculate_capacity(self.test_df)
        
        # Check if capacity was calculated for each cycle
        self.assertEqual(len(processed_df['capacity'].unique()), 
                        len(self.test_df['cycle'].unique()))
        self.assertTrue((processed_df['capacity'] >= 0).all())
        
    def test_calculate_resistance(self):
        """Test resistance calculation."""
        processed_df = self.preprocessor._calculate_resistance(self.test_df)
        
        # Check resistance calculations
        self.assertIn('resistance', processed_df.columns)
        self.assertFalse(processed_df['resistance'].isna().any())
        self.assertTrue((processed_df['resistance'] >= 0).all())
        
    def test_normalize_features(self):
        """Test feature normalization."""
        processed_df = self.preprocessor._normalize_features(self.test_df)
        
        # Check if normalized features are in [0,1] range
        normalized_columns = [col for col in processed_df.columns if 'normalized' in col]
        for col in normalized_columns:
            self.assertTrue(processed_df[col].between(0, 1).all())
            
    def test_full_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        processed_df = self.preprocessor.process_raw_data(self.test_df)
        
        # Check if all expected columns exist
        expected_columns = [
            'cycle', 'voltage', 'current', 'temperature', 'capacity',
            'resistance', 'voltage_normalized', 'current_normalized',
            'temperature_normalized', 'capacity_normalized'
        ]
        for col in expected_columns:
            self.assertIn(col, processed_df.columns)
            
        # Check data quality
        self.assertFalse(processed_df.isna().any().any())
        self.assertTrue(processed_df['voltage'].between(2.5, 4.2).all())
        self.assertTrue(processed_df['capacity'].gt(0.1).all())
        
    def test_detect_outliers(self):
        """Test outlier detection."""
        columns_to_check = ['voltage', 'current', 'temperature']
        processed_df = self.preprocessor.detect_outliers(self.test_df, columns_to_check)
        
        # Check if outlier columns were created
        for col in columns_to_check:
            outlier_col = f'{col}_outlier'
            self.assertIn(outlier_col, processed_df.columns)
            self.assertTrue(processed_df[outlier_col].dtype == bool)
            
    def test_extract_features(self):
        """Test feature extraction."""
        features = self.preprocessor.extract_features(self.test_df)
        
        # Check if all expected features exist
        expected_features = [
            'mean_capacity', 'capacity_fade_rate', 'mean_temperature',
            'temperature_variance', 'mean_resistance', 'resistance_increase',
            'cycle_count'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
            
        # Check feature values
        self.assertTrue(features['cycle_count'] == len(self.test_df['cycle'].unique()))
        self.assertTrue(features['mean_capacity'] > 0)
        
if __name__ == '__main__':
    unittest.main()
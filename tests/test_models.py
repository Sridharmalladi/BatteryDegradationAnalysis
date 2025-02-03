import unittest
import numpy as np
from src.models.capacity_fade import CapacityFadeModel
from src.models.temperature import TemperatureModel

class TestCapacityFadeModel(unittest.TestCase):
    """Test cases for CapacityFadeModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = CapacityFadeModel(temperature_coefficient=0.06)
        
        # Create sample test data
        self.cycles = np.linspace(0, 1000, 100)
        self.temperature = 25 + 5 * np.sin(self.cycles / 100)  # Simulated temperature variation
        self.capacity = 1 - 0.2 * (1 - np.exp(-self.cycles/500)) - 0.0001 * self.cycles
        self.capacity += np.random.normal(0, 0.01, len(self.cycles))  # Add noise
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.temp_coeff, 0.06)
        self.assertIsNone(self.model.fitted_params)
        
    def test_capacity_fade_equation(self):
        """Test the capacity fade equation."""
        cycles = np.array([0, 100, 200])
        result = self.model.capacity_fade_equation(cycles, A=0.2, k=0.001, b=0.0001)
        
        # Check basic properties
        self.assertEqual(len(result), len(cycles))
        self.assertTrue(np.all(result <= 1))  # Capacity should not exceed initial
        self.assertTrue(np.all(result >= 0))  # Capacity should not be negative
        
    def test_model_fitting(self):
        """Test model fitting procedure."""
        fitted_params = self.model.fit(self.cycles, self.capacity, self.temperature)
        
        # Check if parameters were fitted
        self.assertIsNotNone(self.model.fitted_params)
        self.assertIn('A', fitted_params)
        self.assertIn('k', fitted_params)
        self.assertIn('b', fitted_params)
        
        # Check parameter ranges
        self.assertTrue(0 <= fitted_params['A'] <= 1)
        self.assertTrue(fitted_params['k'] > 0)
        self.assertTrue(fitted_params['b'] > 0)
        
    def test_prediction(self):
        """Test model predictions."""
        # Fit the model first
        self.model.fit(self.cycles, self.capacity, self.temperature)
        
        # Make predictions
        predictions = self.model.predict(self.cycles, self.temperature)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.cycles))
        self.assertTrue(np.all(predictions <= 1))
        self.assertTrue(np.all(predictions >= 0))
        
        # Check prediction accuracy
        mse = np.mean((predictions - self.capacity) ** 2)
        self.assertTrue(mse < 0.1)  # Reasonable fit threshold
        
    def test_remaining_life_estimation(self):
        """Test remaining useful life estimation."""
        # Fit the model
        self.model.fit(self.cycles, self.capacity, self.temperature)
        
        # Estimate remaining life
        remaining_life = self.model.estimate_remaining_life(
            current_capacity=0.9,
            temperature=25.0,
            capacity_threshold=0.8
        )
        
        # Check result
        self.assertIsInstance(remaining_life, (int, float))
        self.assertTrue(remaining_life > 0)
        
class TestTemperatureModel(unittest.TestCase):
    """Test cases for TemperatureModel class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.model = TemperatureModel(activation_energy=0.5)
        
        # Create sample temperature data
        self.time_data = np.linspace(0, 1000, 100)
        self.temp_data = 25 + 10 * np.sin(self.time_data / 100)
        
    def test_acceleration_factor(self):
        """Test temperature acceleration factor calculation."""
        factor_25c = self.model.calculate_acceleration_factor(25.0)
        factor_35c = self.model.calculate_acceleration_factor(35.0)
        
        # Check basic properties
        self.assertAlmostEqual(factor_25c, 1.0, places=2)  # Reference temperature
        self.assertTrue(factor_35c > factor_25c)  # Higher temperature = higher factor
        
    def test_temperature_analysis(self):
        """Test temperature history analysis."""
        stats = self.model.analyze_temperature_history(self.temp_data, self.time_data)
        
        # Check if all expected metrics are present
        expected_metrics = ['mean_temp', 'max_temp', 'min_temp', 'std_temp',
                          'cumulative_aging_factor']
        for metric in expected_metrics:
            self.assertIn(metric, stats)
            
        # Check metric values
        self.assertTrue(stats['min_temp'] <= stats['mean_temp'] <= stats['max_temp'])
        self.assertTrue(stats['cumulative_aging_factor'] > 0)
        
    def test_calendar_aging(self):
        """Test calendar aging calculation."""
        capacity_loss = self.model.calculate_calendar_aging(
            storage_temp=35.0,
            time_days=100.0
        )
        
        # Check result
        self.assertTrue(0 <= capacity_loss <= 1)
        
        # Check temperature dependence
        loss_25c = self.model.calculate_calendar_aging(25.0, 100.0)
        loss_35c = self.model.calculate_calendar_aging(35.0, 100.0)
        self.assertTrue(loss_35c > loss_25c)
        
    def test_thermal_stress_analysis(self):
        """Test thermal stress analysis."""
        stress_metrics = self.model.thermal_stress_analysis(
            self.temp_data,
            self.time_data
        )
        
        # Check metrics
        self.assertIn('max_gradient', stress_metrics)
        self.assertIn('mean_gradient', stress_metrics)
        self.assertIn('thermal_cycles', stress_metrics)
        self.assertIn('time_above_40C', stress_metrics)
        
        # Check metric values
        self.assertTrue(stress_metrics['max_gradient'] >= stress_metrics['mean_gradient'])
        self.assertTrue(stress_metrics['thermal_cycles'] >= 0)
        self.assertTrue(stress_metrics['time_above_40C'] >= 0)
        
if __name__ == '__main__':
    unittest.main()
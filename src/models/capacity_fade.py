import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict
import pandas as pd

class CapacityFadeModel:
    """
    A comprehensive model for predicting battery capacity degradation over time.
    Incorporates both cycling and calendar aging effects.
    """
    
    def __init__(self, temperature_coefficient: float = 0.06):
        self.temp_coeff = temperature_coefficient
        self.fitted_params = None
        self.model_metrics = {}
        
    def capacity_fade_equation(self, cycles: np.ndarray, 
                             A: float, k: float, b: float) -> np.ndarray:
        """
        Core capacity fade equation combining both linear and non-linear degradation.
        
        Args:
            cycles: Number of cycles
            A: Pre-exponential factor
            k: Degradation rate constant
            b: Linear degradation coefficient
        """
        return (1 - A * np.exp(-k * cycles) - b * cycles)
    
    def fit(self, cycles: np.ndarray, capacity: np.ndarray, 
            temperature: np.ndarray) -> Dict:
        """
        Fit the capacity fade model to experimental data.
        
        Args:
            cycles: Array of cycle numbers
            capacity: Array of measured capacities (normalized)
            temperature: Array of temperatures during cycling
        """
        # Temperature correction
        temp_factor = np.exp(self.temp_coeff * (temperature - 25))
        adjusted_cycles = cycles * temp_factor
        
        # Fit the model
        popt, pcov = curve_fit(self.capacity_fade_equation, adjusted_cycles, 
                             capacity, bounds=([0, 0, 0], [1, 0.1, 0.001]))
        
        self.fitted_params = {
            'A': popt[0],
            'k': popt[1],
            'b': popt[2]
        }
        
        # Calculate model metrics
        predictions = self.predict(cycles, temperature)
        self.model_metrics = self._calculate_metrics(capacity, predictions)
        
        return self.fitted_params
    
    def predict(self, cycles: np.ndarray, temperature: np.ndarray) -> np.ndarray:
        """
        Predict capacity fade for given cycling conditions.
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before making predictions")
            
        temp_factor = np.exp(self.temp_coeff * (temperature - 25))
        adjusted_cycles = cycles * temp_factor
        
        return self.capacity_fade_equation(
            adjusted_cycles,
            self.fitted_params['A'],
            self.fitted_params['k'],
            self.fitted_params['b']
        )
    
    def estimate_remaining_life(self, current_capacity: float,
                              temperature: float,
                              capacity_threshold: float = 0.8) -> float:
        """
        Estimate remaining useful life until capacity reaches threshold.
        """
        if self.fitted_params is None:
            raise ValueError("Model must be fitted before estimating life")
            
        def capacity_at_cycle(n):
            return self.predict(np.array([n]), np.array([temperature]))[0]
        
        # Binary search for cycle number where capacity reaches threshold
        cycle_min, cycle_max = 0, 10000
        while cycle_max - cycle_min > 1:
            cycle_mid = (cycle_min + cycle_max) // 2
            cap_mid = capacity_at_cycle(cycle_mid)
            
            if cap_mid > capacity_threshold:
                cycle_min = cycle_mid
            else:
                cycle_max = cycle_mid
                
        return cycle_min
    
    def _calculate_metrics(self, actual: np.ndarray, 
                         predicted: np.ndarray) -> Dict:
        """
        Calculate model performance metrics.
        """
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        r2 = 1 - (np.sum((actual - predicted) ** 2) / 
                 np.sum((actual - np.mean(actual)) ** 2))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

# Additional helper functions
def calculate_cycle_stress_factor(depth_of_discharge: float,
                                charge_rate: float,
                                discharge_rate: float) -> float:
    """
    Calculate stress factor based on cycling conditions.
    """
    dod_factor = 1 + 0.5 * depth_of_discharge
    rate_factor = 1 + 0.2 * (charge_rate + discharge_rate)
    return dod_factor * rate_factor

def analyze_degradation_rate(capacity_history: List[float],
                           cycle_numbers: List[int]) -> Dict:
    """
    Analyze the rate of degradation over time.
    """
    capacity_array = np.array(capacity_history)
    cycle_array = np.array(cycle_numbers)
    
    # Calculate degradation rates
    rates = np.diff(capacity_array) / np.diff(cycle_array)
    
    return {
        'mean_rate': np.mean(rates),
        'std_rate': np.std(rates),
        'max_rate': np.min(rates),  # Most negative rate
        'acceleration': np.mean(np.diff(rates))
    }
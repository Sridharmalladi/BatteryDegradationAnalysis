import numpy as np
from scipy.integrate import trapz
from typing import Dict, List, Tuple
import pandas as pd

class TemperatureModel:
    """
    Models temperature-dependent aging effects in lithium-ion batteries.
    Implements Arrhenius-based degradation calculations.
    """
    
    def __init__(self, activation_energy: float = 0.5,
                 gas_constant: float = 8.314):
        self.Ea = activation_energy  # Activation energy in eV
        self.R = gas_constant / 1000  # Gas constant in kJ/mol-K
        self.reference_temp = 298.15  # Reference temperature (25°C) in Kelvin
        
    def calculate_acceleration_factor(self, temperature: float) -> float:
        """
        Calculate temperature-based acceleration factor using Arrhenius equation.
        
        Args:
            temperature: Temperature in Celsius
        Returns:
            Acceleration factor relative to reference temperature
        """
        temp_K = temperature + 273.15
        return np.exp((self.Ea / self.R) * (1/self.reference_temp - 1/temp_K))
    
    def analyze_temperature_history(self, 
                                  temperature_data: np.ndarray,
                                  time_data: np.ndarray) -> Dict:
        """
        Analyze temperature exposure history to calculate cumulative aging effects.
        
        Args:
            temperature_data: Array of temperature measurements (°C)
            time_data: Array of corresponding timestamps
        """
        temp_factors = np.array([self.calculate_acceleration_factor(t) 
                               for t in temperature_data])
        
        # Calculate time-weighted aging
        aging_integral = trapz(temp_factors, time_data)
        
        # Analyze temperature statistics
        stats = {
            'mean_temp': np.mean(temperature_data),
            'max_temp': np.max(temperature_data),
            'min_temp': np.min(temperature_data),
            'std_temp': np.std(temperature_data),
            'cumulative_aging_factor': aging_integral
        }
        
        return stats
    
    def calculate_calendar_aging(self, 
                               storage_temp: float,
                               time_days: float) -> float:
        """
        Calculate calendar aging based on storage conditions.
        
        Args:
            storage_temp: Storage temperature in Celsius
            time_days: Storage time in days
        Returns:
            Estimated capacity loss fraction
        """
        accel_factor = self.calculate_acceleration_factor(storage_temp)
        
        # Calendar aging model parameters
        alpha = 0.0575  # Capacity loss rate coefficient
        beta = 0.5      # Time dependence factor
        
        capacity_loss = alpha * accel_factor * (time_days ** beta)
        return capacity_loss
    
    def thermal_stress_analysis(self, 
                              temp_profile: np.ndarray,
                              time_steps: np.ndarray) -> Dict:
        """
        Analyze thermal stress based on temperature variations.
        
        Args:
            temp_profile: Temperature measurements over time
            time_steps: Corresponding time points
        """
        temp_gradient = np.gradient(temp_profile, time_steps)
        
        # Calculate thermal stress metrics
        stress_metrics = {
            'max_gradient': np.max(np.abs(temp_gradient)),
            'mean_gradient': np.mean(np.abs(temp_gradient)),
            'thermal_cycles': len(self._count_thermal_cycles(temp_profile)),
            'time_above_40C': np.sum(temp_profile > 40) * np.mean(np.diff(time_steps))
        }
        
        return stress_metrics
    
    def _count_thermal_cycles(self, 
                            temperature_data: np.ndarray,
                            threshold: float = 5.0) -> List[Tuple]:
        """
        Count significant thermal cycles in temperature data.
        
        Args:
            temperature_data: Array of temperature measurements
            threshold: Minimum temperature difference to count as a cycle
        """
        cycles = []
        i = 0
        while i < len(temperature_data) - 1:
            # Find rising edge
            while i < len(temperature_data) - 1 and \
                  temperature_data[i+1] <= temperature_data[i]:
                i += 1
            if i >= len(temperature_data) - 1:
                break
                
            start_temp = temperature_data[i]
            
            # Find peak
            while i < len(temperature_data) - 1 and \
                  temperature_data[i+1] >= temperature_data[i]:
                i += 1
            if i >= len(temperature_data) - 1:
                break
                
            peak_temp = temperature_data[i]
            
            # Check if cycle amplitude exceeds threshold
            if peak_temp - start_temp >= threshold:
                cycles.append((start_temp, peak_temp))
                
            i += 1
            
        return cycles

def calculate_temperature_distribution(temp_history: np.ndarray,
                                    bin_width: float = 5.0) -> Dict:
    """
    Calculate temperature exposure distribution.
    
    Args:
        temp_history: Array of temperature measurements
        bin_width: Width of temperature bins in degrees
    """
    bins = np.arange(np.floor(np.min(temp_history)),
                    np.ceil(np.max(temp_history)) + bin_width,
                    bin_width)
    
    hist, edges = np.histogram(temp_history, bins=bins, density=True)
    
    distribution = {
        'temperature_bins': edges[:-1],
        'exposure_fraction': hist,
        'mean_temperature': np.mean(temp_history),
        'temperature_range': np.ptp(temp_history)
    }
    
    return distribution
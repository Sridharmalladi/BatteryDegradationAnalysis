import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

class BatteryVisualizer:
    """
    Class for creating standardized visualizations of battery degradation data.
    """
    
    def __init__(self, style: str = 'seaborn-whitegrid'):
        plt.style.use(style)
        self.default_figsize = (12, 8)
        self.color_palette = sns.color_palette("husl", 8)
        
    def plot_capacity_fade(self, cycles: np.ndarray, capacity: np.ndarray,
                          predictions: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot capacity fade over cycles with optional model predictions.
        
        Args:
            cycles: Array of cycle numbers
            capacity: Array of measured capacities
            predictions: Optional array of model predictions
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.default_figsize)
        
        # Plot measured data
        plt.scatter(cycles, capacity, c=self.color_palette[0], 
                   label='Measured', alpha=0.6)
        
        # Plot predictions if provided
        if predictions is not None:
            plt.plot(cycles, predictions, c=self.color_palette[1],
                    label='Model prediction', linewidth=2)
            
        plt.xlabel('Cycle Number')
        plt.ylabel('Capacity (Ah)')
        plt.title('Battery Capacity Fade Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_temperature_effects(self, df: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
        """
        Create a multi-panel plot showing temperature effects.
        
        Args:
            df: DataFrame containing temperature and degradation data
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature distribution
        sns.histplot(data=df, x='temperature', ax=axes[0,0], bins=30)
        axes[0,0].set_title('Temperature Distribution')
        
        # Temperature vs Capacity
        sns.scatterplot(data=df, x='temperature', y='capacity',
                       ax=axes[0,1], alpha=0.5)
        axes[0,1].set_title('Temperature vs Capacity')
        
        # Temperature over time
        sns.lineplot(data=df, x='cycle', y='temperature',
                    ax=axes[1,0])
        axes[1,0].set_title('Temperature Profile')
        
        # Temperature heatmap
        pivot_temp = df.pivot_table(index='cycle', columns='voltage',
                                  values='temperature', aggfunc='mean')
        sns.heatmap(pivot_temp, ax=axes[1,1], cmap='viridis')
        axes[1,1].set_title('Temperature Heatmap')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_degradation_summary(self, df: pd.DataFrame,
                                 metrics: dict,
                                 save_dir: Optional[str] = None) -> None:
        """
        Create a comprehensive degradation summary plot.
        
        Args:
            df: Processed battery data
            metrics: Dictionary of calculated metrics
            save_dir: Optional directory to save plots
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3)
        
        # Capacity fade
        ax1 = fig.add_subplot(gs[0, :])
        sns.lineplot(data=df, x='cycle', y='capacity', ax=ax1)
        ax1.set_title('Capacity Fade')
        
        # Resistance evolution
        ax2 = fig.add_subplot(gs[1, :2])
        sns.lineplot(data=df, x='cycle', y='resistance', ax=ax2)
        ax2.set_title('Resistance Evolution')
        
        # Temperature distribution
        ax3 = fig.add_subplot(gs[1, 2])
        sns.boxplot(data=df, y='temperature', ax=ax3)
        ax3.set_title('Temperature Distribution')
        
        # Metrics summary
        ax4 = fig.add_subplot(gs[2, :])
        metrics_df = pd.DataFrame(list(metrics.items()),
                                columns=['Metric', 'Value'])
        table = ax4.table(cellText=metrics_df.values,
                         colLabels=metrics_df.columns,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.axis('off')
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(Path(save_dir) / 'degradation_summary.png',
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_voltage_curves(self, df: pd.DataFrame,
                          cycles: List[int],
                          save_path: Optional[str] = None) -> None:
        """
        Plot voltage curves for specified cycles.
        
        Args:
            df: DataFrame containing voltage data
            cycles: List of cycles to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=self.default_figsize)
        
        for cycle in cycles:
            cycle_data = df[df['cycle'] == cycle]
            plt.plot(cycle_data['capacity'], cycle_data['voltage'],
                    label=f'Cycle {cycle}')
            
        plt.xlabel('Capacity (Ah)')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage Curves at Different Cycles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def save_all_figures(directory: str) -> None:
        """
        Save all currently open figures to specified directory.
        
        Args:
            directory: Directory path to save figures
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for i, fig in enumerate(plt.get_fignums()):
            plt.figure(fig)
            plt.savefig(directory / f'figure_{i}.png',
                       dpi=300, bbox_inches='tight')
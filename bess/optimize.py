# bess/optimize.py
"""
High-level optimization interface for BESS dispatch.
Provides simple API for common optimization scenarios.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pandas as pd
import numpy as np

# Fixed imports - use relative imports
from .schema import DataBundle
from .milp_interface import MILPDataInterface
from .milp_dispatcher import MILPDispatcher, OptimizationResults
from .io import DataLoader, DataWriter

logger = logging.getLogger(__name__)


class BESSOptimizer:
    """
    High-level optimizer providing simple interface to BESS optimization.
    
    Example:
        optimizer = BESSOptimizer.from_config('config.yaml')
        results = optimizer.optimize()
        optimizer.save_results('results.csv')
    """
    
    def __init__(self, data_bundle: DataBundle):
        """
        Initialize optimizer with data bundle.
        
        Args:
            data_bundle: Validated DataBundle with all parameters
        """
        self.bundle = data_bundle
        self.results = None
        self.dispatcher = None
        
        # Validate data
        self._validate()
        
        # Create MILP interface
        self.interface = MILPDataInterface(data_bundle)
        
    @classmethod
    def from_config(cls, config_path: Union[str, Path], 
                   timeseries_path: Optional[Union[str, Path]] = None) -> 'BESSOptimizer':
        """
        Create optimizer from configuration files.
        
        Args:
            config_path: Path to configuration YAML/JSON
            timeseries_path: Optional path to time series CSV
            
        Returns:
            Configured BESSOptimizer instance
        """
        bundle = DataLoader.load_bundle(config_path, timeseries_path)
        return cls(bundle)
    
    def _validate(self):
        """Validate data bundle for optimization."""
        try:
            self.bundle.validate_lengths()
            self.bundle.validate_architecture_requirements()
            logger.info("Data validation successful")
        except Exception as e:
            raise ValueError(f"Data validation failed: {e}")
    
    def optimize(self, 
                solver: str = 'gurobi',
                time_limit: Optional[float] = None,
                mip_gap: float = 0.001,
                verbose: bool = False) -> OptimizationResults:
        """
        Run optimization.
        
        Args:
            solver: Solver backend ('gurobi' or 'cplex')
            time_limit: Maximum solve time in seconds
            mip_gap: MIP optimality gap
            verbose: Whether to show solver output
            
        Returns:
            OptimizationResults with solution
        """
        # Get MILP parameters
        params = self.interface.to_milp_params()
        
        # Log scenario summary
        summary = self.interface.get_scenario_summary()
        logger.info(f"Optimizing {summary['scenario_id']}")
        logger.info(f"  Architecture: {summary['architecture']}")
        logger.info(f"  Battery: {summary['battery_capacity_kwh']} kWh, {summary['battery_power_kw']} kW")
        logger.info(f"  Services: {summary['num_services']} ancillary services")
        
        # Create and configure dispatcher
        self.dispatcher = MILPDispatcher(params, solver)
        
        if verbose:
            self.dispatcher.model.setParam('OutputFlag', 1)
        
        # Solve
        logger.info("Starting optimization...")
        self.results = self.dispatcher.solve(time_limit, mip_gap)
        
        # Log results summary
        self._log_results_summary()
        
        return self.results
    
    def _log_results_summary(self):
        """Log summary of optimization results."""
        if not self.results:
            return
        
        r = self.results
        logger.info(f"Optimization complete: {r.status}")
        logger.info(f"  Objective value: €{r.objective_value:,.2f}")
        logger.info(f"  Total revenue: €{r.total_revenue:,.2f}")
        logger.info(f"  Energy cost: €{r.energy_cost:,.2f}")
        logger.info(f"  Ancillary revenue: €{r.ancillary_revenue:,.2f}")
        logger.info(f"  Capacity revenue: €{r.capacity_revenue:,.2f}")
        logger.info(f"  Degradation cost: €{r.degradation_cost:,.2f}")
        logger.info(f"  Solve time: {r.solve_time:.2f}s")
    
    def save_results(self, output_path: Union[str, Path], 
                    format: str = 'csv', 
                    include_metadata: bool = True):
        """
        Save optimization results to file.
        
        Args:
            output_path: Where to save results
            format: Output format ('csv', 'parquet', 'json')
            include_metadata: Whether to include metadata
        """
        if not self.results:
            raise ValueError("No results to save. Run optimize() first.")
        
        DataWriter.save_results(self.results, output_path, format, include_metadata)
    
    def get_kpis(self) -> Dict[str, float]:
        """
        Calculate key performance indicators.
        
        Returns:
            Dictionary of KPIs
        """
        if not self.results:
            raise ValueError("No results available. Run optimize() first.")
        
        r = self.results
        b = self.bundle
        
        # Calculate average daily cycles
        total_discharge = np.sum(r.discharge_load + r.discharge_wholesale + r.discharge_freq)
        days = len(b.timegrid.index()) * b.dt_hours / 24
        avg_cycles = total_discharge / (b.capacity.capacity_nominal_kwh * 
                                       (b.bounds.soc_max - b.bounds.soc_min) * days)
        
        # Calculate capacity factor
        capacity_factor = r.capacity_commitment / b.ratings.p_discharge_max_kw if r.capacity_commitment > 0 else 0
        
        # Calculate utilization
        max_possible_energy = b.capacity.capacity_nominal_kwh * (b.bounds.soc_max - b.bounds.soc_min) * days * 2  # 2 cycles/day max
        utilization = total_discharge / max_possible_energy
        
        return {
            'total_revenue': r.total_revenue,
            'energy_cost': r.energy_cost,
            'ancillary_revenue': r.ancillary_revenue,
            'capacity_revenue': r.capacity_revenue,
            'degradation_cost': r.degradation_cost,
            'avg_daily_cycles': avg_cycles,
            'capacity_factor': capacity_factor,
            'utilization': utilization,
            'total_pv_curtailed': np.sum(r.pv_curtailed),
            'total_energy_discharged': total_discharge,
            'roundtrip_losses': np.sum(r.charge_retail + r.charge_wholesale + r.charge_pv) - total_discharge,
        }
    
    def plot_dispatch(self, save_path: Optional[Union[str, Path]] = None, 
                     show: bool = True):
        """
        Create dispatch visualization.
        
        Args:
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure if created, None if matplotlib not available
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot create plots.")
            return None
        
        if not self.results:
            raise ValueError("No results to plot. Run optimize() first.")
        
        r = self.results
        time_index = self.bundle.timegrid.index()
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Power flows
        ax1 = axes[0]
        ax1.plot(time_index, r.charge_retail + r.charge_wholesale, 
                label='Grid Charge', color='red', alpha=0.7)
        ax1.plot(time_index, -r.discharge_wholesale, 
                label='Grid Discharge', color='green', alpha=0.7)
        ax1.plot(time_index, r.charge_pv, 
                label='PV Charge', color='orange', alpha=0.7)
        ax1.plot(time_index, -r.discharge_load, 
                label='Load Discharge', color='blue', alpha=0.7)
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('Battery Power Flows')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: State of Charge
        ax2 = axes[1]
        ax2.plot(time_index, r.soc_total, color='purple', linewidth=2)
        ax2.fill_between(time_index, 0, r.soc_total, alpha=0.3, color='purple')
        ax2.set_ylabel('SoC (kWh)')
        ax2.set_title('State of Charge')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Prices and reserves
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        # Prices on left axis
        ax3.plot(time_index, self.bundle.tariffs.price_wholesale, 
                label='Wholesale Price', color='blue', alpha=0.7)
        ax3.plot(time_index, self.bundle.tariffs.price_retail, 
                label='Retail Price', color='red', alpha=0.7)
        ax3.set_ylabel('Price (€/kWh)')
        ax3.set_xlabel('Time')
        ax3.legend(loc='upper left')
        
        # Total reserves on right axis
        total_reserves = np.zeros(len(time_index))
        for k, reserves in r.reserve_capacity.items():
            total_reserves += reserves
        ax3_twin.plot(time_index, total_reserves, 
                     label='Total Reserves', color='green', linewidth=2)
        ax3_twin.set_ylabel('Reserve Capacity (kW)')
        ax3_twin.legend(loc='upper right')
        
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


def quick_optimize(config_path: Union[str, Path], 
                  output_path: Optional[Union[str, Path]] = None,
                  **kwargs) -> OptimizationResults:
    """
    Quick one-line optimization from config file.
    
    Args:
        config_path: Path to configuration file
        output_path: Optional path to save results
        **kwargs: Additional arguments for optimize()
        
    Returns:
        OptimizationResults
        
    Example:
        results = quick_optimize('config.yaml', 'results.csv')
    """
    optimizer = BESSOptimizer.from_config(config_path)
    results = optimizer.optimize(**kwargs)
    
    if output_path:
        optimizer.save_results(output_path)
    
    # Print KPIs
    kpis = optimizer.get_kpis()
    print("\n=== Optimization Results ===")
    for key, value in kpis.items():
        if 'revenue' in key or 'cost' in key:
            print(f"{key:25s}: €{value:,.2f}")
        else:
            print(f"{key:25s}: {value:.3f}")
    
    return results
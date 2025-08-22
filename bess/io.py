# bess/io.py
"""
I/O utilities for loading and saving BESS optimization data.
Supports YAML, JSON, and CSV formats with validation.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime, timedelta

# Fixed imports - use relative imports within package
from .schema import (
    DataBundle, TimeGrid, Architecture, Ratings, Efficiencies,
    Bounds, Degradation, Tariffs, Exogenous, Capacity, AncillaryService
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Flexible data loader supporting multiple formats and sources."""
    
    @staticmethod
    def load_bundle(config_path: Union[str, Path], 
                   timeseries_path: Optional[Union[str, Path]] = None,
                   validate: bool = True) -> DataBundle:
        """
        Load DataBundle from configuration and time series files.
        
        Args:
            config_path: Path to configuration file (YAML/JSON)
            timeseries_path: Optional path to time series data (CSV/Parquet)
            validate: Whether to validate the loaded data
            
        Returns:
            Validated DataBundle ready for optimization
        """
        config_path = Path(config_path)
        
        # Load configuration
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Load time series if provided separately
        if timeseries_path:
            timeseries = DataLoader._load_timeseries(timeseries_path)
            config = DataLoader._merge_timeseries(config, timeseries)
        
        # Create DataBundle
        bundle = DataLoader._config_to_bundle(config)
        
        # Validate if requested
        if validate:
            bundle.validate_lengths()
            bundle.validate_architecture_requirements()
            logger.info(f"Data validation successful for {config_path.name}")
        
        return bundle
    
    @staticmethod
    def _load_timeseries(path: Union[str, Path]) -> pd.DataFrame:
        """Load time series data from CSV or Parquet."""
        path = Path(path)
        
        if path.suffix == '.csv':
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.suffix in ['.parquet', '.pq']:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported time series format: {path.suffix}")
        
        logger.info(f"Loaded {len(df)} timesteps from {path.name}")
        return df
    
    @staticmethod
    def _merge_timeseries(config: Dict, timeseries: pd.DataFrame) -> Dict:
        """Merge time series data into configuration."""
        # Map DataFrame columns to config paths
        column_mapping = {
            'load_kw': 'exogenous.load_ac',
            'pv_kw': 'exogenous.pv_dc',
            'price_retail': 'tariffs.price_retail',
            'price_wholesale': 'tariffs.price_wholesale',
            'price_capacity': 'tariffs.price_cap',
            'capacity_mask': 'tariffs.cap_mask',
        }
        
        for col, path in column_mapping.items():
            if col in timeseries.columns:
                DataLoader._set_nested(config, path, timeseries[col].values)
        
        # Handle ancillary service prices
        for col in timeseries.columns:
            if col.startswith('price_as_'):
                service_name = col.replace('price_as_', '')
                if 'tariffs' not in config:
                    config['tariffs'] = {}
                if 'price_ancillary' not in config['tariffs']:
                    config['tariffs']['price_ancillary'] = {}
                config['tariffs']['price_ancillary'][service_name] = timeseries[col].values
        
        return config
    
    @staticmethod
    def _set_nested(dict_obj: Dict, path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = path.split('.')
        for key in keys[:-1]:
            dict_obj = dict_obj.setdefault(key, {})
        dict_obj[keys[-1]] = value
    
    @staticmethod
    def _config_to_bundle(config: Dict) -> DataBundle:
        """Convert configuration dictionary to DataBundle."""
        # Extract main sections
        timegrid_cfg = config['timegrid']
        arch_cfg = config['architecture']
        
        # Create TimeGrid
        timegrid = TimeGrid(**timegrid_cfg)
        T = len(timegrid.index())
        dt_hours = timegrid.dt_minutes / 60
        
        # Parse ancillary services
        services = []
        if 'ancillary_services' in arch_cfg:
            for service_cfg in arch_cfg['ancillary_services']:
                services.append(AncillaryService(**service_cfg))
        
        # Create Architecture
        architecture = Architecture(
            kind=arch_cfg['kind'],
            forbid_retail_to_load=arch_cfg.get('forbid_retail_to_load', True),
            ancillary_services=services,
            permissions=arch_cfg.get('permissions'),
            import_limit_kw=arch_cfg.get('import_limit_kw'),
            export_limit_kw=arch_cfg.get('export_limit_kw')
        )
        
        # Create other components
        ratings = Ratings(**config['ratings'])
        efficiencies = Efficiencies(**config['efficiencies'])
        bounds = Bounds(**config['bounds'])
        degradation = Degradation(**config.get('degradation', {}))
        capacity = Capacity(**config['capacity'])
        
        # Process tariffs with proper numpy conversion
        tariffs_cfg = config['tariffs']
        
        # Convert all price arrays to numpy
        def to_numpy(val):
            if isinstance(val, (list, tuple)):
                return np.array(val)
            elif isinstance(val, np.ndarray):
                return val
            else:
                return val
        
        # Process price_ancillary dict
        price_ancillary = {}
        if 'price_ancillary' in tariffs_cfg:
            for k, v in tariffs_cfg['price_ancillary'].items():
                price_ancillary[k] = to_numpy(v)
        
        tariffs = Tariffs(
            price_retail=to_numpy(tariffs_cfg['price_retail']),
            price_wholesale=to_numpy(tariffs_cfg['price_wholesale']),
            price_cap=to_numpy(tariffs_cfg.get('price_cap', np.zeros(T))),
            cap_mask=to_numpy(tariffs_cfg.get('cap_mask', np.zeros(T, dtype=bool))),
            price_ancillary=price_ancillary,
            price_activation_up=tariffs_cfg.get('price_activation_up', {}),
            price_activation_down=tariffs_cfg.get('price_activation_down', {}),
            price_demand=tariffs_cfg.get('price_demand', 0.0),
            cap_duration_hours=tariffs_cfg.get('cap_duration_hours', 4.0),
            activation_up=tariffs_cfg.get('activation_up', {}),
            activation_down=tariffs_cfg.get('activation_down', {})
        )
        
        # Process exogenous data
        exogenous_cfg = config['exogenous']
        
        # Handle optional PV scaling
        pv_dc = to_numpy(exogenous_cfg['pv_dc'])
        if 'pv_scale_factor' in exogenous_cfg:
            pv_dc *= exogenous_cfg['pv_scale_factor']
        
        # Convert kW to kWh if needed
        load_ac = to_numpy(exogenous_cfg['load_ac'])
        if exogenous_cfg.get('units') == 'kw':
            load_ac *= dt_hours
            pv_dc *= dt_hours
        
        exogenous = Exogenous(
            load_ac=load_ac,
            pv_dc=pv_dc
        )
        
        return DataBundle(
            timegrid=timegrid,
            arch=architecture,
            ratings=ratings,
            eff=efficiencies,
            bounds=bounds,
            degradation=degradation,
            tariffs=tariffs,
            exogenous=exogenous,
            capacity=capacity
        )


class DataWriter:
    """Write optimization results and data to various formats."""
    
    @staticmethod
    def save_results(results, output_path: Union[str, Path], 
                    format: str = 'csv', include_metadata: bool = True):
        """
        Save optimization results to file.
        
        Args:
            results: OptimizationResults object
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'json')
            include_metadata: Whether to include metadata
        """
        output_path = Path(output_path)
        
        # Create results DataFrame
        df = DataWriter._results_to_dataframe(results)
        
        # Save based on format
        if format == 'csv':
            df.to_csv(output_path)
            
            # Save metadata separately if requested
            if include_metadata:
                meta_path = output_path.with_suffix('.meta.json')
                DataWriter._save_metadata(results, meta_path)
                
        elif format == 'parquet':
            # Include metadata in parquet
            if include_metadata:
                df.attrs = DataWriter._get_metadata_dict(results)
            df.to_parquet(output_path)
            
        elif format == 'json':
            # Combine data and metadata
            output = {
                'data': df.to_dict('records'),
                'metadata': DataWriter._get_metadata_dict(results) if include_metadata else {}
            }
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_path}")
    
    @staticmethod
    def _results_to_dataframe(results) -> pd.DataFrame:
        """Convert OptimizationResults to DataFrame."""
        data = {
            'charge_retail': results.charge_retail,
            'charge_wholesale': results.charge_wholesale,
            'charge_pv': results.charge_pv,
            'charge_freq': results.charge_freq,
            'discharge_load': results.discharge_load,
            'discharge_wholesale': results.discharge_wholesale,
            'discharge_freq': results.discharge_freq,
            'soc_total': results.soc_total,
            'pv_to_load': results.pv_to_load,
            'pv_curtailed': results.pv_curtailed,
            'grid_to_load': results.grid_to_load,
        }
        
        # Add reserve capacities
        for k, values in results.reserve_capacity.items():
            data[f'reserve_{k}'] = values
        
        # Add activation energy
        for k, values in results.activation_up.items():
            data[f'activation_up_{k}'] = values
        for k, values in results.activation_down.items():
            data[f'activation_down_{k}'] = values
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _get_metadata_dict(results) -> Dict:
        """Extract metadata dictionary from results."""
        return {
            'objective_value': results.objective_value,
            'solve_time': results.solve_time,
            'status': results.status,
            'capacity_commitment': results.capacity_commitment,
            'financial_metrics': {
                'energy_cost': results.energy_cost,
                'degradation_cost': results.degradation_cost,
                'ancillary_revenue': results.ancillary_revenue,
                'capacity_revenue': results.capacity_revenue,
                'total_revenue': results.total_revenue,
            }
        }
    
    @staticmethod
    def _save_metadata(results, path: Path):
        """Save metadata to JSON file."""
        metadata = DataWriter._get_metadata_dict(results)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def generate_template(output_path: Union[str, Path], 
                     architecture: str = 'ac_coupled',
                     include_ancillary: bool = True):
    """
    Generate a template configuration file.
    
    Args:
        output_path: Where to save the template
        architecture: System architecture type
        include_ancillary: Whether to include ancillary services
    """
    template = {
        'timegrid': {
            'start': '2024-01-01T00:00:00',
            'end': '2024-01-02T00:00:00',
            'dt_minutes': 15,
            'tz': 'UTC'
        },
        'architecture': {
            'kind': architecture,
            'forbid_retail_to_load': True,
            'import_limit_kw': 5000,
            'export_limit_kw': 5000,
        },
        'ratings': {
            'p_charge_max_kw': 2500,
            'p_discharge_max_kw': 2500,
        },
        'efficiencies': {
            'eta_c': 0.95,
            'eta_d': 0.95,
            'self_discharge': 0.0001,
        },
        'bounds': {
            'soc_min': 0.1,
            'soc_max': 0.9,
            'soc_initial': 0.5,
            'enforce_terminal_equality': True,
        },
        'capacity': {
            'capacity_nominal_kwh': 10000,
        },
        'degradation': {
            'kind': 'throughput',
            'phi_c_eur_per_kwh': 0.01,
            'phi_d_eur_per_kwh': 0.01,
        }
    }
    
    # Add architecture-specific parameters
    if architecture == 'ac_coupled':
        template['ratings'].update({
            'p_inv_pv_kw': 3000,
            'p_inv_bess_kw': 2500,
        })
        template['efficiencies'].update({
            'eta_pv_ac_from_dc': 0.98,
            'eta_bess_dc_from_ac': 0.96,
            'eta_bess_ac_from_dc': 0.96,
        })
    else:  # dc_coupled or hybrid
        template['ratings']['p_inv_shared_kw'] = 3000
        template['efficiencies'].update({
            'eta_shared_dc_from_ac': 0.96,
            'eta_shared_ac_from_dc': 0.96,
        })
    
    # Add ancillary services if requested
    if include_ancillary:
        template['architecture']['ancillary_services'] = [
            {
                'name': 'fcr',
                'direction': 'both',
                'sustain_duration_hours': 0.25,
                'settlement_method': 'availability_only',
            },
            {
                'name': 'afrr_up',
                'direction': 'up',
                'sustain_duration_hours': 1.0,
                'activation_fraction_up': 0.1,
                'settlement_method': 'wholesale_settled',
            }
        ]
    
    # Add dummy time series data
    T = 96  # 24 hours at 15-min resolution
    template['tariffs'] = {
        'price_retail': [0.15] * T,
        'price_wholesale': [0.10] * T,
        'price_cap': [100.0] * T,
        'cap_mask': [False] * T,
        'cap_duration_hours': 4.0,
    }
    
    template['exogenous'] = {
        'load_ac': [100] * T,  # 100 kW constant load
        'pv_dc': [0] * T,  # Will be replaced with actual PV profile
        'units': 'kw'
    }
    
    # Save template
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Template saved to {output_path}")
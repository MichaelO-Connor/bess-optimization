# bess/data_sources/entsoe_to_bess.py
"""
Integration module to connect ENTSO-E data directly to BESS optimization.
Handles the complete pipeline from API to optimization results.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging

from ..schema import (
    DataBundle, TimeGrid, Architecture, Ratings, Efficiencies,
    Bounds, Degradation, Tariffs, Exogenous, Capacity, AncillaryService
)
from ..optimize import BESSOptimizer
from .entsoe_integration import (
    ENTSOEDataFetcher, 
    create_milp_parameters_from_entsoe,
    get_country_config
)

logger = logging.getLogger(__name__)


class ENTSOEBESSOptimizer:
    """
    Complete integration of ENTSO-E data with BESS optimization.
    Fetches real market data and runs MILP optimization.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize ENTSO-E BESS optimizer.
        
        Args:
            api_key: ENTSO-E API key (can also use ENTSOE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ENTSOE_API_KEY')
        if not self.api_key:
            raise ValueError("ENTSO-E API key required. Set ENTSOE_API_KEY env var or pass api_key")
    
    def create_bundle_from_entsoe(self,
                                 country: str,
                                 start: datetime,
                                 end: datetime,
                                 bess_config: Dict[str, Any],
                                 dt_minutes: int = 15,
                                 custom_services: List[Dict] = None,
                                 pv_profile: np.ndarray = None,
                                 load_profile: np.ndarray = None,
                                 retail_tariff: Union[float, np.ndarray] = None) -> DataBundle:
        """
        Create a complete DataBundle with ENTSO-E market data.
        
        NOTE: Site-specific load (L_t) MUST be provided via load_profile parameter
        or a default profile will be used. ENTSO-E only provides system-wide grid
        load, not behind-the-meter site load!
        
        Args:
            country: Country name or bidding zone (e.g., 'germany', 'DE_LU', 'netherlands')
            start: Start datetime
            end: End datetime  
            bess_config: BESS configuration dict with keys:
                - capacity_kwh: Battery capacity in kWh
                - power_kw: Battery power in kW (or separate charge/discharge)
                - efficiency: Round-trip efficiency (or separate charge/discharge)
                - soc_min/soc_max: SoC bounds (fractions)
                - architecture: 'ac_coupled', 'dc_coupled', or 'hybrid'
                - degradation_cost: EUR/kWh throughput cost
            dt_minutes: Timestep in minutes (15, 30, or 60)
            custom_services: Override default AS for country
            pv_profile: PV generation profile (kW), optional
            load_profile: Site-specific load profile (kW) - REQUIRED for accurate results!
            retail_tariff: Retail price (EUR/kWh), constant or array
            
        Returns:
            Complete DataBundle ready for optimization
        """
        # Get country configuration
        if country.upper() in ['DE_LU', 'FR', 'GB', 'NL', 'BE', 'AT', 'CH']:
            # Direct bidding zone code
            config = {
                'bidding_zone': country.upper(),
                'timezone': 'Europe/Brussels',  # Default
                'services': custom_services or []
            }
        else:
            # Use pre-configured country settings
            config = get_country_config(country)
            if custom_services:
                config['services'] = custom_services
        
        # Fetch ENTSO-E data
        logger.info(f"Fetching ENTSO-E data for {config['bidding_zone']}")
        
        # Prepare site load if provided
        site_load_kwh = None
        if load_profile is not None:
            # Convert kW to kWh per timestep
            site_load_kwh = load_profile * (dt_minutes / 60.0)
        
        entsoe_data = create_milp_parameters_from_entsoe(
            api_key=self.api_key,
            bidding_zone=config['bidding_zone'],
            start=start,
            end=end,
            dt_minutes=dt_minutes,
            ancillary_services=config['services'],
            timezone=config['timezone'],
            site_load_kwh=site_load_kwh  # Pass site load if available
        )
        
        T = entsoe_data['T']
        dt_hours = entsoe_data['dt_hours']
        
        # Create TimeGrid
        timegrid = TimeGrid(
            start=start.isoformat(),
            end=end.isoformat(),
            dt_minutes=dt_minutes,
            tz=config['timezone']
        )
        
        # Parse BESS configuration
        capacity_kwh = bess_config['capacity_kwh']
        
        # Handle power ratings
        if 'power_kw' in bess_config:
            power_charge = bess_config['power_kw']
            power_discharge = bess_config['power_kw']
        else:
            power_charge = bess_config.get('power_charge_kw', capacity_kwh / 4)
            power_discharge = bess_config.get('power_discharge_kw', capacity_kwh / 4)
        
        # Handle efficiencies
        if 'efficiency' in bess_config:
            # Round-trip efficiency, split evenly
            rt_eff = bess_config['efficiency']
            eta_charge = np.sqrt(rt_eff)
            eta_discharge = np.sqrt(rt_eff)
        else:
            eta_charge = bess_config.get('eta_charge', 0.95)
            eta_discharge = bess_config.get('eta_discharge', 0.95)
        
        # Architecture
        arch_type = bess_config.get('architecture', 'ac_coupled')
        
        # Create Architecture with ancillary services
        ancillary_services = []
        for svc in config['services']:
            ancillary_services.append(AncillaryService(
                name=svc['name'],
                direction=svc.get('direction', 'both'),
                sustain_duration_hours=svc.get('sustain_duration_hours', 1.0),
                settlement_method=svc.get('settlement_method', 'availability_only'),
                activation_fraction_up=entsoe_data['alpha_k_up_t'].get(svc['name'], [0.0])[0] if 'up' in svc.get('direction', 'both') else None,
                activation_fraction_down=entsoe_data['alpha_k_down_t'].get(svc['name'], [0.0])[0] if 'down' in svc.get('direction', 'both') else None
            ))
        
        architecture = Architecture(
            kind=arch_type,
            forbid_retail_to_load=bess_config.get('forbid_retail_to_load', True),
            ancillary_services=ancillary_services
        )
        
        # Create Ratings based on architecture
        ratings_dict = {
            'p_charge_max_kw': power_charge,
            'p_discharge_max_kw': power_discharge,
        }
        
        if arch_type == 'ac_coupled':
            ratings_dict.update({
                'p_inv_pv_kw': bess_config.get('pv_inverter_kw', 0),
                'p_inv_bess_kw': max(power_charge, power_discharge)
            })
        elif arch_type in ['dc_coupled', 'hybrid']:
            ratings_dict.update({
                'p_inv_shared_kw': max(power_charge, power_discharge) * 1.2
            })
        
        ratings = Ratings(**ratings_dict)
        
        # Create Efficiencies based on architecture
        eff_dict = {
            'eta_c': eta_charge,
            'eta_d': eta_discharge,
            'self_discharge': bess_config.get('self_discharge', 0.0001)
        }
        
        if arch_type == 'ac_coupled':
            eff_dict.update({
                'eta_pv_ac_from_dc': 0.98,
                'eta_bess_dc_from_ac': 0.97,
                'eta_bess_ac_from_dc': 0.97
            })
        elif arch_type in ['dc_coupled', 'hybrid']:
            eff_dict.update({
                'eta_shared_dc_from_ac': 0.97,
                'eta_shared_ac_from_dc': 0.97
            })
        
        efficiencies = Efficiencies(**eff_dict)
        
        # Create Bounds
        bounds = Bounds(
            soc_min=bess_config.get('soc_min', 0.1),
            soc_max=bess_config.get('soc_max', 0.9),
            soc_initial=bess_config.get('soc_initial', 0.5),
            enforce_terminal_equality=bess_config.get('enforce_terminal_equality', True),
            cycles_per_day=bess_config.get('cycles_per_day', None)
        )
        
        # Create Capacity
        capacity = Capacity(
            capacity_nominal_kwh=capacity_kwh,
            usable_capacity_kwh=None  # Could add degradation curve
        )
        
        # Create Degradation
        deg_cost = bess_config.get('degradation_cost', 0.01)
        degradation = Degradation(
            kind='throughput',
            phi_c_eur_per_kwh=deg_cost,
            phi_d_eur_per_kwh=deg_cost
        )
        
        # Create Tariffs from ENTSO-E data
        # Wholesale prices from ENTSO-E
        price_wholesale = entsoe_data['P_wh_t']
        
        # Retail prices - use provided or default to wholesale * markup
        if retail_tariff is not None:
            if isinstance(retail_tariff, (int, float)):
                price_retail = np.full(T, retail_tariff)
            else:
                price_retail = retail_tariff
        else:
            # Default: wholesale + 0.10 EUR/kWh markup
            price_retail = price_wholesale + 0.10
        
        # Ancillary service prices
        price_ancillary = entsoe_data.get('pi_k_t', {})
        price_activation_up = entsoe_data.get('u_k_up_t', {})
        price_activation_down = entsoe_data.get('u_k_down_t', {})
        activation_up = entsoe_data.get('alpha_k_up_t', {})
        activation_down = entsoe_data.get('alpha_k_down_t', {})
        
        tariffs = Tariffs(
            price_retail=price_retail,
            price_wholesale=price_wholesale,
            price_ancillary=price_ancillary,
            price_activation_up=price_activation_up,
            price_activation_down=price_activation_down,
            activation_up=activation_up,
            activation_down=activation_down,
            price_cap=np.zeros(T),  # No capacity market by default
            cap_mask=np.zeros(T, dtype=bool),
            cap_duration_hours=4.0
        )
        
        # Create Exogenous data
        # Load - MUST be provided by user (site-specific)
        if load_profile is not None:
            # User provided load profile in kW
            load_kwh = load_profile * dt_hours  # Convert kW to kWh
        else:
            # Default profile - simple commercial building pattern
            hours = np.arange(T) * dt_hours % 24  # Hour of day
            load_kwh = 100 * (1 + 0.3 * np.sin(2 * np.pi * (hours - 6) / 12)) * dt_hours
            load_kwh[hours < 6] *= 0.5  # Lower at night
            load_kwh[hours > 20] *= 0.5
            logger.warning("No site load provided - using default commercial building profile")
        
        # PV - use provided or default
        if pv_profile is not None:
            pv_kwh = pv_profile * dt_hours  # Convert kW to kWh
        else:
            # Default: no PV unless inverter specified
            if bess_config.get('pv_inverter_kw', 0) > 0:
                hours = np.arange(T) * dt_hours % 24
                pv_kwh = np.maximum(0, bess_config['pv_inverter_kw'] * 
                                   np.sin(np.pi * np.maximum(0, hours - 6) / 12) ** 2) * dt_hours
                pv_kwh[hours < 6] = 0
                pv_kwh[hours > 18] = 0
            else:
                pv_kwh = np.zeros(T)
        
        exogenous = Exogenous(
            load_ac=load_kwh,
            pv_dc=pv_kwh
        )
        
        # Create and validate bundle
        bundle = DataBundle(
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
        
        bundle.validate_lengths()
        bundle.validate_architecture_requirements()
        
        return bundle
    
    def optimize_with_entsoe(self,
                            country: str,
                            start: datetime,
                            end: datetime,
                            bess_config: Dict[str, Any],
                            **kwargs):
        """
        Complete optimization with ENTSO-E data.
        
        Args:
            country: Country or bidding zone
            start: Start datetime
            end: End datetime
            bess_config: BESS configuration
            **kwargs: Additional arguments for create_bundle_from_entsoe
            
        Returns:
            OptimizationResults
            
        Example:
            >>> optimizer = ENTSOEBESSOptimizer()
            >>> results = optimizer.optimize_with_entsoe(
            ...     country='germany',
            ...     start=datetime(2024, 6, 1),
            ...     end=datetime(2024, 6, 2),
            ...     bess_config={
            ...         'capacity_kwh': 1000,
            ...         'power_kw': 250,
            ...         'efficiency': 0.90,
            ...         'soc_min': 0.1,
            ...         'soc_max': 0.9
            ...     }
            ... )
        """
        # Create bundle with ENTSO-E data
        bundle = self.create_bundle_from_entsoe(
            country, start, end, bess_config, **kwargs
        )
        
        # Run optimization
        optimizer = BESSOptimizer(bundle)
        results = optimizer.optimize()
        
        # Print summary
        kpis = optimizer.get_kpis()
        print(f"\n=== Optimization Results for {country.upper()} ===")
        print(f"Period: {start} to {end}")
        print(f"Battery: {bess_config['capacity_kwh']} kWh")
        print(f"\nFinancial Results:")
        print(f"  Total Revenue: €{kpis['total_revenue']:,.2f}")
        print(f"  Energy Cost: €{kpis['energy_cost']:,.2f}")
        print(f"  Ancillary Revenue: €{kpis['ancillary_revenue']:,.2f}")
        print(f"  Degradation Cost: €{kpis['degradation_cost']:,.2f}")
        print(f"\nOperational Metrics:")
        print(f"  Avg Daily Cycles: {kpis['avg_daily_cycles']:.2f}")
        print(f"  Utilization: {kpis['utilization']:.1%}")
        
        return results


def quick_entsoe_optimization(country: str = 'germany',
                             days: int = 1,
                             capacity_kwh: float = 1000,
                             power_kw: float = 250,
                             api_key: str = None) -> Any:
    """
    Quick one-liner to run BESS optimization with real ENTSO-E data.
    
    Args:
        country: Country name ('germany', 'netherlands', 'france', 'gb', 'belgium')
        days: Number of days to optimize
        capacity_kwh: Battery capacity in kWh
        power_kw: Battery power in kW
        api_key: ENTSO-E API key (or set ENTSOE_API_KEY env var)
        
    Returns:
        OptimizationResults
        
    Example:
        >>> # Requires: export ENTSOE_API_KEY='your_key_here'
        >>> results = quick_entsoe_optimization('netherlands', days=7, capacity_kwh=2000)
    """
    # Use tomorrow as start date (today's data might not be complete)
    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=days)
    
    # Simple BESS config
    bess_config = {
        'capacity_kwh': capacity_kwh,
        'power_kw': power_kw,
        'efficiency': 0.90,
        'soc_min': 0.1,
        'soc_max': 0.9,
        'architecture': 'ac_coupled',
        'degradation_cost': 0.02  # 20 EUR/MWh
    }
    
    # Run optimization
    optimizer = ENTSOEBESSOptimizer(api_key)
    results = optimizer.optimize_with_entsoe(
        country=country,
        start=start,
        end=end,
        bess_config=bess_config,
        dt_minutes=60  # Hourly for quick run
    )
    
    return results
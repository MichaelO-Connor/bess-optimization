# bess/data_sources/entsoe_integration.py
"""
ENTSO-E data integration using entsoe-py library.
Fetches and formats market data for MILP optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient
from entsoe.mappings import BIDDING_ZONES, NEIGHBOURS
import logging
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class AncillaryServiceConfig:
    """Configuration for an ancillary service product."""
    name: str                           # Internal name (e.g., 'fcr', 'afrr_up')
    direction: str                       # 'up', 'down', or 'both'
    sustain_duration_hours: float       # H^k - from market rules
    settlement_method: str               # 'availability_only', 'wholesale_settled', 'explicit'
    entsoe_product_type: Optional[str]  # For reserve queries (e.g., 'FCR', 'aFRR')
    activation_default: float = 0.0     # Default α^k if no historical data


class ENTSOEDataFetcher:
    """
    Fetches and formats ENTSO-E data for MILP optimization.
    Maps ENTSO-E data to MILP parameters according to Section 8 formulation.
    """
    
    # ENTSO-E to MILP parameter mapping
    PARAM_MAPPING = {
        'wholesale_price': 'P_wh_t',          # Day-ahead prices
        'ancillary_availability': 'pi_k_t',   # Reserve availability prices
        'activation_price': 'u_k_t',          # Activation settlement prices
        'activation_fraction': 'alpha_k_t',   # Expected activation rates
        'load_forecast': 'L_t',               # Load forecast (if available)
    }
    
    def __init__(self, api_key: str, bidding_zone: str = 'DE_LU', 
                 timezone: str = 'Europe/Berlin'):
        """
        Initialize ENTSO-E data fetcher.
        
        Args:
            api_key: ENTSO-E API security token
            bidding_zone: Market area code (e.g., 'DE_LU', 'FR', 'GB', 'NL')
            timezone: Target timezone for data alignment
        """
        self.client = EntsoePandasClient(api_key=api_key)
        self.bidding_zone = bidding_zone
        self.timezone = timezone
        
        # Validate bidding zone
        if bidding_zone not in BIDDING_ZONES.values():
            logger.warning(f"Bidding zone {bidding_zone} not in standard list")
    
    def fetch_milp_data(self, 
                        start: pd.Timestamp,
                        end: pd.Timestamp,
                        dt_minutes: int = 15,
                        ancillary_services: List[AncillaryServiceConfig] = None,
                        include_system_data: bool = False) -> Dict[str, Any]:
        """
        Fetch market data from ENTSO-E for MILP optimization.
        
        NOTE: Does NOT fetch site-specific load (L_t) as this must come from 
        meter data or user input. ENTSO-E only has system-wide grid load.
        
        Args:
            start: Start timestamp (timezone-aware)
            end: End timestamp (timezone-aware)
            dt_minutes: MILP timestep in minutes (must match optimization)
            ancillary_services: List of AS configurations to fetch
            include_system_data: Whether to fetch system-wide data (for analysis, not MILP)
            
        Returns:
            Dictionary with MILP-ready parameters:
            {
                'P_wh_t': np.array,           # Wholesale prices (EUR/kWh)
                'pi_k_t': Dict[str, np.array], # AS availability prices (EUR/kW/period)
                'u_k_up_t': Dict[str, np.array],   # Upward activation prices (EUR/kWh)
                'u_k_down_t': Dict[str, np.array], # Downward activation prices (EUR/kWh)
                'alpha_k_up_t': Dict[str, np.array],   # Expected activation up
                'alpha_k_down_t': Dict[str, np.array], # Expected activation down
                'system_load': np.array,      # System-wide load if requested (MW) - NOT for MILP
                'timestamps': pd.DatetimeIndex,
                'metadata': Dict
            }
        """
        logger.info(f"Fetching ENTSO-E data for {self.bidding_zone} from {start} to {end}")
        
        # Initialize result containers
        result = {
            'pi_k_t': {},
            'u_k_up_t': {},
            'u_k_down_t': {},
            'alpha_k_up_t': {},
            'alpha_k_down_t': {},
            'timestamps': pd.date_range(start, end, freq=f'{dt_minutes}min', 
                                       inclusive='left', tz=self.timezone),
            'metadata': {
                'bidding_zone': self.bidding_zone,
                'timezone': self.timezone,
                'dt_minutes': dt_minutes,
                'fetch_time': pd.Timestamp.now()
            }
        }
        
        # Number of timesteps
        T = len(result['timestamps'])
        dt_hours = dt_minutes / 60.0
        
        # 1. Fetch wholesale prices (day-ahead)
        result['P_wh_t'] = self._fetch_wholesale_prices(start, end, dt_minutes)
        
        # 2. Fetch ancillary service data
        if ancillary_services:
            for service in ancillary_services:
                self._fetch_ancillary_service_data(
                    service, start, end, dt_minutes, dt_hours, result
                )
        
        # 3. Optionally fetch system-wide data (NOT for MILP L_t!)
        if include_system_data:
            result['system_load'] = self._fetch_system_load(start, end, dt_minutes)
            logger.info("Fetched system-wide load (NOT site-specific L_t)")
        
        # 4. Validate and align all time series
        result = self._validate_and_align_data(result, T)
        
        return result
    
    def _fetch_wholesale_prices(self, start: pd.Timestamp, end: pd.Timestamp, 
                               dt_minutes: int) -> np.ndarray:
        """Fetch day-ahead wholesale prices."""
        try:
            logger.info("Fetching day-ahead prices...")
            
            # Query day-ahead prices (returns EUR/MWh)
            prices = self.client.query_day_ahead_prices(
                self.bidding_zone, 
                start=start, 
                end=end
            )
            
            # Convert to target timezone and resample
            prices = prices.tz_convert(self.timezone)
            prices = prices.resample(f'{dt_minutes}min').ffill()
            
            # Convert EUR/MWh to EUR/kWh
            prices = prices / 1000.0
            
            # Ensure we have the right time range
            prices = prices[start:end]
            
            logger.info(f"Fetched {len(prices)} wholesale price points")
            return prices.values
            
        except Exception as e:
            logger.error(f"Failed to fetch wholesale prices: {e}")
            # Return zeros as fallback
            return np.zeros(len(pd.date_range(start, end, freq=f'{dt_minutes}min', 
                                             inclusive='left')))
    
    def _fetch_ancillary_service_data(self, 
                                     service: AncillaryServiceConfig,
                                     start: pd.Timestamp,
                                     end: pd.Timestamp,
                                     dt_minutes: int,
                                     dt_hours: float,
                                     result: Dict) -> None:
        """Fetch data for a specific ancillary service."""
        logger.info(f"Fetching ancillary service data for {service.name}")
        
        # Fetch availability prices
        availability_prices = self._fetch_reserve_prices(
            service, start, end, dt_minutes
        )
        
        if availability_prices is not None:
            result['pi_k_t'][service.name] = availability_prices
        
        # Fetch or set activation prices based on settlement method
        if service.settlement_method == 'availability_only':
            # No activation settlement
            if service.direction in ['up', 'both']:
                result['u_k_up_t'][service.name] = np.zeros(len(result['timestamps']))
            if service.direction in ['down', 'both']:
                result['u_k_down_t'][service.name] = np.zeros(len(result['timestamps']))
                
        elif service.settlement_method == 'wholesale_settled':
            # Use wholesale prices for settlement
            if service.direction in ['up', 'both']:
                result['u_k_up_t'][service.name] = result['P_wh_t'].copy()
            if service.direction in ['down', 'both']:
                result['u_k_down_t'][service.name] = result['P_wh_t'].copy()
                
        else:  # explicit prices
            activation_prices = self._fetch_activation_prices(
                service, start, end, dt_minutes
            )
            if activation_prices is not None:
                if service.direction in ['up', 'both']:
                    result['u_k_up_t'][service.name] = activation_prices.get('up', 
                                                        np.zeros(len(result['timestamps'])))
                if service.direction in ['down', 'both']:
                    result['u_k_down_t'][service.name] = activation_prices.get('down',
                                                         np.zeros(len(result['timestamps'])))
        
        # Fetch or estimate activation fractions
        activation_fractions = self._fetch_activation_fractions(
            service, start, end, dt_minutes, dt_hours
        )
        
        if service.direction in ['up', 'both']:
            result['alpha_k_up_t'][service.name] = activation_fractions.get('up',
                                                   np.full(len(result['timestamps']), 
                                                          service.activation_default))
        if service.direction in ['down', 'both']:
            result['alpha_k_down_t'][service.name] = activation_fractions.get('down',
                                                     np.full(len(result['timestamps']),
                                                            service.activation_default))
    
    def _fetch_reserve_prices(self, service: AncillaryServiceConfig,
                             start: pd.Timestamp, end: pd.Timestamp,
                             dt_minutes: int) -> Optional[np.ndarray]:
        """Fetch reserve availability prices."""
        try:
            # Map service names to ENTSO-E types
            # This varies by country - these are common mappings
            reserve_type_map = {
                'fcr': 'FCR',
                'afrr_up': 'Secondary reserve',
                'afrr_down': 'Secondary reserve',
                'mfrr_up': 'Tertiary reserve',
                'mfrr_down': 'Tertiary reserve',
            }
            
            reserve_type = service.entsoe_product_type or reserve_type_map.get(service.name)
            
            if not reserve_type:
                logger.warning(f"No ENTSO-E mapping for service {service.name}")
                return None
            
            # Try different query methods depending on market
            prices = None
            
            # Method 1: Try contracted reserve prices
            try:
                prices = self.client.query_contracted_reserve_prices(
                    self.bidding_zone,
                    start=start,
                    end=end,
                    type_marketagreement_type=reserve_type,
                    psr_type=None  # All types
                )
            except Exception as e:
                logger.debug(f"Contracted reserve prices not available: {e}")
            
            # Method 2: Try procured balancing capacity
            if prices is None or prices.empty:
                try:
                    prices = self.client.query_procured_balancing_capacity(
                        self.bidding_zone,
                        start=start,
                        end=end,
                        process_type='A51' if 'up' in service.name else 'A52'
                    )
                except Exception as e:
                    logger.debug(f"Procured capacity not available: {e}")
            
            if prices is None or prices.empty:
                logger.warning(f"No reserve prices found for {service.name}")
                return None
            
            # Process prices
            if isinstance(prices, pd.DataFrame):
                # Take mean across columns if multiple products
                prices = prices.mean(axis=1)
            
            # Resample and convert
            prices = prices.tz_convert(self.timezone)
            prices = prices.resample(f'{dt_minutes}min').ffill()
            
            # Prices are typically in EUR/MW/h, convert to EUR/kW/period
            # EUR/MW/h * (1 MW / 1000 kW) * (dt_hours h / period) = EUR/kW/period
            prices = prices / 1000.0 * (dt_minutes / 60.0)
            
            # Ensure correct time range
            prices = prices[start:end]
            
            logger.info(f"Fetched {len(prices)} reserve price points for {service.name}")
            return prices.values
            
        except Exception as e:
            logger.error(f"Failed to fetch reserve prices for {service.name}: {e}")
            return None
    
    def _fetch_activation_prices(self, service: AncillaryServiceConfig,
                                start: pd.Timestamp, end: pd.Timestamp,
                                dt_minutes: int) -> Optional[Dict[str, np.ndarray]]:
        """Fetch activation/imbalance prices."""
        try:
            result = {}
            
            # Query activated balancing energy prices
            # These are the prices paid for actual energy delivery
            prices = self.client.query_imbalance_prices(
                self.bidding_zone,
                start=start,
                end=end,
                psr_type=None
            )
            
            if prices is None or prices.empty:
                return None
            
            # Process for up/down directions
            # Column names vary by market
            if isinstance(prices, pd.DataFrame):
                # Look for positive/negative imbalance columns
                for col in prices.columns:
                    if 'Long' in col or 'Surplus' in col or 'Positive' in col:
                        down_prices = prices[col]
                    elif 'Short' in col or 'Deficit' in col or 'Negative' in col:
                        up_prices = prices[col]
                
                # Resample and convert
                if 'up' in service.direction or service.direction == 'both':
                    if 'up_prices' in locals():
                        up_prices = up_prices.tz_convert(self.timezone)
                        up_prices = up_prices.resample(f'{dt_minutes}min').ffill()
                        up_prices = up_prices / 1000.0  # EUR/MWh to EUR/kWh
                        result['up'] = up_prices[start:end].values
                
                if 'down' in service.direction or service.direction == 'both':
                    if 'down_prices' in locals():
                        down_prices = down_prices.tz_convert(self.timezone)
                        down_prices = down_prices.resample(f'{dt_minutes}min').ffill()
                        down_prices = down_prices / 1000.0  # EUR/MWh to EUR/kWh
                        result['down'] = down_prices[start:end].values
            
            return result if result else None
            
        except Exception as e:
            logger.debug(f"Failed to fetch activation prices for {service.name}: {e}")
            return None
    
    def _fetch_activation_fractions(self, service: AncillaryServiceConfig,
                                   start: pd.Timestamp, end: pd.Timestamp,
                                   dt_minutes: int, dt_hours: float) -> Dict[str, np.ndarray]:
        """
        Estimate activation fractions from historical data.
        α = activated_energy / (committed_capacity * dt)
        """
        result = {}
        
        try:
            # For historical estimation, look back further
            hist_start = start - pd.Timedelta(days=30)
            
            # Fetch activated energy
            activated = self.client.query_activated_balancing_energy(
                self.bidding_zone,
                start=hist_start,
                end=start,
                business_type='A96' if 'up' in service.name else 'A97'
            )
            
            # Fetch committed capacity
            committed = self.client.query_procured_balancing_capacity(
                self.bidding_zone,
                start=hist_start,
                end=start,
                process_type='A51' if 'up' in service.name else 'A52'
            )
            
            if activated is not None and committed is not None:
                # Calculate activation fraction
                # Both should be resampled to same frequency
                activated = activated.resample(f'{dt_minutes}min').sum()  # MWh
                committed = committed.resample(f'{dt_minutes}min').mean()  # MW
                
                # α = MWh / (MW * hours)
                alpha = activated / (committed * dt_hours)
                alpha = alpha.clip(0, 1)  # Bound to [0,1]
                
                # Use mean as estimate for future
                mean_alpha = alpha.mean()
                if pd.notna(mean_alpha):
                    if 'up' in service.direction or service.direction == 'both':
                        result['up'] = np.full(len(pd.date_range(start, end, 
                                                                freq=f'{dt_minutes}min',
                                                                inclusive='left')),
                                             mean_alpha)
                    if 'down' in service.direction or service.direction == 'both':
                        result['down'] = np.full(len(pd.date_range(start, end,
                                                                  freq=f'{dt_minutes}min',
                                                                  inclusive='left')),
                                               mean_alpha)
                    
                    logger.info(f"Estimated activation fraction for {service.name}: {mean_alpha:.3f}")
            
        except Exception as e:
            logger.debug(f"Could not estimate activation fractions for {service.name}: {e}")
        
        return result
    
    def _fetch_system_load(self, start: pd.Timestamp, end: pd.Timestamp,
                          dt_minutes: int) -> np.ndarray:
        """
        Fetch system-wide load forecast (entire grid).
        NOTE: This is NOT site-specific load for MILP!
        
        Returns:
            System load in MW (not kWh) for analysis purposes
        """
        try:
            logger.info("Fetching system-wide load forecast...")
            
            # Query day-ahead load forecast for entire bidding zone
            load = self.client.query_load_forecast(
                self.bidding_zone,
                start=start,
                end=end
            )
            
            if load is None or load.empty:
                logger.warning("No system load forecast available")
                return np.array([])
            
            # Convert to target timezone and resample
            load = load.tz_convert(self.timezone)
            load = load.resample(f'{dt_minutes}min').interpolate()
            
            # Keep in MW for system-wide data
            # Ensure correct time range
            load = load[start:end]
            
            logger.info(f"Fetched {len(load)} system load points (MW)")
            return load.values
            
        except Exception as e:
            logger.error(f"Failed to fetch system load: {e}")
            return np.array([])
    
    def _validate_and_align_data(self, result: Dict, T: int) -> Dict:
        """Validate and align all time series to consistent length."""
        # Ensure all arrays are correct length
        if len(result.get('P_wh_t', [])) != T:
            logger.warning(f"Wholesale prices length mismatch, padding/truncating")
            result['P_wh_t'] = self._ensure_length(result.get('P_wh_t', []), T, 0.1)
        
        # Ensure all AS arrays are correct length
        for service_dict in ['pi_k_t', 'u_k_up_t', 'u_k_down_t', 
                            'alpha_k_up_t', 'alpha_k_down_t']:
            if service_dict in result:
                for service_name, values in result[service_dict].items():
                    if len(values) != T:
                        default = 0.0 if 'alpha' not in service_dict else 0.0
                        result[service_dict][service_name] = self._ensure_length(values, T, default)
        
        # System load is optional and not used in MILP
        if 'system_load' in result and len(result.get('system_load', [])) > 0:
            if len(result['system_load']) != T:
                result['system_load'] = self._ensure_length(result['system_load'], T, 0.0)
        
        return result
    
    def _ensure_length(self, arr: np.ndarray, target_length: int, 
                      fill_value: float) -> np.ndarray:
        """Ensure array has target length by padding or truncating."""
        if len(arr) == 0:
            return np.full(target_length, fill_value)
        elif len(arr) < target_length:
            # Pad with last value or fill_value
            return np.pad(arr, (0, target_length - len(arr)), 
                         'constant', constant_values=arr[-1])
        elif len(arr) > target_length:
            # Truncate
            return arr[:target_length]
        return arr


def create_milp_parameters_from_entsoe(
    api_key: str,
    bidding_zone: str,
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    ancillary_services: List[Dict[str, Any]] = None,
    timezone: str = 'Europe/Berlin',
    site_load_kwh: np.ndarray = None
) -> Dict[str, Any]:
    """
    Convenience function to fetch ENTSO-E data and format for MILP.
    
    Args:
        api_key: ENTSO-E API token
        bidding_zone: Market area (e.g., 'DE_LU', 'FR', 'GB', 'NL')
        start: Start time
        end: End time
        dt_minutes: Timestep in minutes
        ancillary_services: List of AS configurations
        timezone: Target timezone
        site_load_kwh: Site-specific load profile (kWh per timestep)
                      Must be provided by user - NOT from ENTSO-E!
        
    Returns:
        Dictionary ready for MILP parameter creation
    
    Example:
        >>> # You must provide site load!
        >>> site_load = np.array([100, 120, 150, ...])  # kWh per timestep
        >>> services = [
        ...     {'name': 'fcr', 'direction': 'both', 
        ...      'sustain_duration_hours': 0.25,
        ...      'settlement_method': 'availability_only'},
        ...     {'name': 'afrr_up', 'direction': 'up',
        ...      'sustain_duration_hours': 1.0,
        ...      'settlement_method': 'wholesale_settled'}
        ... ]
        >>> params = create_milp_parameters_from_entsoe(
        ...     api_key='your_key',
        ...     bidding_zone='DE_LU',
        ...     start=datetime(2024, 6, 1),
        ...     end=datetime(2024, 6, 2),
        ...     ancillary_services=services,
        ...     site_load_kwh=site_load
        ... )
    """
    # Convert to pandas timestamps
    start_ts = pd.Timestamp(start, tz=timezone)
    end_ts = pd.Timestamp(end, tz=timezone)
    
    # Create service configs
    service_configs = []
    if ancillary_services:
        for svc in ancillary_services:
            service_configs.append(AncillaryServiceConfig(
                name=svc['name'],
                direction=svc.get('direction', 'both'),
                sustain_duration_hours=svc.get('sustain_duration_hours', 1.0),
                settlement_method=svc.get('settlement_method', 'availability_only'),
                entsoe_product_type=svc.get('entsoe_product_type'),
                activation_default=svc.get('activation_default', 0.0)
            ))
    
    # Create fetcher and get data
    fetcher = ENTSOEDataFetcher(api_key, bidding_zone, timezone)
    data = fetcher.fetch_milp_data(
        start_ts, end_ts, dt_minutes, 
        service_configs, include_system_data=False
    )
    
    T = len(data['timestamps'])
    
    # Handle site load
    if site_load_kwh is not None:
        if len(site_load_kwh) != T:
            raise ValueError(f"Site load length {len(site_load_kwh)} != {T} timesteps")
        L_t = site_load_kwh
    else:
        # Default profile if not provided
        logger.warning("No site load provided - using default 100 kWh profile")
        L_t = np.full(T, 100.0)
    
    # Format for MILP
    milp_params = {
        'T': T,
        'dt_hours': dt_minutes / 60.0,
        'time_index': data['timestamps'],
        'tz': timezone,
        
        # Core time series
        'P_wh_t': data['P_wh_t'],
        'L_t': L_t,  # Site-specific load from user
        
        # Ancillary services
        'K': [svc['name'] for svc in (ancillary_services or [])],
        'K_up': [svc['name'] for svc in (ancillary_services or []) 
                 if svc.get('direction') in ['up', 'both']],
        'K_down': [svc['name'] for svc in (ancillary_services or [])
                   if svc.get('direction') in ['down', 'both']],
        
        'pi_k_t': data.get('pi_k_t', {}),
        'u_k_up_t': data.get('u_k_up_t', {}),
        'u_k_down_t': data.get('u_k_down_t', {}),
        'alpha_k_up_t': data.get('alpha_k_up_t', {}),
        'alpha_k_down_t': data.get('alpha_k_down_t', {}),
        
        # Sustain durations (from service configs)
        'H_k_up': {svc['name']: svc['sustain_duration_hours'] 
                   for svc in (ancillary_services or [])
                   if svc.get('direction') in ['up', 'both']},
        'H_k_down': {svc['name']: svc['sustain_duration_hours']
                     for svc in (ancillary_services or [])
                     if svc.get('direction') in ['down', 'both']},
        
        'metadata': data.get('metadata', {})
    }
    
    return milp_params


# Pre-configured country settings for common markets
COUNTRY_CONFIGS = {
    'germany': {
        'bidding_zone': 'DE_LU',
        'timezone': 'Europe/Berlin',
        'services': [
            {'name': 'fcr', 'direction': 'both', 'sustain_duration_hours': 0.25,
             'settlement_method': 'availability_only', 'entsoe_product_type': 'FCR'},
            {'name': 'afrr_up', 'direction': 'up', 'sustain_duration_hours': 1.0,
             'settlement_method': 'wholesale_settled'},
            {'name': 'afrr_down', 'direction': 'down', 'sustain_duration_hours': 1.0,
             'settlement_method': 'wholesale_settled'},
        ]
    },
    'netherlands': {
        'bidding_zone': 'NL',
        'timezone': 'Europe/Amsterdam',
        'services': [
            {'name': 'fcr', 'direction': 'both', 'sustain_duration_hours': 0.25,
             'settlement_method': 'availability_only'},
            {'name': 'afrr', 'direction': 'both', 'sustain_duration_hours': 1.0,
             'settlement_method': 'wholesale_settled'},
        ]
    },
    'france': {
        'bidding_zone': 'FR',
        'timezone': 'Europe/Paris',
        'services': [
            {'name': 'fcr', 'direction': 'both', 'sustain_duration_hours': 0.5,
             'settlement_method': 'availability_only'},
            {'name': 'afrr', 'direction': 'both', 'sustain_duration_hours': 1.0,
             'settlement_method': 'explicit'},
        ]
    },
    'gb': {
        'bidding_zone': 'GB',
        'timezone': 'Europe/London',
        'services': [
            {'name': 'ffr', 'direction': 'both', 'sustain_duration_hours': 0.5,
             'settlement_method': 'availability_only'},
            {'name': 'dc', 'direction': 'both', 'sustain_duration_hours': 1.0,
             'settlement_method': 'availability_only'},
        ]
    },
    'belgium': {
        'bidding_zone': 'BE',
        'timezone': 'Europe/Brussels',
        'services': [
            {'name': 'fcr', 'direction': 'both', 'sustain_duration_hours': 0.25,
             'settlement_method': 'availability_only'},
            {'name': 'afrr', 'direction': 'both', 'sustain_duration_hours': 1.0,
             'settlement_method': 'wholesale_settled'},
        ]
    }
}


def get_country_config(country: str) -> Dict[str, Any]:
    """Get pre-configured settings for a country."""
    country_lower = country.lower()
    if country_lower in COUNTRY_CONFIGS:
        return COUNTRY_CONFIGS[country_lower]
    else:
        raise ValueError(f"No configuration for country: {country}. "
                        f"Available: {list(COUNTRY_CONFIGS.keys())}")

# bess/data_sources/load_profiles.py
"""
Helper functions for creating site-specific load profiles.
Since ENTSO-E doesn't provide site load, these utilities help create realistic profiles.

Now includes standard load profiles for more realistic generation based on actual data.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Try to import standard load profile generator
try:
    from .standard_load_profiles import create_standard_load_profile, StandardLoadProfileGenerator
    STANDARD_PROFILES_AVAILABLE = True
except ImportError:
    STANDARD_PROFILES_AVAILABLE = False
    logger.warning("Standard load profiles not available")


def create_commercial_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    base_load_kw: float = 100,
    peak_load_kw: float = 500,
    business_hours: tuple = (7, 19),
    weekend_factor: float = 0.3,
    use_standard_profiles: bool = True
) -> np.ndarray:
    """
    Create a commercial building load profile.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        base_load_kw: Nighttime/weekend base load (only used for synthetic)
        peak_load_kw: Peak daytime load
        business_hours: (start_hour, end_hour) for business operations (only used for synthetic)
        weekend_factor: Weekend load as fraction of weekday (only used for synthetic)
        use_standard_profiles: Use standard profiles if available
        
    Returns:
        Load profile in kW
    """
    if use_standard_profiles and STANDARD_PROFILES_AVAILABLE:
        return _create_standard_profile(
            start=start,
            end=end,
            dt_minutes=dt_minutes,
            peak_power_kw=peak_load_kw,
            profile_type='commercial'
        )
    
    # Fallback to synthetic profile
    return _create_synthetic_commercial_profile(
        start, end, dt_minutes, base_load_kw, peak_load_kw, 
        business_hours, weekend_factor
    )


def create_industrial_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    base_load_kw: float = 500,
    production_load_kw: float = 2000,
    shifts: list = [(6, 14), (14, 22)],  # 2-shift operation
    weekend_production: bool = False,
    use_standard_profiles: bool = True
) -> np.ndarray:
    """
    Create an industrial facility load profile with shifts.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        base_load_kw: Continuous process base load (only used for synthetic)
        production_load_kw: Additional load during production (only used for synthetic)
        shifts: List of (start_hour, end_hour) tuples for shifts (only used for synthetic)
        weekend_production: Whether production runs on weekends (only used for synthetic)
        use_standard_profiles: Use standard profiles if available
        
    Returns:
        Load profile in kW
    """
    total_peak = base_load_kw + production_load_kw
    
    if use_standard_profiles and STANDARD_PROFILES_AVAILABLE:
        return _create_standard_profile(
            start=start,
            end=end,
            dt_minutes=dt_minutes,
            peak_power_kw=total_peak,
            profile_type='industrial'
        )
    
    # Fallback to synthetic profile
    return _create_synthetic_industrial_profile(
        start, end, dt_minutes, base_load_kw, production_load_kw,
        shifts, weekend_production
    )


def create_residential_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    num_households: int = 100,
    avg_daily_kwh: float = 10,
    morning_peak_hour: int = 7,
    evening_peak_hour: int = 19,
    use_standard_profiles: bool = True
) -> np.ndarray:
    """
    Create aggregated residential load profile.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        num_households: Number of households (only used for synthetic)
        avg_daily_kwh: Average daily consumption per household (only used for synthetic)
        morning_peak_hour: Hour of morning peak (only used for synthetic)
        evening_peak_hour: Hour of evening peak (only used for synthetic)
        use_standard_profiles: Use standard profiles if available
        
    Returns:
        Load profile in kW
    """
    # Estimate peak power from daily consumption for standard profiles
    peak_power_kw = num_households * avg_daily_kwh / 24 * 3.0  # Rough peak factor
    
    if use_standard_profiles and STANDARD_PROFILES_AVAILABLE:
        return _create_standard_profile(
            start=start,
            end=end,
            dt_minutes=dt_minutes,
            peak_power_kw=peak_power_kw,
            profile_type='domestic'
        )
    
    # Fallback to synthetic profile
    return _create_synthetic_residential_profile(
        start, end, dt_minutes, num_households, avg_daily_kwh,
        morning_peak_hour, evening_peak_hour
    )


def _create_standard_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int,
    peak_power_kw: float,
    profile_type: str
) -> np.ndarray:
    """Create load profile using standard profiles."""
    try:
        days = (end - start).days + 1
        return create_standard_load_profile(
            peak_power_kw=peak_power_kw,
            start_date=start,
            days=days,
            profile_type=profile_type,  # This is ignored in the simplified version
            dt_minutes=dt_minutes
        )
    except Exception as e:
        logger.warning(f"Failed to create standard profile: {e}")
        # Fall back to synthetic
        if profile_type in ['commercial', 'industrial']:
            return _create_synthetic_commercial_profile(
                start, end, dt_minutes, peak_power_kw * 0.3, peak_power_kw, (7, 19), 0.3
            )
        else:  # domestic/residential
            return _create_synthetic_residential_profile(
                start, end, dt_minutes, 100, 10, 7, 19
            )


def _create_synthetic_commercial_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int,
    base_load_kw: float,
    peak_load_kw: float,
    business_hours: tuple,
    weekend_factor: float
) -> np.ndarray:
    """Create synthetic commercial profile (original implementation)."""
    time_index = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    load = np.zeros(len(time_index))
    
    for i, ts in enumerate(time_index):
        hour = ts.hour + ts.minute / 60
        is_weekend = ts.weekday() >= 5
        
        if is_weekend:
            # Weekend load
            load[i] = base_load_kw + (peak_load_kw - base_load_kw) * weekend_factor * 0.5
        elif business_hours[0] <= hour < business_hours[1]:
            # Business hours - ramp up/down pattern
            hours_into_day = hour - business_hours[0]
            hours_in_operation = business_hours[1] - business_hours[0]
            
            # Bell curve during business hours
            normalized_time = hours_into_day / hours_in_operation
            activity_factor = np.sin(np.pi * normalized_time) ** 0.5
            
            load[i] = base_load_kw + (peak_load_kw - base_load_kw) * activity_factor
        else:
            # Non-business hours
            load[i] = base_load_kw
    
    return load


def _create_synthetic_industrial_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int,
    base_load_kw: float,
    production_load_kw: float,
    shifts: list,
    weekend_production: bool
) -> np.ndarray:
    """Create synthetic industrial profile (original implementation)."""
    time_index = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    load = np.zeros(len(time_index))
    
    for i, ts in enumerate(time_index):
        hour = ts.hour + ts.minute / 60
        is_weekend = ts.weekday() >= 5
        
        # Base load always present
        load[i] = base_load_kw
        
        # Check if production should run
        if is_weekend and not weekend_production:
            continue
            
        # Check if within any shift
        for shift_start, shift_end in shifts:
            if shift_start <= hour < shift_end:
                # Ramp up/down at shift boundaries (30 min ramp)
                ramp_duration = 0.5  # hours
                
                if hour < shift_start + ramp_duration:
                    # Ramping up
                    ramp_factor = (hour - shift_start) / ramp_duration
                elif hour > shift_end - ramp_duration:
                    # Ramping down
                    ramp_factor = (shift_end - hour) / ramp_duration
                else:
                    # Full production
                    ramp_factor = 1.0
                
                load[i] += production_load_kw * ramp_factor
                break
    
    return load


def _create_synthetic_residential_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int,
    num_households: int,
    avg_daily_kwh: float,
    morning_peak_hour: int,
    evening_peak_hour: int
) -> np.ndarray:
    """Create synthetic residential profile (original implementation)."""
    time_index = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    load = np.zeros(len(time_index))
    
    # Base load per household (kW)
    base_load_per_hh = avg_daily_kwh / 24 * 0.5  # 50% is base
    
    for i, ts in enumerate(time_index):
        hour = ts.hour + ts.minute / 60
        
        # Base load
        load_factor = 1.0
        
        # Morning peak (6-9 AM)
        if 6 <= hour < 9:
            peak_factor = np.exp(-((hour - morning_peak_hour) ** 2) / 2)
            load_factor = 1.0 + 1.5 * peak_factor
        
        # Evening peak (5-10 PM)
        elif 17 <= hour < 22:
            peak_factor = np.exp(-((hour - evening_peak_hour) ** 2) / 4)
            load_factor = 1.0 + 2.0 * peak_factor
        
        # Night time reduction
        elif 0 <= hour < 6 or hour >= 23:
            load_factor = 0.4
        
        # Weekend adjustment
        if ts.weekday() >= 5:
            load_factor *= 1.1  # 10% higher on weekends
        
        load[i] = num_households * base_load_per_hh * load_factor
    
    # Add some random variation
    load += np.random.normal(0, num_households * 0.1, len(load))
    load = np.maximum(load, num_households * base_load_per_hh * 0.3)  # Minimum load
    
    return load


def create_ev_charging_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    num_chargers: int = 10,
    charger_power_kw: float = 50,
    arrival_hour: int = 17,
    departure_hour: int = 7,
    utilization: float = 0.6
) -> np.ndarray:
    """
    Create EV charging station load profile.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        num_chargers: Number of charging points
        charger_power_kw: Power per charger
        arrival_hour: Peak arrival time
        departure_hour: Peak departure time
        utilization: Average utilization factor
        
    Returns:
        Load profile in kW
    """
    time_index = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    load = np.zeros(len(time_index))
    
    for i, ts in enumerate(time_index):
        hour = ts.hour + ts.minute / 60
        is_weekend = ts.weekday() >= 5
        
        # Different patterns for workplace vs public charging
        if 7 <= hour < 19:  # Daytime
            if is_weekend:
                # Weekend public charging
                occupied = utilization * 0.7
            else:
                # Workplace charging
                if 8 <= hour < 17:
                    occupied = utilization
                else:
                    occupied = utilization * 0.3
        else:  # Evening/night
            # Overnight charging
            if 19 <= hour or hour < 6:
                occupied = utilization * 0.8
            else:
                occupied = utilization * 0.2
        
        load[i] = num_chargers * charger_power_kw * occupied
    
    # Add some randomness
    load += np.random.normal(0, num_chargers * charger_power_kw * 0.05, len(load))
    load = np.maximum(load, 0)
    
    return load


def combine_load_profiles(*profiles: np.ndarray) -> np.ndarray:
    """
    Combine multiple load profiles (e.g., building + EV charging).
    
    Args:
        *profiles: Variable number of load profile arrays
        
    Returns:
        Combined load profile
    """
    # Check all profiles have same length
    lengths = [len(p) for p in profiles]
    if len(set(lengths)) > 1:
        raise ValueError(f"All profiles must have same length, got {lengths}")
    
    return sum(profiles)


def scale_load_profile(
    profile: np.ndarray,
    target_daily_kwh: Optional[float] = None,
    target_peak_kw: Optional[float] = None
) -> np.ndarray:
    """
    Scale a load profile to match target energy or peak.
    
    Args:
        profile: Original load profile in kW
        target_daily_kwh: Target daily energy consumption
        target_peak_kw: Target peak power
        
    Returns:
        Scaled load profile
    """
    if target_daily_kwh is not None:
        # Assuming consistent timestep
        current_daily_kwh = np.sum(profile[:96]) * 0.25  # Assuming 15-min data
        scale_factor = target_daily_kwh / current_daily_kwh
        return profile * scale_factor
    
    elif target_peak_kw is not None:
        current_peak = np.max(profile)
        scale_factor = target_peak_kw / current_peak
        return profile * scale_factor
    
    return profile


# Enhanced convenience function for BESS optimization
def get_load_profile_for_optimization(
    profile_type: str,
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    use_standard_profiles: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Get a load profile ready for BESS optimization.
    
    Args:
        profile_type: 'commercial', 'industrial', 'residential', 'ev_charging', 'standard'
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        use_standard_profiles: Use standard profiles when available
        **kwargs: Additional parameters for specific profile type
        
    Returns:
        Load profile in kW
        
    Example:
        >>> load = get_load_profile_for_optimization(
        ...     'commercial',
        ...     datetime(2024, 6, 1),
        ...     datetime(2024, 6, 2),
        ...     peak_load_kw=500
        ... )
    """
    profile_generators = {
        'commercial': create_commercial_load_profile,
        'industrial': create_industrial_load_profile,
        'residential': create_residential_load_profile,
        'ev_charging': create_ev_charging_load_profile
    }
    
    # Special case for direct standard profile
    if profile_type == 'standard':
        if not STANDARD_PROFILES_AVAILABLE:
            raise ValueError("Standard profiles not available. Install standard_load_profiles.py")
        
        peak_power_kw = kwargs.get('peak_power_kw', 500)
        days = (end - start).days + 1
        
        return create_standard_load_profile(
            peak_power_kw=peak_power_kw,
            start_date=start,
            days=days,
            dt_minutes=dt_minutes
        )
    
    if profile_type not in profile_generators:
        raise ValueError(f"Unknown profile type: {profile_type}. "
                        f"Choose from: {list(profile_generators.keys())} or 'standard'")
    
    generator = profile_generators[profile_type]
    
    # Add use_standard_profiles parameter if supported
    if profile_type in ['commercial', 'industrial', 'residential']:
        kwargs['use_standard_profiles'] = use_standard_profiles
    
    return generator(start, end, dt_minutes, **kwargs)


def get_available_standard_profiles() -> Optional[List[str]]:
    """
    Get list of available standard load profile classes.
    
    Returns:
        List of profile class names, or None if standard profiles not available
    """
    if not STANDARD_PROFILES_AVAILABLE:
        return None
    
    try:
        # Try to find the CSV file and load profiles
        csv_path = _find_standard_profiles_csv()
        if csv_path:
            generator = StandardLoadProfileGenerator(csv_path)
            return generator.get_available_profiles()
    except Exception as e:
        logger.warning(f"Could not load standard profiles: {e}")
    
    return None


def _find_standard_profiles_csv() -> Optional[str]:
    """Find the standard load profiles CSV file."""
    possible_paths = [
        'Standard Load Profiles.csv',
        'data/Standard Load Profiles.csv', 
        'data/templates/Standard Load Profiles.csv',
        Path(__file__).parent.parent / 'data' / 'templates' / 'Standard Load Profiles.csv'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return str(path)
    
    return None


# Add to __init__.py exports
__all__ = [
    'create_commercial_load_profile',
    'create_industrial_load_profile', 
    'create_residential_load_profile',
    'create_ev_charging_load_profile',
    'combine_load_profiles',
    'scale_load_profile',
    'get_load_profile_for_optimization',
    'get_available_standard_profiles'
]

if STANDARD_PROFILES_AVAILABLE:
    __all__.extend(['create_standard_load_profile', 'StandardLoadProfileGenerator'])
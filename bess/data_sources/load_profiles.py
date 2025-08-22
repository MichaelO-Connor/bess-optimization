# bess/data_sources/load_profiles.py
"""
Helper functions for creating site-specific load profiles.
Since ENTSO-E doesn't provide site load, these utilities help create realistic profiles.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
from datetime import datetime, timedelta


def create_commercial_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    base_load_kw: float = 100,
    peak_load_kw: float = 500,
    business_hours: tuple = (7, 19),
    weekend_factor: float = 0.3
) -> np.ndarray:
    """
    Create a commercial building load profile.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        base_load_kw: Nighttime/weekend base load
        peak_load_kw: Peak daytime load
        business_hours: (start_hour, end_hour) for business operations
        weekend_factor: Weekend load as fraction of weekday
        
    Returns:
        Load profile in kW
    """
    # Create time index
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


def create_industrial_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    base_load_kw: float = 500,
    production_load_kw: float = 2000,
    shifts: list = [(6, 14), (14, 22)],  # 2-shift operation
    weekend_production: bool = False
) -> np.ndarray:
    """
    Create an industrial facility load profile with shifts.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        base_load_kw: Continuous process base load
        production_load_kw: Additional load during production
        shifts: List of (start_hour, end_hour) tuples for shifts
        weekend_production: Whether production runs on weekends
        
    Returns:
        Load profile in kW
    """
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


def create_residential_load_profile(
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    num_households: int = 100,
    avg_daily_kwh: float = 10,
    morning_peak_hour: int = 7,
    evening_peak_hour: int = 19
) -> np.ndarray:
    """
    Create aggregated residential load profile.
    
    Args:
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        num_households: Number of households
        avg_daily_kwh: Average daily consumption per household
        morning_peak_hour: Hour of morning peak
        evening_peak_hour: Hour of evening peak
        
    Returns:
        Load profile in kW
    """
    time_index = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    load = np.zeros(len(time_index))
    
    # Base load per household (kW)
    base_load_per_hh = avg_daily_kwh / 24 * 0.5  # 50% is base
    peak_load_per_hh = avg_daily_kwh / 24 * 2.5  # 250% during peaks
    
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


# Convenience function for BESS optimization
def get_load_profile_for_optimization(
    profile_type: str,
    start: datetime,
    end: datetime,
    dt_minutes: int = 15,
    **kwargs
) -> np.ndarray:
    """
    Get a load profile ready for BESS optimization.
    
    Args:
        profile_type: 'commercial', 'industrial', 'residential', 'ev_charging'
        start: Start datetime
        end: End datetime
        dt_minutes: Timestep in minutes
        **kwargs: Additional parameters for specific profile type
        
    Returns:
        Load profile in kW
        
    Example:
        >>> load = get_load_profile_for_optimization(
        ...     'commercial',
        ...     datetime(2024, 6, 1),
        ...     datetime(2024, 6, 2),
        ...     base_load_kw=100,
        ...     peak_load_kw=500
        ... )
    """
    profile_generators = {
        'commercial': create_commercial_load_profile,
        'industrial': create_industrial_load_profile,
        'residential': create_residential_load_profile,
        'ev_charging': create_ev_charging_load_profile
    }
    
    if profile_type not in profile_generators:
        raise ValueError(f"Unknown profile type: {profile_type}. "
                        f"Choose from: {list(profile_generators.keys())}")
    
    generator = profile_generators[profile_type]
    return generator(start, end, dt_minutes, **kwargs)
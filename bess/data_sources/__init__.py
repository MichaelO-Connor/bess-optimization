# bess/data_sources/__init__.py
"""
Data source loaders and integrations.
"""

from .csv_loader import CSVDataLoader, TimeSeriesData, validate_timeseries, create_sample_csv
from .load_profiles import (
    create_commercial_load_profile,
    create_industrial_load_profile, 
    create_residential_load_profile,
    create_ev_charging_load_profile,
    combine_load_profiles,
    scale_load_profile,
    get_load_profile_for_optimization,
    get_available_standard_profiles
)

# Try to import standard load profiles
try:
    from .standard_load_profiles import (
        create_standard_load_profile, 
        StandardLoadProfileGenerator,
        validate_standard_load_csv,
        create_bess_load_profile_from_standard
    )
    STANDARD_PROFILES_AVAILABLE = True
    
    __all__ = [
        "CSVDataLoader",
        "TimeSeriesData", 
        "validate_timeseries",
        "create_sample_csv",
        "create_commercial_load_profile",
        "create_industrial_load_profile",
        "create_residential_load_profile",
        "create_ev_charging_load_profile",
        "combine_load_profiles",
        "scale_load_profile", 
        "get_load_profile_for_optimization",
        "get_available_standard_profiles",
        "create_standard_load_profile",
        "StandardLoadProfileGenerator",
        "validate_standard_load_csv",
        "create_bess_load_profile_from_standard",
    ]
    
except ImportError:
    STANDARD_PROFILES_AVAILABLE = False
    
    __all__ = [
        "CSVDataLoader",
        "TimeSeriesData", 
        "validate_timeseries",
        "create_sample_csv",
        "create_commercial_load_profile",
        "create_industrial_load_profile",
        "create_residential_load_profile",
        "create_ev_charging_load_profile", 
        "combine_load_profiles",
        "scale_load_profile",
        "get_load_profile_for_optimization",
        "get_available_standard_profiles",
    ]

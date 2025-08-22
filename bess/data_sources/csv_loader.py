"""
CSV Data Loader for BESS Optimization
Handles loading and validation of time series data from CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Container for validated time series data."""
    
    # Core time series
    timestamps: pd.DatetimeIndex
    site_load_kw: np.ndarray
    retail_price: np.ndarray
    wholesale_price: np.ndarray
    
    # Optional time series
    pv_generation_kw: Optional[np.ndarray] = None
    capacity_price: Optional[np.ndarray] = None
    capacity_hours: Optional[np.ndarray] = None
    
    # Ancillary services (dict of service_name -> prices)
    ancillary_prices: Dict[str, np.ndarray] = None
    ancillary_activation_up: Dict[str, np.ndarray] = None
    ancillary_activation_down: Dict[str, np.ndarray] = None
    
    # Metadata
    dt_minutes: int = 60
    timezone: str = "UTC"
    
    def __post_init__(self):
        """Validate data consistency after initialization."""
        self._validate_lengths()
        self._validate_values()
    
    def _validate_lengths(self):
        """Ensure all arrays have consistent length."""
        n = len(self.timestamps)
        
        # Check required fields
        for name, arr in [
            ("site_load_kw", self.site_load_kw),
            ("retail_price", self.retail_price),
            ("wholesale_price", self.wholesale_price),
        ]:
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != timestamps length {n}")
        
        # Check optional fields
        if self.pv_generation_kw is not None and len(self.pv_generation_kw) != n:
            raise ValueError(f"PV generation length mismatch")
        
        if self.capacity_price is not None and len(self.capacity_price) != n:
            raise ValueError(f"Capacity price length mismatch")
    
    def _validate_values(self):
        """Validate data ranges and values."""
        # Check for negative loads (usually an error)
        if np.any(self.site_load_kw < 0):
            logger.warning("Negative site load detected - treating as export")
        
        # Check for negative PV (should not happen)
        if self.pv_generation_kw is not None and np.any(self.pv_generation_kw < 0):
            raise ValueError("PV generation cannot be negative")
        
        # Check for NaN values
        for name, arr in [
            ("site_load_kw", self.site_load_kw),
            ("retail_price", self.retail_price),
            ("wholesale_price", self.wholesale_price),
        ]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"{name} contains NaN values")
    
    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the data."""
        return len(self.timestamps)
    
    @property
    def duration_hours(self) -> float:
        """Total duration in hours."""
        return self.n_timesteps * self.dt_minutes / 60.0
    
    def to_kwh(self) -> 'TimeSeriesData':
        """Convert power (kW) to energy (kWh) for the timestep."""
        dt_hours = self.dt_minutes / 60.0
        
        # Create a copy with energy values
        return TimeSeriesData(
            timestamps=self.timestamps,
            site_load_kw=self.site_load_kw * dt_hours,  # Now kWh
            retail_price=self.retail_price,  # Already €/kWh
            wholesale_price=self.wholesale_price,  # Already €/kWh
            pv_generation_kw=self.pv_generation_kw * dt_hours if self.pv_generation_kw is not None else None,
            capacity_price=self.capacity_price,
            capacity_hours=self.capacity_hours,
            ancillary_prices=self.ancillary_prices,
            ancillary_activation_up=self.ancillary_activation_up,
            ancillary_activation_down=self.ancillary_activation_down,
            dt_minutes=self.dt_minutes,
            timezone=self.timezone
        )


class CSVDataLoader:
    """Load and process time series data from CSV files."""
    
    # Expected column mappings
    COLUMN_MAPPINGS = {
        # Required columns
        'timestamp': ['timestamp', 'datetime', 'time', 'date'],
        'site_load': ['site_load_kw', 'load_kw', 'load', 'demand_kw', 'demand'],
        'retail_price': ['retail_price', 'retail_price_eur_kwh', 'retail', 'import_price'],
        'wholesale_price': ['wholesale_price', 'wholesale_price_eur_kwh', 'wholesale', 'spot_price'],
        
        # Optional columns
        'pv_generation': ['pv_kw', 'pv_generation_kw', 'solar_kw', 'pv', 'solar'],
        'capacity_price': ['capacity_price', 'capacity_price_eur_kw', 'capacity'],
        'capacity_hours': ['capacity_hours', 'capacity_mask', 'peak_hours'],
        
        # Ancillary services (examples)
        'fcr_price': ['fcr_price', 'fcr_eur_mw', 'fcr'],
        'afrr_up_price': ['afrr_up_price', 'afrr_up', 'reg_up_price'],
        'afrr_down_price': ['afrr_down_price', 'afrr_down', 'reg_down_price'],
    }
    
    @classmethod
    def load_from_file(
        cls,
        filepath: Union[str, Path],
        dt_minutes: Optional[int] = None,
        timezone: str = "UTC",
        date_format: Optional[str] = None,
        **kwargs
    ) -> TimeSeriesData:
        """
        Load time series data from a single CSV file.
        
        Args:
            filepath: Path to CSV file
            dt_minutes: Timestep in minutes (auto-detected if None)
            timezone: Timezone for timestamps
            date_format: Date parsing format
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            TimeSeriesData object with validated data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        logger.info(f"Loading time series from {filepath}")
        
        # Read CSV
        df = pd.read_csv(filepath, **kwargs)
        
        # Find and parse timestamp column
        timestamp_col = cls._find_column(df, 'timestamp')
        if timestamp_col is None:
            raise ValueError("No timestamp column found in CSV")
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=date_format)
        df = df.set_index(timestamp_col).sort_index()
        
        # Detect timestep if not provided
        if dt_minutes is None:
            dt_minutes = cls._detect_timestep(df.index)
            logger.info(f"Detected timestep: {dt_minutes} minutes")
        
        # Extract required columns
        site_load = cls._extract_column(df, 'site_load', required=True)
        retail_price = cls._extract_column(df, 'retail_price', required=True)
        wholesale_price = cls._extract_column(df, 'wholesale_price', required=True)
        
        # Extract optional columns
        pv_generation = cls._extract_column(df, 'pv_generation', required=False)
        capacity_price = cls._extract_column(df, 'capacity_price', required=False)
        capacity_hours = cls._extract_column(df, 'capacity_hours', required=False)
        
        # Extract ancillary service data
        ancillary_prices = {}
        ancillary_activation_up = {}
        ancillary_activation_down = {}
        
        # Look for ancillary service columns
        for col in df.columns:
            col_lower = col.lower()
            
            # FCR, aFRR, mFRR prices
            if 'fcr' in col_lower and 'price' in col_lower:
                ancillary_prices['fcr'] = df[col].values
            elif 'afrr_up' in col_lower and 'price' in col_lower:
                ancillary_prices['afrr_up'] = df[col].values
            elif 'afrr_down' in col_lower and 'price' in col_lower:
                ancillary_prices['afrr_down'] = df[col].values
            
            # Activation rates
            elif 'activation_up' in col_lower:
                service = col.split('_')[0].lower()
                ancillary_activation_up[service] = df[col].values
            elif 'activation_down' in col_lower:
                service = col.split('_')[0].lower()
                ancillary_activation_down[service] = df[col].values
        
        # Create TimeSeriesData object
        return TimeSeriesData(
            timestamps=df.index,
            site_load_kw=site_load,
            retail_price=retail_price,
            wholesale_price=wholesale_price,
            pv_generation_kw=pv_generation,
            capacity_price=capacity_price,
            capacity_hours=capacity_hours,
            ancillary_prices=ancillary_prices if ancillary_prices else None,
            ancillary_activation_up=ancillary_activation_up if ancillary_activation_up else None,
            ancillary_activation_down=ancillary_activation_down if ancillary_activation_down else None,
            dt_minutes=dt_minutes,
            timezone=timezone
        )
    
    @classmethod
    def load_from_multiple_files(
        cls,
        files_dict: Dict[str, Union[str, Path]],
        dt_minutes: Optional[int] = None,
        timezone: str = "UTC",
        **kwargs
    ) -> TimeSeriesData:
        """
        Load time series data from multiple CSV files.
        
        Args:
            files_dict: Dictionary mapping data type to file path
                       e.g., {'load': 'load.csv', 'prices': 'prices.csv'}
            dt_minutes: Timestep in minutes
            timezone: Timezone for timestamps
            **kwargs: Additional pandas.read_csv arguments
            
        Returns:
            TimeSeriesData object with combined data
        """
        data_frames = {}
        
        # Load each file
        for data_type, filepath in files_dict.items():
            filepath = Path(filepath)
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            df = pd.read_csv(filepath, **kwargs)
            
            # Parse timestamps
            timestamp_col = cls._find_column(df, 'timestamp')
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.set_index(timestamp_col).sort_index()
            
            data_frames[data_type] = df
        
        # Combine all dataframes
        if not data_frames:
            raise ValueError("No valid data files found")
        
        # Merge on timestamp index
        combined_df = pd.concat(data_frames.values(), axis=1, join='inner')
        
        # Detect timestep
        if dt_minutes is None:
            dt_minutes = cls._detect_timestep(combined_df.index)
        
        # Extract data similar to single file
        return cls._extract_from_dataframe(combined_df, dt_minutes, timezone)
    
    @classmethod
    def _find_column(cls, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """Find column by type using mappings."""
        if column_type not in cls.COLUMN_MAPPINGS:
            return None
        
        possible_names = cls.COLUMN_MAPPINGS[column_type]
        
        for col in df.columns:
            col_lower = col.lower()
            for name in possible_names:
                if name in col_lower:
                    return col
        
        return None
    
    @classmethod
    def _extract_column(
        cls,
        df: pd.DataFrame,
        column_type: str,
        required: bool = True
    ) -> Optional[np.ndarray]:
        """Extract column data by type."""
        col = cls._find_column(df, column_type)
        
        if col is None:
            if required:
                raise ValueError(f"Required column type '{column_type}' not found")
            return None
        
        return df[col].values
    
    @classmethod
    def _detect_timestep(cls, timestamps: pd.DatetimeIndex) -> int:
        """Auto-detect timestep from timestamps."""
        if len(timestamps) < 2:
            return 60  # Default to hourly
        
        # Calculate differences
        diffs = timestamps[1:] - timestamps[:-1]
        
        # Get most common difference
        mode_diff = pd.Series(diffs).mode()[0]
        
        # Convert to minutes
        return int(mode_diff.total_seconds() / 60)
    
    @classmethod
    def _extract_from_dataframe(
        cls,
        df: pd.DataFrame,
        dt_minutes: int,
        timezone: str
    ) -> TimeSeriesData:
        """Extract TimeSeriesData from a combined DataFrame."""
        # Similar to load_from_file but works with pre-combined data
        site_load = cls._extract_column(df, 'site_load', required=True)
        retail_price = cls._extract_column(df, 'retail_price', required=True)
        wholesale_price = cls._extract_column(df, 'wholesale_price', required=True)
        
        # Optional columns
        pv_generation = cls._extract_column(df, 'pv_generation', required=False)
        capacity_price = cls._extract_column(df, 'capacity_price', required=False)
        capacity_hours = cls._extract_column(df, 'capacity_hours', required=False)
        
        return TimeSeriesData(
            timestamps=df.index,
            site_load_kw=site_load,
            retail_price=retail_price,
            wholesale_price=wholesale_price,
            pv_generation_kw=pv_generation,
            capacity_price=capacity_price,
            capacity_hours=capacity_hours,
            dt_minutes=dt_minutes,
            timezone=timezone
        )


def validate_timeseries(
    data: TimeSeriesData,
    min_timesteps: int = 24,
    max_price: float = 10.0,  # €/kWh
    max_load: float = 1e6,     # kW
) -> List[str]:
    """
    Validate time series data for common issues.
    
    Args:
        data: TimeSeriesData to validate
        min_timesteps: Minimum required timesteps
        max_price: Maximum reasonable price (€/kWh)
        max_load: Maximum reasonable load (kW)
        
    Returns:
        List of validation warnings (empty if all OK)
    """
    warnings = []
    
    # Check length
    if data.n_timesteps < min_timesteps:
        warnings.append(f"Only {data.n_timesteps} timesteps, minimum {min_timesteps} recommended")
    
    # Check prices
    if np.any(data.retail_price < 0):
        warnings.append("Negative retail prices detected")
    if np.any(data.retail_price > max_price):
        warnings.append(f"Retail prices exceed {max_price} €/kWh")
    
    if np.any(data.wholesale_price < -max_price):
        warnings.append(f"Wholesale prices below -{max_price} €/kWh (extreme negative pricing)")
    if np.any(data.wholesale_price > max_price):
        warnings.append(f"Wholesale prices exceed {max_price} €/kWh")
    
    # Check load
    if np.any(np.abs(data.site_load_kw) > max_load):
        warnings.append(f"Site load exceeds {max_load} kW")
    
    # Check PV
    if data.pv_generation_kw is not None:
        if np.any(data.pv_generation_kw < 0):
            warnings.append("Negative PV generation detected")
        if np.any(data.pv_generation_kw > max_load):
            warnings.append(f"PV generation exceeds {max_load} kW")
    
    # Check timestamp spacing
    diffs = data.timestamps[1:] - data.timestamps[:-1]
    expected_diff = pd.Timedelta(minutes=data.dt_minutes)
    
    irregular = np.any(diffs != expected_diff)
    if irregular:
        warnings.append("Irregular timestamp spacing detected")
    
    # Check for data gaps
    if np.any(diffs > expected_diff * 2):
        warnings.append("Data gaps detected in time series")
    
    return warnings


def create_sample_csv(
    output_path: Union[str, Path],
    days: int = 7,
    dt_minutes: int = 60,
    include_pv: bool = True,
    include_ancillary: bool = True
):
    """
    Create a sample CSV file with realistic data for testing.
    
    Args:
        output_path: Where to save the CSV
        days: Number of days of data
        dt_minutes: Timestep in minutes
        include_pv: Include PV generation
        include_ancillary: Include ancillary service prices
    """
    output_path = Path(output_path)
    
    # Create timestamps
    start = pd.Timestamp('2024-01-01', tz='UTC')
    end = start + pd.Timedelta(days=days)
    timestamps = pd.date_range(start, end, freq=f'{dt_minutes}min', inclusive='left')
    n = len(timestamps)
    
    # Create data
    data = {
        'timestamp': timestamps,
        'site_load_kw': 100 + 50 * np.sin(np.arange(n) * 2 * np.pi / (24 * 60/dt_minutes)),
        'retail_price': 0.15 + 0.05 * np.sin(np.arange(n) * 2 * np.pi / (24 * 60/dt_minutes)),
        'wholesale_price': 0.10 + 0.08 * np.sin(np.arange(n) * 2 * np.pi / (24 * 60/dt_minutes) + 1),
    }
    
    if include_pv:
        # Simple PV profile (daylight hours only)
        pv = np.zeros(n)
        for i in range(n):
            hour = timestamps[i].hour
            if 6 <= hour <= 18:
                pv[i] = 50 * np.sin((hour - 6) * np.pi / 12)
        data['pv_kw'] = pv
    
    if include_ancillary:
        data['fcr_price'] = np.full(n, 15.0)  # €/MW/h
        data['afrr_up_price'] = np.full(n, 10.0)
        data['afrr_down_price'] = np.full(n, 8.0)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample CSV created: {output_path}")
    return output_path
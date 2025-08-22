# bess/data_sources/standard_load_profiles.py
"""
Standard Load Profile Generator

Generates realistic site load profiles using standard load profiles.
Takes only peak power as input and scales the standard profile accordingly.
Supports simplified CSV format with single column of percentage values.

CSV Format Expected:
- Single column with header "% of Peak Power" or similar
- Values as percentages (0-100) representing fraction of peak load
- Half-hourly resolution (48 values per day, ~17,520 per year)

Usage:
    from bess.data_sources.standard_load_profiles import create_standard_load_profile
    
    load_profile = create_standard_load_profile(
        peak_power_kw=500,
        start_date='2024-01-01',
        days=7,
        profile_class='Commercial'
    )
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, List, Dict
import logging

logger = logging.getLogger(__name__)


class StandardLoadProfileGenerator:
    """
    Generator for site load profiles using standard load profile data.
    
    Handles simplified CSV format with single column of percentage values
    representing load as a fraction of peak power at half-hourly resolution.
    """
    
    def __init__(self, csv_path: Union[str, Path]):
        """
        Initialize with standard load profiles CSV.
        
        Args:
            csv_path: Path to CSV file with load profile percentages
        """
        self.csv_path = Path(csv_path)
        self.profiles_df = None
        self.profile_data = None
        self.profile_classes = []
        self._load_profiles()
    
    def _load_profiles(self):
        """Load and parse the standard load profiles CSV."""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'iso-8859-1', 'cp1252']:
                try:
                    self.profiles_df = pd.read_csv(self.csv_path, encoding=encoding)
                    logger.info(f"Successfully loaded profiles with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.profiles_df is None:
                raise ValueError("Could not read CSV file with any common encoding")
            
            # Inspect the structure
            logger.info(f"CSV shape: {self.profiles_df.shape}")
            logger.info(f"Columns: {list(self.profiles_df.columns)}")
            
            # Process the simplified CSV format
            self._process_simple_format()
            
        except Exception as e:
            logger.error(f"Failed to load standard load profiles: {e}")
            raise
    
    def _process_simple_format(self):
        """Process simplified CSV format with single percentage column."""
        df = self.profiles_df
        
        # Find the percentage column (should be the only data column)
        percentage_col = None
        
        # Look for column with percentage-related keywords
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['peak', 'power', '%', 'percent', 'profile']):
                # Check if it contains numeric data
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    percentage_col = col
                    break
                except:
                    continue
        
        # If no keyword match, use the first numeric column
        if percentage_col is None:
            for col in df.columns:
                try:
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_data.isna().all():
                        percentage_col = col
                        break
                except:
                    continue
        
        if percentage_col is None:
            raise ValueError("Could not find percentage data column in CSV")
        
        # Extract the percentage values
        percentage_values = pd.to_numeric(df[percentage_col], errors='coerce').dropna()
        
        logger.info(f"Found {len(percentage_values)} percentage values")
        logger.info(f"Range: {percentage_values.min():.1f}% to {percentage_values.max():.1f}%")
        
        # Validate data range
        if percentage_values.min() < 0 or percentage_values.max() > 200:
            logger.warning(f"Unusual percentage range: {percentage_values.min():.1f}% to {percentage_values.max():.1f}%")
        
        # Create datetime index assuming half-hourly data starting Jan 1
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        self.datetime_index = pd.date_range(
            start=start_date, 
            periods=len(percentage_values), 
            freq='30min'
        )
        
        # Store the profile data
        self.profile_data = percentage_values.values
        
        # For compatibility, create a single "default" profile class
        self.profile_classes = ['Standard']
        
        logger.info(f"Processed {len(self.profile_data)} half-hourly values")
        logger.info(f"Detected duration: {len(self.profile_data) / 48:.1f} days")
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available load profile classes."""
        return self.profile_classes
    
    def generate_load_profile(self,
                            peak_power_kw: float,
                            start_date: Union[str, datetime],
                            days: int = 7,
                            profile_class: Optional[str] = None,
                            dt_minutes: int = 30) -> np.ndarray:
        """
        Generate load profile scaled to peak power.
        
        Args:
            peak_power_kw: Maximum site load in kW
            start_date: Start date for the profile
            days: Number of days to generate
            profile_class: Which profile class to use (ignored for single profile)
            dt_minutes: Output timestep in minutes (30 for half-hourly, 60 for hourly)
            
        Returns:
            Load profile in kW for each timestep
        """
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        # Extract the relevant time period
        profile_percentages = self._extract_time_period(start_date, days)
        
        # Resample to target resolution if needed
        if dt_minutes != 30:
            profile_percentages = self._resample_profile(
                profile_percentages, start_date, dt_minutes
            )
        
        # Scale by peak power
        # Profile data is in percentages (0-100), convert to kW
        load_profile_kw = profile_percentages * (peak_power_kw / 100.0)
        
        logger.info(f"Generated {len(load_profile_kw)} timesteps of load profile")
        logger.info(f"Peak: {load_profile_kw.max():.1f} kW, "
                   f"Min: {load_profile_kw.min():.1f} kW, "
                   f"Mean: {load_profile_kw.mean():.1f} kW")
        
        return load_profile_kw
    
    def _extract_time_period(self, start_date: pd.Timestamp, days: int) -> np.ndarray:
        """Extract profile data for the specified time period."""
        total_periods = len(self.profile_data)
        
        # Handle year wrapping - map requested dates to available data
        if total_periods <= 17568:  # One year of half-hourly data (366 * 48)
            # Map start_date to the same day-of-year in the profile data
            start_day_of_year = start_date.dayofyear
            
            # Calculate start index (48 half-hours per day)
            start_idx = (start_day_of_year - 1) * 48
            end_idx = start_idx + (days * 48)
            
            # Handle year boundary crossing
            if end_idx > total_periods:
                # Split into two parts
                part1 = self.profile_data[start_idx:total_periods]
                remaining_periods = end_idx - total_periods
                part2 = self.profile_data[:remaining_periods]
                profile_data = np.concatenate([part1, part2])
            else:
                profile_data = self.profile_data[start_idx:end_idx]
        else:
            # Multiple years of data - use directly based on date
            # This is more complex and would need actual date matching
            # For now, use a simple approach
            start_idx = 0
            end_idx = min(days * 48, total_periods)
            profile_data = self.profile_data[start_idx:end_idx]
            
            logger.warning("Multiple years of data detected - using simple extraction")
        
        # Handle missing data
        if len(profile_data) == 0:
            logger.warning("No data found for specified period, using first week")
            profile_data = self.profile_data[:min(days * 48, total_periods)]
        
        # Ensure we have the right number of periods
        target_periods = days * 48  # Half-hourly
        if len(profile_data) < target_periods:
            # Repeat the data to fill the required period
            repeats = int(np.ceil(target_periods / len(profile_data)))
            profile_data = np.tile(profile_data, repeats)[:target_periods]
            logger.warning(f"Repeated profile data to fill {days} days")
        elif len(profile_data) > target_periods:
            # Truncate to exact length
            profile_data = profile_data[:target_periods]
        
        return profile_data
    
    def _resample_profile(self, profile_data: np.ndarray, 
                         start_date: pd.Timestamp, dt_minutes: int) -> np.ndarray:
        """Resample profile to target resolution."""
        if dt_minutes == 30:
            return profile_data
        
        # Create time series for resampling
        current_index = pd.date_range(
            start=start_date,
            periods=len(profile_data),
            freq='30min'
        )
        
        # Create target index
        target_index = pd.date_range(
            start=start_date,
            end=start_date + pd.Timedelta(days=len(profile_data) / 48),
            freq=f'{dt_minutes}min'
        )[:-1]  # Remove last point to avoid boundary issues
        
        # Create series and resample
        series = pd.Series(profile_data, index=current_index)
        
        if dt_minutes > 30:
            # Downsample - take mean
            resampled = series.resample(f'{dt_minutes}min').mean()
        else:
            # Upsample - interpolate
            series_extended = series.reindex(current_index.union(target_index))
            resampled = series_extended.interpolate('linear').reindex(target_index)
        
        return resampled.values
    
    def plot_profile(self, load_profile_kw: np.ndarray, 
                    title: str = "Generated Load Profile",
                    dt_minutes: int = 30):
        """Plot the generated load profile."""
        try:
            import matplotlib.pyplot as plt
            
            hours = np.arange(len(load_profile_kw)) * (dt_minutes / 60)
            
            plt.figure(figsize=(12, 6))
            plt.plot(hours, load_profile_kw)
            plt.title(title)
            plt.xlabel('Hours')
            plt.ylabel('Load (kW)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def create_standard_load_profile(peak_power_kw: float,
                          start_date: Union[str, datetime],
                          days: int = 7,
                          profile_type: str = 'auto',
                          dt_minutes: int = 30,
                          csv_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to generate standard load profile.
    
    Args:
        peak_power_kw: Maximum site load in kW
        start_date: Start date for the profile
        days: Number of days to generate
        profile_type: Profile type (ignored for single-profile CSV)
        dt_minutes: Timestep in minutes (30 for half-hourly, 60 for hourly)
        csv_path: Path to standard load profiles CSV (auto-detect if None)
        
    Returns:
        Load profile in kW
        
    Example:
        >>> load = create_standard_load_profile(
        ...     peak_power_kw=500,
        ...     start_date='2024-06-01',
        ...     days=7,
        ...     dt_minutes=30
        ... )
    """
    # Auto-detect CSV path if not provided
    if csv_path is None:
        # Look in several possible locations
        possible_paths = [
            'Standard Load Profiles.csv',
            'data/Standard Load Profiles.csv',
            'data/templates/Standard Load Profiles.csv',
            Path(__file__).parent.parent / 'data' / 'templates' / 'Standard Load Profiles.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                csv_path = str(path)
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                "Could not find 'Standard Load Profiles.csv'. "
                "Please specify csv_path or place the file in the current directory."
            )
    
    generator = StandardLoadProfileGenerator(csv_path)
    
    return generator.generate_load_profile(
        peak_power_kw=peak_power_kw,
        start_date=start_date,
        days=days,
        profile_class=None,  # Ignored for single profile
        dt_minutes=dt_minutes
    )


# Integration with BESS framework
def create_bess_load_profile_from_standard(peak_power_kw: float,
                                            start: datetime,
                                            end: datetime,
                                            dt_minutes: int = 30,
                                            csv_path: Optional[str] = None) -> np.ndarray:
    """
    Create load profile compatible with BESS optimization framework.
    
    Args:
        peak_power_kw: Maximum site load in kW
        start: Start datetime
        end: End datetime  
        dt_minutes: Timestep in minutes
        csv_path: Path to standard profiles CSV
        
    Returns:
        Load profile in kW (compatible with BESS framework)
    """
    days = (end - start).days + 1
    
    load_profile = create_standard_load_profile(
        peak_power_kw=peak_power_kw,
        start_date=start,
        days=days,
        dt_minutes=dt_minutes,
        csv_path=csv_path
    )
    
    return load_profile


def validate_standard_load_csv(csv_path: Union[str, Path]) -> Dict[str, any]:
    """
    Validate a standard load profile CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary with validation results
    """
    csv_path = Path(csv_path)
    
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        # Try to load the CSV
        generator = StandardLoadProfileGenerator(csv_path)
        
        # Check data quality
        data = generator.profile_data
        
        result['info'] = {
            'file_size_mb': csv_path.stat().st_size / (1024 * 1024),
            'num_values': len(data),
            'estimated_days': len(data) / 48,
            'min_percentage': data.min(),
            'max_percentage': data.max(),
            'mean_percentage': data.mean()
        }
        
        # Validation checks
        if len(data) < 48:
            result['errors'].append("Less than 1 day of data (48 half-hourly values)")
        
        if data.min() < 0:
            result['errors'].append("Negative percentage values found")
        
        if data.max() > 150:
            result['warnings'].append(f"Very high percentage values (max: {data.max():.1f}%)")
        
        if data.max() < 50:
            result['warnings'].append(f"Low maximum percentage (max: {data.max():.1f}%)")
        
        # Check for reasonable variation
        if data.std() < 5:
            result['warnings'].append("Low variation in load profile (std < 5%)")
        
        if len(result['errors']) == 0:
            result['valid'] = True
            
    except Exception as e:
        result['errors'].append(f"Failed to load CSV: {str(e)}")
    
    return result
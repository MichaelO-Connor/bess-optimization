#!/usr/bin/env python3
"""
Example: BESS Optimization with CSV Data Input
Shows how to load time series data from CSV files and run optimization.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from bess import BESSOptimizer, DataBundle, TimeGrid, Architecture
from bess.schema import (
    Ratings, Efficiencies, Bounds, Degradation,
    Tariffs, Exogenous, Capacity, AncillaryService
)
from bess.data_sources.csv_loader import CSVDataLoader, create_sample_csv, validate_timeseries
from bess.utils.validators import validate_data_bundle, generate_validation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_single_csv():
    """Example 1: Load all data from a single CSV file."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single CSV File")
    print("="*60)
    
    # Create sample CSV if it doesn't exist
    csv_path = Path('data/sample/sample_data.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print("Creating sample CSV file...")
        create_sample_csv(
            csv_path,
            days=7,
            dt_minutes=60,
            include_pv=True,
            include_ancillary=True
        )
    
    # Load data from CSV
    print(f"Loading data from {csv_path}...")
    timeseries = CSVDataLoader.load_from_file(csv_path)
    
    # Validate the data
    warnings = validate_timeseries(timeseries)
    if warnings:
        print("Data validation warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    else:
        print("✓ Data validation passed")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"  Timesteps: {timeseries.n_timesteps}")
    print(f"  Duration: {timeseries.duration_hours:.1f} hours")
    print(f"  Load range: {timeseries.site_load_kw.min():.1f} - {timeseries.site_load_kw.max():.1f} kW")
    print(f"  Price range: €{timeseries.wholesale_price.min():.3f} - €{timeseries.wholesale_price.max():.3f}/kWh")
    
    if timeseries.pv_generation_kw is not None:
        print(f"  PV peak: {timeseries.pv_generation_kw.max():.1f} kW")
    
    # Create DataBundle from CSV data
    bundle = create_bundle_from_csv(timeseries)
    
    # Run optimization
    print("\nRunning optimization...")
    optimizer = BESSOptimizer(bundle)
    results = optimizer.optimize(time_limit=30)
    
    # Display results
    print(f"\n✓ Optimization complete!")
    print(f"  Status: {results.status}")
    print(f"  Total revenue: €{results.total_revenue:.2f}")
    print(f"  Energy cost: €{results.energy_cost:.2f}")
    print(f"  Ancillary revenue: €{results.ancillary_revenue:.2f}")
    
    return results


def example_multiple_csvs():
    """Example 2: Load data from multiple CSV files."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple CSV Files")
    print("="*60)
    
    # Create sample CSV files
    data_dir = Path('data/sample')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate files for different data types
    files = {
        'load': data_dir / 'site_load.csv',
        'prices': data_dir / 'prices.csv',
        'pv': data_dir / 'pv_generation.csv',
    }
    
    # Generate sample data
    print("Creating sample CSV files...")
    timestamps = pd.date_range('2024-01-01', periods=168, freq='1h')  # 1 week
    
    # Site load
    if not files['load'].exists():
        load_data = pd.DataFrame({
            'timestamp': timestamps,
            'site_load_kw': 100 + 50 * np.sin(np.arange(168) * 2 * np.pi / 24)
        })
        load_data.to_csv(files['load'], index=False)
    
    # Prices
    if not files['prices'].exists():
        price_data = pd.DataFrame({
            'timestamp': timestamps,
            'retail_price': 0.15 + 0.05 * np.sin(np.arange(168) * 2 * np.pi / 24),
            'wholesale_price': 0.10 + 0.08 * np.sin(np.arange(168) * 2 * np.pi / 24 + 1),
            'fcr_price': np.full(168, 15.0),
            'afrr_up_price': np.full(168, 10.0)
        })
        price_data.to_csv(files['prices'], index=False)
    
    # PV generation
    if not files['pv'].exists():
        pv = np.zeros(168)
        for i in range(168):
            hour = i % 24
            if 6 <= hour <= 18:
                pv[i] = 200 * np.sin((hour - 6) * np.pi / 12)
        
        pv_data = pd.DataFrame({
            'timestamp': timestamps,
            'pv_kw': pv
        })
        pv_data.to_csv(files['pv'], index=False)
    
    # Load from multiple files
    print(f"Loading data from {len(files)} files...")
    timeseries = CSVDataLoader.load_from_multiple_files(files)
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Files: {', '.join(f.name for f in files.values())}")
    print(f"  Timesteps: {timeseries.n_timesteps}")
    
    # Create bundle and optimize
    bundle = create_bundle_from_csv(timeseries, include_ancillary=True)
    
    print("\nRunning optimization...")
    optimizer = BESSOptimizer(bundle)
    results = optimizer.optimize(time_limit=30)
    
    print(f"\n✓ Optimization complete!")
    kpis = optimizer.get_kpis()
    
    print("\nKey Performance Indicators:")
    for key, value in kpis.items():
        if 'revenue' in key or 'cost' in key:
            print(f"  {key}: €{value:.2f}")
        else:
            print(f"  {key}: {value:.3f}")
    
    return results


def example_with_validation():
    """Example 3: Full validation workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Validation Workflow")
    print("="*60)
    
    # Load data
    csv_path = Path('data/sample/sample_data.csv')
    if not csv_path.exists():
        create_sample_csv(csv_path, days=2)
    
    timeseries = CSVDataLoader.load_from_file(csv_path)
    
    # Create bundle
    bundle = create_bundle_from_csv(timeseries)
    
    # Generate validation report
    print("Generating validation report...")
    report_path = Path('data/validation_report.txt')
    report = generate_validation_report(bundle, report_path)
    
    print(f"\nValidation report saved to {report_path}")
    print("\nReport Preview:")
    print("-" * 40)
    print('\n'.join(report.split('\n')[:20]))  # Show first 20 lines
    print("...")
    
    # Validate with strict mode
    try:
        is_valid = validate_data_bundle(bundle, strict=True)
        print(f"\n✓ Data validation passed (strict mode)")
    except Exception as e:
        print(f"\n✗ Data validation failed: {e}")
        is_valid = False
    
    if is_valid:
        # Run optimization
        optimizer = BESSOptimizer(bundle)
        results = optimizer.optimize()
        
        # Save results to CSV
        results_path = Path('data/optimization_results.csv')
        optimizer.save_results(results_path)
        print(f"\n✓ Results saved to {results_path}")
    
    return bundle


def create_bundle_from_csv(
    timeseries: 'TimeSeriesData',
    battery_kwh: float = 1000,
    battery_kw: float = 250,
    include_ancillary: bool = False
) -> DataBundle:
    """
    Create a DataBundle from TimeSeriesData.
    
    Args:
        timeseries: Loaded time series data
        battery_kwh: Battery capacity in kWh
        battery_kw: Battery power in kW
        include_ancillary: Include ancillary services
        
    Returns:
        Complete DataBundle ready for optimization
    """
    # Convert to energy (kWh) for the timestep
    ts_energy = timeseries.to_kwh()
    
    # Create time grid
    timegrid = TimeGrid(
        start=timeseries.timestamps[0].isoformat(),
        end=timeseries.timestamps[-1].isoformat(),
        dt_minutes=timeseries.dt_minutes,
        tz=timeseries.timezone
    )
    
    # Architecture
    services = []
    if include_ancillary and timeseries.ancillary_prices:
        if 'fcr' in timeseries.ancillary_prices:
            services.append(AncillaryService(
                name='fcr',
                direction='both',
                sustain_duration_hours=0.25,
                settlement_method='availability_only'
            ))
        if 'afrr_up' in timeseries.ancillary_prices:
            services.append(AncillaryService(
                name='afrr_up',
                direction='up',
                sustain_duration_hours=1.0,
                activation_fraction_up=0.05,
                settlement_method='wholesale_settled'
            ))
    
    architecture = Architecture(
        kind='ac_coupled',
        forbid_retail_to_load=True,
        ancillary_services=services
    )
    
    # Ratings
    pv_inverter_kw = 0
    if timeseries.pv_generation_kw is not None:
        pv_inverter_kw = timeseries.pv_generation_kw.max() * 1.1  # 10% oversized
    
    ratings = Ratings(
        p_charge_max_kw=battery_kw,
        p_discharge_max_kw=battery_kw,
        p_inv_pv_kw=pv_inverter_kw,
        p_inv_bess_kw=battery_kw
    )
    
    # Efficiencies
    efficiencies = Efficiencies(
        eta_c=0.95,
        eta_d=0.95,
        self_discharge=0.0001,
        eta_pv_ac_from_dc=0.98,
        eta_bess_dc_from_ac=0.96,
        eta_bess_ac_from_dc=0.96
    )
    
    # Bounds
    bounds = Bounds(
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
        enforce_terminal_equality=True,
        cycles_per_day=2.0
    )
    
    # Capacity
    capacity = Capacity(capacity_nominal_kwh=battery_kwh)
    
    # Degradation
    degradation = Degradation(
        kind='throughput',
        phi_c_eur_per_kwh=0.02,
        phi_d_eur_per_kwh=0.02
    )
    
    # Tariffs
    tariffs_dict = {
        'price_retail': ts_energy.retail_price,
        'price_wholesale': ts_energy.wholesale_price,
        'price_cap': ts_energy.capacity_price if ts_energy.capacity_price is not None else np.zeros(len(ts_energy.timestamps)),
        'cap_mask': ts_energy.capacity_hours.astype(bool) if ts_energy.capacity_hours is not None else np.zeros(len(ts_energy.timestamps), dtype=bool),
        'cap_duration_hours': 4.0
    }
    
    # Add ancillary prices if available
    if ts_energy.ancillary_prices:
        tariffs_dict['price_ancillary'] = ts_energy.ancillary_prices
    else:
        tariffs_dict['price_ancillary'] = {}
    
    tariffs = Tariffs(**tariffs_dict)
    
    # Exogenous
    exogenous = Exogenous(
        load_ac=ts_energy.site_load_kw,
        pv_dc=ts_energy.pv_generation_kw if ts_energy.pv_generation_kw is not None else np.zeros(len(ts_energy.timestamps))
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


def main():
    """Run all examples."""
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     BESS OPTIMIZATION WITH CSV DATA EXAMPLES          ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    print("Select example to run:")
    print("1. Single CSV file")
    print("2. Multiple CSV files")
    print("3. Full validation workflow")
    print("4. Run all examples")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    try:
        if choice == '1':
            example_single_csv()
        elif choice == '2':
            example_multiple_csvs()
        elif choice == '3':
            example_with_validation()
        elif choice == '4':
            example_single_csv()
            example_multiple_csvs()
            example_with_validation()
        else:
            print("Invalid choice")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
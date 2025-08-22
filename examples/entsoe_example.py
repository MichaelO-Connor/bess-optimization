# examples/entsoe_optimization_example.py
"""
Complete example of BESS optimization with real ENTSO-E market data.
Shows how to fetch live prices and ancillary service data for any European market.

IMPORTANT: Site-specific load (L_t) must be provided by you!
ENTSO-E only has system-wide grid load, not your building's consumption.
"""

import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer, quick_entsoe_optimization
from bess.data_sources.entsoe_integration import get_country_config, COUNTRY_CONFIGS
from bess.data_sources.load_profiles import (
    create_commercial_load_profile,
    create_industrial_load_profile,
    create_residential_load_profile
)


def example_basic_optimization():
    """
    Basic example: Optimize BESS for German market with proper site load.
    """
    print("=" * 60)
    print("BASIC EXAMPLE: German Market with Industrial Load")
    print("=" * 60)
    
    # Make sure API key is set
    if not os.getenv('ENTSOE_API_KEY'):
        print("\n❌ Please set ENTSOE_API_KEY environment variable")
        print("Get your key from: https://transparency.entsoe.eu/")
        return
    
    # Import load profile helper
    from bess.data_sources.load_profiles import create_industrial_load_profile
    
    # Define optimization period
    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=1)
    
    # Create site-specific industrial load profile
    site_load = create_industrial_load_profile(
        start=start,
        end=end,
        dt_minutes=60,
        base_load_kw=200,        # 200 kW continuous process
        production_load_kw=800,  # +800 kW during production
        shifts=[(6, 14), (14, 22)],  # Two shifts
        weekend_production=False
    )
    
    print(f"\n📊 Site Load Profile:")
    print(f"  Base load: 200 kW")
    print(f"  Peak load: 1000 kW")
    print(f"  Daily consumption: {site_load.sum():.0f} kWh")
    
    # Quick optimization with site load
    from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer
    
    optimizer = ENTSOEBESSOptimizer()
    
    bess_config = {
        'capacity_kwh': 1000,  # 1 MWh battery
        'power_kw': 250,       # 250 kW (4-hour battery)
        'efficiency': 0.90,
        'degradation_cost': 0.02
    }
    
    results = optimizer.optimize_with_entsoe(
        country='germany',
        start=start,
        end=end,
        bess_config=bess_config,
        dt_minutes=60,
        load_profile=site_load,  # IMPORTANT: Site-specific load!
        retail_tariff=0.30       # German industrial tariff
    )
    
    print("\n✅ Optimization complete!")
    print(f"Daily revenue: €{results.total_revenue:.2f}")
    
    return results


def example_detailed_optimization():
    """
    Detailed example: Custom configuration with specific ancillary services.
    """
    print("\n" + "=" * 60)
    print("DETAILED EXAMPLE: Netherlands with Custom Services")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ENTSOEBESSOptimizer()
    
    # Define optimization period (next week)
    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=7)
    
    print(f"\nOptimization period: {start.date()} to {end.date()}")
    
    # Custom BESS configuration
    bess_config = {
        'capacity_kwh': 2000,      # 2 MWh
        'power_charge_kw': 500,    # 500 kW charge
        'power_discharge_kw': 500, # 500 kW discharge
        'eta_charge': 0.95,        # 95% charge efficiency
        'eta_discharge': 0.95,     # 95% discharge efficiency
        'soc_min': 0.05,           # 5% minimum SoC
        'soc_max': 0.95,           # 95% maximum SoC
        'soc_initial': 0.5,        # Start at 50%
        'cycles_per_day': 2.0,     # Max 2 cycles per day
        'architecture': 'ac_coupled',
        'degradation_cost': 0.015, # 15 EUR/MWh throughput cost
        'forbid_retail_to_load': True
    }
    
    # Custom ancillary services for Netherlands
    custom_services = [
        {
            'name': 'fcr',
            'direction': 'both',
            'sustain_duration_hours': 0.25,  # 15 minutes
            'settlement_method': 'availability_only',
            'entsoe_product_type': 'FCR'
        },
        {
            'name': 'afrr_up',
            'direction': 'up',
            'sustain_duration_hours': 1.0,  # 1 hour
            'settlement_method': 'wholesale_settled',
            'activation_default': 0.05  # Expect 5% activation
        },
        {
            'name': 'afrr_down',
            'direction': 'down', 
            'sustain_duration_hours': 1.0,
            'settlement_method': 'wholesale_settled',
            'activation_default': 0.05
        }
    ]
    
    # Run optimization
    results = optimizer.optimize_with_entsoe(
        country='netherlands',
        start=start,
        end=end,
        bess_config=bess_config,
        dt_minutes=15,  # 15-minute resolution
        custom_services=custom_services
    )
    
    return results


def example_with_pv_and_load():
    """
    Example with PV generation and site-specific load profile.
    """
    print("\n" + "=" * 60)
    print("PV + LOAD EXAMPLE: Belgium Commercial Building")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = ENTSOEBESSOptimizer()
    
    # Tomorrow's optimization
    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=1)
    
    # Import load profile helper
    from bess.data_sources.load_profiles import create_commercial_load_profile
    
    # Create realistic commercial building load (kW)
    site_load_kw = create_commercial_load_profile(
        start=start,
        end=end,
        dt_minutes=60,
        base_load_kw=50,     # 50 kW overnight
        peak_load_kw=200,    # 200 kW during business hours
        business_hours=(7, 19),
        weekend_factor=0.3
    )
    
    # Create PV profile (1 MW peak solar)
    hours = 24
    time_array = np.arange(hours)
    pv_profile_kw = np.maximum(0, 1000 * np.sin(np.pi * (time_array - 6) / 12) ** 2)
    pv_profile_kw[time_array < 6] = 0
    pv_profile_kw[time_array > 18] = 0
    
    print(f"Site load: {site_load_kw.min():.0f} - {site_load_kw.max():.0f} kW")
    print(f"Daily consumption: {site_load_kw.sum():.0f} kWh")
    print(f"PV peak: {pv_profile_kw.max():.0f} kW")
    
    # BESS with PV
    bess_config = {
        'capacity_kwh': 500,
        'power_kw': 125,           # 4-hour battery
        'efficiency': 0.92,
        'architecture': 'ac_coupled',
        'pv_inverter_kw': 1000,    # 1 MW PV inverter
        'degradation_cost': 0.02
    }
    
    # Run optimization with site-specific load
    results = optimizer.optimize_with_entsoe(
        country='belgium',
        start=start,
        end=end,
        bess_config=bess_config,
        dt_minutes=60,
        pv_profile=pv_profile_kw,      # PV in kW
        load_profile=site_load_kw,     # Site load in kW (NOT from ENTSO-E!)
        retail_tariff=0.25              # Your retail tariff EUR/kWh
    )
    
    # Display hourly dispatch
    print("\n📊 Hourly Dispatch Summary:")
    print("Hour | Load  | PV    | SoC   | Action")
    print("-----|-------|-------|-------|--------")
    
    for h in range(min(24, len(results.soc_total))):
        action = "idle"
        if results.charge_retail[h] > 10:
            action = f"charge {results.charge_retail[h]:.0f} kWh"
        elif results.discharge_load[h] > 10:
            action = f"discharge {results.discharge_load[h]:.0f} kWh"
        elif results.charge_pv[h] > 10:
            action = f"PV charge {results.charge_pv[h]:.0f} kWh"
        
        print(f" {h:2d}  | {site_load_kw[h]:5.0f} | {pv_profile_kw[h]:5.0f} | "
              f"{results.soc_total[h]:5.0f} | {action}")
    
    return results


def example_multi_country_comparison():
    """
    Compare BESS economics across different European markets.
    """
    print("\n" + "=" * 60)
    print("MULTI-COUNTRY COMPARISON")
    print("=" * 60)
    
    # Same BESS in different countries
    bess_config = {
        'capacity_kwh': 1000,
        'power_kw': 250,
        'efficiency': 0.90,
        'architecture': 'ac_coupled',
        'degradation_cost': 0.02
    }
    
    # Tomorrow's market
    start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=1)
    
    optimizer = ENTSOEBESSOptimizer()
    
    countries = ['germany', 'france', 'netherlands', 'belgium']
    results_summary = []
    
    for country in countries:
        try:
            print(f"\n🔄 Optimizing for {country.upper()}...")
            
            bundle = optimizer.create_bundle_from_entsoe(
                country=country,
                start=start,
                end=end,
                bess_config=bess_config,
                dt_minutes=60
            )
            
            # Get prices for summary
            avg_wholesale = np.mean(bundle.tariffs.price_wholesale)
            price_spread = np.max(bundle.tariffs.price_wholesale) - np.min(bundle.tariffs.price_wholesale)
            
            # Run optimization
            from bess.optimize import BESSOptimizer
            bess_opt = BESSOptimizer(bundle)
            results = bess_opt.optimize()
            kpis = bess_opt.get_kpis()
            
            results_summary.append({
                'Country': country.upper(),
                'Avg Price (€/MWh)': avg_wholesale * 1000,
                'Price Spread (€/MWh)': price_spread * 1000,
                'Revenue (€)': kpis['total_revenue'],
                'Cycles': kpis['avg_daily_cycles']
            })
            
        except Exception as e:
            print(f"  ⚠️ Failed for {country}: {e}")
            continue
    
    # Display comparison
    if results_summary:
        print("\n📊 Market Comparison Results:")
        print("-" * 70)
        print(f"{'Country':<12} | {'Avg Price':<12} | {'Spread':<12} | {'Revenue':<10} | {'Cycles':<6}")
        print("-" * 70)
        
        for r in results_summary:
            print(f"{r['Country']:<12} | "
                  f"{r['Avg Price (€/MWh)']:>11.2f} | "
                  f"{r['Price Spread (€/MWh)']:>11.2f} | "
                  f"{r['Revenue (€)']:>9.2f} | "
                  f"{r['Cycles']:>5.2f}")
        
        # Find best market
        best = max(results_summary, key=lambda x: x['Revenue (€)'])
        print("-" * 70)
        print(f"🏆 Best market: {best['Country']} with €{best['Revenue (€)']:.2f} daily revenue")


def main():
    """Run all examples."""
    
    print("""
╔════════════════════════════════════════════════════════════╗
║     BESS OPTIMIZATION WITH REAL ENTSO-E MARKET DATA       ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check for API key
    if not os.getenv('ENTSOE_API_KEY'):
        print("⚠️  ENTSO-E API key not found!")
        print("\nTo get your free API key:")
        print("1. Register at: https://transparency.entsoe.eu/")
        print("2. Email transparency@entsoe.eu requesting 'Restful API access'")
        print("3. Generate token in your account settings")
        print("4. Set environment variable: export ENTSOE_API_KEY='your_key_here'")
        print("\nFor testing, you can use this read-only demo key (may have limits):")
        print("export ENTSOE_API_KEY='1d9cd4bd-f8aa-476c-8cc1-3442dc91506d'")
        return
    
    # Run examples
    print("\nSelect example to run:")
    print("1. Basic optimization (Germany, 1 day)")
    print("2. Detailed optimization (Netherlands, 7 days, custom services)")
    print("3. PV + Load optimization (Belgium, with solar)")
    print("4. Multi-country comparison")
    print("5. Run all examples")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        example_basic_optimization()
    elif choice == '2':
        example_detailed_optimization()
    elif choice == '3':
        example_with_pv_and_load()
    elif choice == '4':
        example_multi_country_comparison()
    elif choice == '5':
        example_basic_optimization()
        example_detailed_optimization()
        example_with_pv_and_load()
        example_multi_country_comparison()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
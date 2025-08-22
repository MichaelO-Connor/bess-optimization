# examples/example_usage.py
"""Example of using the BESS optimization API"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bess import DataBundle, TimeGrid, Architecture, BESSOptimizer
from bess.schema import (
    Ratings, Efficiencies, Bounds, Degradation, 
    Tariffs, Exogenous, Capacity, AncillaryService
)


def create_realistic_prices(hours=24):
    """Create realistic hourly price patterns"""
    # Base prices with daily pattern
    base_retail = 0.15
    base_wholesale = 0.10
    
    # Add daily pattern (higher during day, lower at night)
    hours_array = np.arange(hours)
    
    # Retail: Higher 7am-10pm
    retail_pattern = np.where((hours_array >= 7) & (hours_array < 22), 1.3, 0.8)
    price_retail = base_retail * retail_pattern
    
    # Wholesale: Duck curve - low midday, high evening
    wholesale_pattern = 1.0 + 0.5 * np.sin((hours_array - 6) * np.pi / 12)
    wholesale_pattern[10:15] *= 0.5  # Solar depression
    price_wholesale = base_wholesale * wholesale_pattern
    
    return price_retail, price_wholesale


def create_pv_profile(hours=24, peak_kw=1000):
    """Create realistic PV generation profile"""
    hours_array = np.arange(hours)
    
    # Zero before 6am and after 7pm
    pv = np.zeros(hours)
    
    # Bell curve during daylight
    daylight = (hours_array >= 6) & (hours_array <= 18)
    pv[daylight] = peak_kw * np.sin((hours_array[daylight] - 6) * np.pi / 12) ** 2
    
    return pv


def create_load_profile(hours=24, base_kw=500):
    """Create realistic load profile"""
    hours_array = np.arange(hours)
    
    # Base load with peaks morning and evening
    load = base_kw * (1.0 + 
                     0.3 * np.exp(-((hours_array - 8)**2) / 8) +  # Morning peak
                     0.4 * np.exp(-((hours_array - 19)**2) / 6))   # Evening peak
    
    return load


def main():
    """Run example optimization with realistic data"""
    
    print("=== BESS Revenue Optimization Example ===\n")
    
    # Configuration
    HOURS = 24
    BATTERY_KWH = 1000  # 1 MWh
    BATTERY_KW = 250    # 250 kW (4-hour battery)
    PV_PEAK_KW = 500    # 500 kW peak PV
    
    # Create time grid
    start_time = datetime(2024, 6, 15)  # Summer day
    timegrid = TimeGrid(
        start=start_time.isoformat(),
        end=(start_time + timedelta(hours=HOURS)).isoformat(),
        dt_minutes=60,
        tz="UTC"
    )
    
    print(f"Optimization period: {HOURS} hours")
    print(f"Battery: {BATTERY_KWH} kWh / {BATTERY_KW} kW")
    print(f"PV: {PV_PEAK_KW} kW peak\n")
    
    # Create realistic time series
    price_retail, price_wholesale = create_realistic_prices(HOURS)
    pv_generation = create_pv_profile(HOURS, PV_PEAK_KW)
    load = create_load_profile(HOURS)
    
    # Architecture with ancillary services
    architecture = Architecture(
        kind="ac_coupled",
        forbid_retail_to_load=True,
        ancillary_services=[
            AncillaryService(
                name="fcr",
                direction="both",
                sustain_duration_hours=0.5,
                settlement_method="availability_only"
            ),
            AncillaryService(
                name="afrr_up",
                direction="up",
                sustain_duration_hours=1.0,
                activation_fraction_up=0.05,  # 5% expected activation
                settlement_method="wholesale_settled"
            )
        ],
        import_limit_kw=500,
        export_limit_kw=500
    )
    
    # System ratings
    ratings = Ratings(
        p_charge_max_kw=BATTERY_KW,
        p_discharge_max_kw=BATTERY_KW,
        p_inv_pv_kw=PV_PEAK_KW,
        p_inv_bess_kw=BATTERY_KW
    )
    
    # Efficiencies (realistic values)
    efficiencies = Efficiencies(
        eta_c=0.95,
        eta_d=0.95,
        self_discharge=0.0001,  # 0.01% per hour
        eta_pv_ac_from_dc=0.98,
        eta_bess_dc_from_ac=0.97,
        eta_bess_ac_from_dc=0.97
    )
    
    # Operating bounds
    bounds = Bounds(
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
        enforce_terminal_equality=True,
        cycles_per_day=2.0  # Max 2 cycles per day
    )
    
    # Capacity
    capacity = Capacity(capacity_nominal_kwh=BATTERY_KWH)
    
    # Degradation model
    degradation = Degradation(
        kind="throughput",
        phi_c_eur_per_kwh=0.02,  # €20/MWh degradation
        phi_d_eur_per_kwh=0.02
    )
    
    # Tariffs
    tariffs = Tariffs(
        price_retail=price_retail,
        price_wholesale=price_wholesale,
        price_ancillary={
            "fcr": np.full(HOURS, 15.0),  # €15/MW/h
            "afrr_up": np.full(HOURS, 10.0)  # €10/MW/h
        },
        price_cap=np.zeros(HOURS),
        cap_mask=np.zeros(HOURS, dtype=bool),
        cap_duration_hours=4.0
    )
    
    # Exogenous data
    exogenous = Exogenous(
        load_ac=load,
        pv_dc=pv_generation
    )
    
    # Create bundle
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
    
    # Optimize
    print("Running optimization...")
    optimizer = BESSOptimizer(bundle)
    results = optimizer.optimize(time_limit=30, mip_gap=0.001)
    
    # Get KPIs
    kpis = optimizer.get_kpis()
    
    print("\n=== Optimization Results ===")
    print(f"Status: {results.status}")
    print(f"Solve time: {results.solve_time:.2f}s")
    
    print("\n=== Financial Summary ===")
    print(f"Total revenue:        €{kpis['total_revenue']:8.2f}")
    print(f"Energy cost:          €{kpis['energy_cost']:8.2f}")
    print(f"Ancillary revenue:    €{kpis['ancillary_revenue']:8.2f}")
    print(f"Capacity revenue:     €{kpis['capacity_revenue']:8.2f}")
    print(f"Degradation cost:     €{kpis['degradation_cost']:8.2f}")
    
    print("\n=== Operational Metrics ===")
    print(f"Average daily cycles: {kpis['avg_daily_cycles']:.2f}")
    print(f"Battery utilization:  {kpis['utilization']:.1%}")
    print(f"PV curtailed:         {kpis['total_pv_curtailed']:.1f} kWh")
    print(f"Round-trip losses:    {kpis['roundtrip_losses']:.1f} kWh")
    
    # Create hourly summary
    print("\n=== Hourly Dispatch (first 12 hours) ===")
    print("Hour | Load  | PV   | Price | SoC   | Action")
    print("-----|-------|------|-------|-------|--------")
    
    for h in range(min(12, HOURS)):
        action = "idle"
        if results.charge_retail[h] > 1:
            action = f"charge {results.charge_retail[h]:.0f}kW"
        elif results.discharge_load[h] > 1:
            action = f"discharge {results.discharge_load[h]:.0f}kW"
        elif results.charge_pv[h] > 1:
            action = f"PV charge {results.charge_pv[h]:.0f}kW"
            
        print(f" {h:2d}  | {load[h]:5.0f} | {pv_generation[h]:4.0f} | "
              f"{price_wholesale[h]:5.3f} | {results.soc_total[h]/BATTERY_KWH:5.1%} | {action}")
    
    # Save results
    print("\nSaving results...")
    optimizer.save_results("example_results.csv")
    
    # Plot if matplotlib available
    try:
        fig = optimizer.plot_dispatch("example_dispatch.png", show=False)
        if fig:
            print("Dispatch plot saved to example_dispatch.png")
    except:
        pass
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
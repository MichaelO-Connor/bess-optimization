# tests/test_optimization.py
"""Simple test to verify BESS optimization works"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bess import DataBundle, TimeGrid, Architecture, BESSOptimizer
from bess.schema import (
    Ratings, Efficiencies, Bounds, Degradation, 
    Tariffs, Exogenous, Capacity, AncillaryService
)


def create_test_bundle():
    """Create a minimal test DataBundle"""
    
    # Time grid - 4 hours at 1-hour resolution
    timegrid = TimeGrid(
        start="2024-01-01T00:00:00",
        end="2024-01-01T04:00:00",
        dt_minutes=60,
        tz="UTC"
    )
    T = 4
    
    # Simple AC-coupled architecture
    architecture = Architecture(
        kind="ac_coupled",
        forbid_retail_to_load=True,
        ancillary_services=[
            AncillaryService(
                name="fcr",
                direction="both",
                sustain_duration_hours=0.5,
                settlement_method="availability_only"
            )
        ]
    )
    
    # Ratings
    ratings = Ratings(
        p_charge_max_kw=100,
        p_discharge_max_kw=100,
        p_inv_pv_kw=50,
        p_inv_bess_kw=100
    )
    
    # Efficiencies
    efficiencies = Efficiencies(
        eta_c=0.95,
        eta_d=0.95,
        self_discharge=0.0,
        eta_pv_ac_from_dc=0.98,
        eta_bess_dc_from_ac=0.96,
        eta_bess_ac_from_dc=0.96
    )
    
    # Bounds
    bounds = Bounds(
        soc_min=0.1,
        soc_max=0.9,
        soc_initial=0.5,
        enforce_terminal_equality=True
    )
    
    # Capacity
    capacity = Capacity(capacity_nominal_kwh=200)
    
    # Degradation
    degradation = Degradation(
        kind="throughput",
        phi_c_eur_per_kwh=0.01,
        phi_d_eur_per_kwh=0.01
    )
    
    # Tariffs - simple price pattern
    tariffs = Tariffs(
        price_retail=np.array([0.20, 0.15, 0.10, 0.25]),  # Peak at end
        price_wholesale=np.array([0.10, 0.08, 0.05, 0.20]),  # Peak at end
        price_ancillary={"fcr": np.array([10, 10, 10, 10])},  # €/MW/h
        price_cap=np.array([0, 0, 0, 0]),
        cap_mask=np.array([False, False, False, False]),
        cap_duration_hours=1.0
    )
    
    # Exogenous - simple load and PV
    exogenous = Exogenous(
        load_ac=np.array([50, 60, 70, 80]),  # kWh per hour
        pv_dc=np.array([0, 30, 40, 0])  # kWh per hour
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


def test_basic_optimization():
    """Test basic optimization functionality"""
    print("Creating test bundle...")
    bundle = create_test_bundle()
    
    print("Validating bundle...")
    bundle.validate_lengths()
    bundle.validate_architecture_requirements()
    
    print("Creating optimizer...")
    optimizer = BESSOptimizer(bundle)
    
    print("Running optimization...")
    results = optimizer.optimize(verbose=False, time_limit=10)
    
    print("\n=== Results ===")
    print(f"Status: {results.status}")
    print(f"Objective: €{results.objective_value:.2f}")
    print(f"Total revenue: €{results.total_revenue:.2f}")
    print(f"Solve time: {results.solve_time:.2f}s")
    
    # Check basic feasibility
    assert results.status in [2, 9], f"Unexpected status: {results.status}"  # OPTIMAL or TIME_LIMIT
    
    # Check SoC bounds
    soc_min = bundle.bounds.soc_min * bundle.capacity.capacity_nominal_kwh
    soc_max = bundle.bounds.soc_max * bundle.capacity.capacity_nominal_kwh
    assert np.all(results.soc_total >= soc_min - 1e-6), "SoC below minimum"
    assert np.all(results.soc_total <= soc_max + 1e-6), "SoC above maximum"
    
    # Check terminal condition
    initial_soc = bundle.bounds.soc_initial * bundle.capacity.capacity_nominal_kwh
    terminal_soc = results.soc_total[-1]
    if bundle.bounds.enforce_terminal_equality:
        assert abs(terminal_soc - initial_soc) < 1e-3, f"Terminal SoC {terminal_soc:.2f} != Initial {initial_soc:.2f}"
    
    print("\n✓ All tests passed!")
    
    # Print dispatch summary
    print("\n=== Dispatch Summary ===")
    print("Time | SoC   | Charge | Discharge | PV->Load | Price")
    print("-----|-------|--------|-----------|----------|-------")
    for t in range(len(results.soc_total)):
        print(f" {t:2d}  | {results.soc_total[t]:5.1f} | "
              f"{results.charge_retail[t] + results.charge_wholesale[t]:6.1f} | "
              f"{results.discharge_load[t] + results.discharge_wholesale[t]:9.1f} | "
              f"{results.pv_to_load[t]:8.1f} | "
              f"{bundle.tariffs.price_wholesale[t]:.2f}")
    
    return results


if __name__ == "__main__":
    try:
        results = test_basic_optimization()
        print("\n✅ BESS optimization test successful!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
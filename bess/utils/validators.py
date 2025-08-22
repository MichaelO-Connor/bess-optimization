"""
Data validation utilities for BESS optimization.
Provides comprehensive validation for input data and configurations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

from bess.schema import DataBundle, Architecture

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Comprehensive data validation for BESS optimization."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, raise exceptions on errors. If False, log warnings.
        """
        self.strict = strict
        self.errors = []
        self.warnings = []
    
    def validate_data_bundle(self, bundle: DataBundle) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete DataBundle.
        
        Args:
            bundle: DataBundle to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_time_series_lengths(bundle)
        self._validate_time_series_values(bundle)
        self._validate_architecture_consistency(bundle)
        self._validate_power_ratings(bundle)
        self._validate_efficiencies(bundle)
        self._validate_soc_bounds(bundle)
        self._validate_prices(bundle)
        self._validate_ancillary_services(bundle)
        self._validate_capacity_constraints(bundle)
        
        is_valid = len(self.errors) == 0
        
        if not is_valid and self.strict:
            raise ValidationError(f"Validation failed with {len(self.errors)} errors:\n" + 
                                "\n".join(self.errors))
        
        return is_valid, self.errors, self.warnings
    
    def _validate_time_series_lengths(self, bundle: DataBundle):
        """Check all time series have consistent length."""
        T = len(bundle.timegrid.index())
        
        # Core time series
        series_to_check = [
            ("load_ac", bundle.exogenous.load_ac),
            ("pv_dc", bundle.exogenous.pv_dc),
            ("price_retail", bundle.tariffs.price_retail),
            ("price_wholesale", bundle.tariffs.price_wholesale),
            ("price_cap", bundle.tariffs.price_cap),
            ("cap_mask", bundle.tariffs.cap_mask),
        ]
        
        for name, series in series_to_check:
            if len(series) != T:
                self.errors.append(f"{name}: length {len(series)} != expected {T}")
        
        # Ancillary service prices
        for service_name, prices in bundle.tariffs.price_ancillary.items():
            if len(prices) != T:
                self.errors.append(f"Ancillary price {service_name}: length {len(prices)} != {T}")
        
        # Optional time-varying capacity
        if bundle.capacity.usable_capacity_kwh is not None:
            if len(bundle.capacity.usable_capacity_kwh) != T:
                self.errors.append(f"Usable capacity: length != {T}")
    
    def _validate_time_series_values(self, bundle: DataBundle):
        """Check time series values are within reasonable ranges."""
        # Check for NaN/inf
        for name, series in [
            ("load_ac", bundle.exogenous.load_ac),
            ("pv_dc", bundle.exogenous.pv_dc),
            ("price_retail", bundle.tariffs.price_retail),
            ("price_wholesale", bundle.tariffs.price_wholesale),
        ]:
            if np.any(~np.isfinite(series)):
                self.errors.append(f"{name} contains NaN or infinite values")
        
        # Check non-negativity where required
        if np.any(bundle.exogenous.pv_dc < 0):
            self.errors.append("PV generation cannot be negative")
        
        # Check price reasonableness
        max_price = 10.0  # €/kWh
        if np.any(bundle.tariffs.price_retail > max_price):
            self.warnings.append(f"Retail prices exceed {max_price} €/kWh")
        
        if np.any(np.abs(bundle.tariffs.price_wholesale) > max_price):
            self.warnings.append(f"Wholesale prices exceed ±{max_price} €/kWh")
        
        # Check load reasonableness
        max_load = bundle.ratings.p_discharge_max_kw * 10  # 10x battery power
        if np.any(np.abs(bundle.exogenous.load_ac) > max_load):
            self.warnings.append(f"Site load exceeds {max_load} kW")
    
    def _validate_architecture_consistency(self, bundle: DataBundle):
        """Validate architecture-specific requirements."""
        arch = bundle.arch.kind
        
        if arch == "ac_coupled":
            required = [
                (bundle.ratings.p_inv_pv_kw, "p_inv_pv_kw"),
                (bundle.ratings.p_inv_bess_kw, "p_inv_bess_kw"),
                (bundle.eff.eta_pv_ac_from_dc, "eta_pv_ac_from_dc"),
                (bundle.eff.eta_bess_dc_from_ac, "eta_bess_dc_from_ac"),
                (bundle.eff.eta_bess_ac_from_dc, "eta_bess_ac_from_dc"),
            ]
        elif arch in ["dc_coupled", "hybrid"]:
            required = [
                (bundle.ratings.p_inv_shared_kw, "p_inv_shared_kw"),
                (bundle.eff.eta_shared_dc_from_ac, "eta_shared_dc_from_ac"),
                (bundle.eff.eta_shared_ac_from_dc, "eta_shared_ac_from_dc"),
            ]
        else:
            self.errors.append(f"Unknown architecture: {arch}")
            return
        
        for value, name in required:
            if value is None:
                self.errors.append(f"{arch} requires {name}")
    
    def _validate_power_ratings(self, bundle: DataBundle):
        """Validate power ratings are consistent and reasonable."""
        # Basic positivity
        if bundle.ratings.p_charge_max_kw <= 0:
            self.errors.append("Charge power must be positive")
        if bundle.ratings.p_discharge_max_kw <= 0:
            self.errors.append("Discharge power must be positive")
        
        # C-rate check
        c_rate_charge = bundle.ratings.p_charge_max_kw / bundle.capacity.capacity_nominal_kwh
        c_rate_discharge = bundle.ratings.p_discharge_max_kw / bundle.capacity.capacity_nominal_kwh
        
        if c_rate_charge > 10:
            self.warnings.append(f"Very high C-rate for charging: {c_rate_charge:.1f}C")
        if c_rate_discharge > 10:
            self.warnings.append(f"Very high C-rate for discharging: {c_rate_discharge:.1f}C")
        
        # Architecture-specific checks
        if bundle.arch.kind == "ac_coupled":
            if bundle.ratings.p_inv_bess_kw < max(bundle.ratings.p_charge_max_kw, 
                                                  bundle.ratings.p_discharge_max_kw):
                self.warnings.append("BESS inverter may limit battery power")
    
    def _validate_efficiencies(self, bundle: DataBundle):
        """Validate efficiency values."""
        # Battery efficiencies
        if not 0 < bundle.eff.eta_c <= 1:
            self.errors.append(f"Charge efficiency {bundle.eff.eta_c} not in (0,1]")
        if not 0 < bundle.eff.eta_d <= 1:
            self.errors.append(f"Discharge efficiency {bundle.eff.eta_d} not in (0,1]")
        
        # Round-trip efficiency check
        rt_eff = bundle.eff.eta_c * bundle.eff.eta_d
        if rt_eff < 0.5:
            self.warnings.append(f"Very low round-trip efficiency: {rt_eff:.1%}")
        
        # Self-discharge
        if not 0 <= bundle.eff.self_discharge < 0.1:
            self.warnings.append(f"Unusual self-discharge rate: {bundle.eff.self_discharge}")
    
    def _validate_soc_bounds(self, bundle: DataBundle):
        """Validate state of charge bounds."""
        if not 0 <= bundle.bounds.soc_min < bundle.bounds.soc_max <= 1:
            self.errors.append("Invalid SoC bounds: require 0 ≤ soc_min < soc_max ≤ 1")
        
        # Check usable capacity
        usable_fraction = bundle.bounds.soc_max - bundle.bounds.soc_min
        if usable_fraction < 0.5:
            self.warnings.append(f"Low usable capacity: {usable_fraction:.1%}")
        
        # Initial SoC
        if bundle.bounds.soc_initial is not None:
            if not bundle.bounds.soc_min <= bundle.bounds.soc_initial <= bundle.bounds.soc_max:
                self.errors.append("Initial SoC outside of bounds")
        
        # Cycles per day
        if bundle.bounds.cycles_per_day is not None:
            if bundle.bounds.cycles_per_day > 10:
                self.warnings.append(f"Very high cycling limit: {bundle.bounds.cycles_per_day} cycles/day")
    
    def _validate_prices(self, bundle: DataBundle):
        """Validate price data."""
        # Check for all zero prices
        if np.all(bundle.tariffs.price_retail == bundle.tariffs.price_retail[0]):
            self.warnings.append("Retail prices are constant - no arbitrage opportunity")
        
        if np.all(bundle.tariffs.price_wholesale == bundle.tariffs.price_wholesale[0]):
            self.warnings.append("Wholesale prices are constant - no arbitrage opportunity")
        
        # Check retail vs wholesale spread
        avg_spread = np.mean(bundle.tariffs.price_retail - bundle.tariffs.price_wholesale)
        if avg_spread < 0:
            self.warnings.append("Average retail price below wholesale - unusual market")
        
        # Capacity market
        if np.any(bundle.tariffs.cap_mask):
            if bundle.tariffs.cap_duration_hours <= 0:
                self.errors.append("Capacity duration must be positive when capacity market active")
    
    def _validate_ancillary_services(self, bundle: DataBundle):
        """Validate ancillary service configuration."""
        for service in bundle.arch.ancillary_services:
            # Check price data exists
            if service.name not in bundle.tariffs.price_ancillary:
                self.errors.append(f"No price data for service {service.name}")
            
            # Check sustain duration
            if service.sustain_duration_hours <= 0:
                self.errors.append(f"Service {service.name}: sustain duration must be positive")
            
            # Check sustain duration feasibility
            max_hours = (bundle.bounds.soc_max - bundle.bounds.soc_min) * \
                       bundle.capacity.capacity_nominal_kwh / bundle.ratings.p_discharge_max_kw
            
            if service.sustain_duration_hours > max_hours:
                self.warnings.append(f"Service {service.name}: sustain duration {service.sustain_duration_hours}h "
                                   f"may not be deliverable (max ~{max_hours:.1f}h)")
            
            # Check activation fractions
            if service.activation_fraction_up is not None:
                if not 0 <= service.activation_fraction_up <= 1:
                    self.errors.append(f"Service {service.name}: invalid activation fraction up")
            
            if service.activation_fraction_down is not None:
                if not 0 <= service.activation_fraction_down <= 1:
                    self.errors.append(f"Service {service.name}: invalid activation fraction down")
    
    def _validate_capacity_constraints(self, bundle: DataBundle):
        """Validate battery capacity and related constraints."""
        if bundle.capacity.capacity_nominal_kwh <= 0:
            self.errors.append("Battery capacity must be positive")
        
        # Check capacity vs power (C-rate)
        min_duration_hours = bundle.capacity.capacity_nominal_kwh / bundle.ratings.p_discharge_max_kw
        if min_duration_hours < 0.25:
            self.warnings.append(f"Very short duration battery: {min_duration_hours:.2f} hours")
        elif min_duration_hours > 10:
            self.warnings.append(f"Very long duration battery: {min_duration_hours:.1f} hours")
        
        # Degradation costs
        if bundle.degradation.kind == "throughput":
            total_deg_cost = bundle.degradation.phi_c_eur_per_kwh + bundle.degradation.phi_d_eur_per_kwh
            
            if total_deg_cost > 0.1:  # €100/MWh
                self.warnings.append(f"High degradation cost: €{total_deg_cost*1000:.0f}/MWh")


def validate_data_bundle(bundle: DataBundle, strict: bool = True) -> bool:
    """
    Convenience function to validate a DataBundle.
    
    Args:
        bundle: DataBundle to validate
        strict: If True, raise exception on errors
        
    Returns:
        True if valid, False otherwise
    """
    validator = DataValidator(strict=strict)
    is_valid, errors, warnings = validator.validate_data_bundle(bundle)
    
    # Log results
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    if errors:
        for error in errors:
            logger.error(error)
    
    return is_valid


def check_data_consistency(
    bundle: DataBundle,
    timeseries_data: Optional['TimeSeriesData'] = None
) -> Dict[str, Any]:
    """
    Check consistency between DataBundle and TimeSeriesData.
    
    Args:
        bundle: DataBundle configuration
        timeseries_data: Optional TimeSeriesData to cross-check
        
    Returns:
        Dictionary with consistency check results
    """
    results = {
        'consistent': True,
        'issues': [],
        'statistics': {}
    }
    
    # Basic statistics
    results['statistics']['timesteps'] = len(bundle.timegrid.index())
    results['statistics']['duration_hours'] = results['statistics']['timesteps'] * bundle.dt_hours
    results['statistics']['battery_duration_hours'] = (
        bundle.capacity.capacity_nominal_kwh / bundle.ratings.p_discharge_max_kw
    )
    
    # Check time alignment if timeseries provided
    if timeseries_data is not None:
        bundle_timestamps = bundle.timegrid.index()
        
        if len(bundle_timestamps) != len(timeseries_data.timestamps):
            results['consistent'] = False
            results['issues'].append(
                f"Timestamp count mismatch: bundle has {len(bundle_timestamps)}, "
                f"timeseries has {len(timeseries_data.timestamps)}"
            )
        
        # Check dt consistency
        if bundle.timegrid.dt_minutes != timeseries_data.dt_minutes:
            results['consistent'] = False
            results['issues'].append(
                f"Timestep mismatch: bundle has {bundle.timegrid.dt_minutes} min, "
                f"timeseries has {timeseries_data.dt_minutes} min"
            )
    
    # Check if PV configured but no generation data
    if bundle.arch.kind == "ac_coupled" and bundle.ratings.p_inv_pv_kw:
        if bundle.ratings.p_inv_pv_kw > 0 and np.all(bundle.exogenous.pv_dc == 0):
            results['issues'].append("PV inverter configured but no PV generation data")
    
    # Check ancillary service feasibility
    for service in bundle.arch.ancillary_services:
        required_energy = service.sustain_duration_hours * bundle.ratings.p_discharge_max_kw
        available_energy = (bundle.bounds.soc_max - bundle.bounds.soc_min) * \
                          bundle.capacity.capacity_nominal_kwh * bundle.eff.eta_d
        
        if required_energy > available_energy:
            results['issues'].append(
                f"Service {service.name} may not be fully deliverable: "
                f"requires {required_energy:.0f} kWh, available {available_energy:.0f} kWh"
            )
    
    # Revenue opportunity check
    price_spread = np.max(bundle.tariffs.price_wholesale) - np.min(bundle.tariffs.price_wholesale)
    if price_spread < 0.01:  # €10/MWh
        results['issues'].append(f"Very low price spread: €{price_spread*1000:.1f}/MWh")
    
    results['statistics']['price_spread_eur_mwh'] = price_spread * 1000
    results['statistics']['avg_load_kw'] = np.mean(bundle.exogenous.load_ac)
    
    if results['issues']:
        results['consistent'] = False
    
    return results


def generate_validation_report(
    bundle: DataBundle,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive validation report.
    
    Args:
        bundle: DataBundle to analyze
        output_path: Optional path to save report
        
    Returns:
        Report as string
    """
    validator = DataValidator(strict=False)
    is_valid, errors, warnings = validator.validate_data_bundle(bundle)
    consistency = check_data_consistency(bundle)
    
    # Build report
    lines = [
        "=" * 60,
        "BESS OPTIMIZATION DATA VALIDATION REPORT",
        "=" * 60,
        f"Generated: {pd.Timestamp.now()}",
        "",
        "CONFIGURATION SUMMARY",
        "-" * 40,
        f"Architecture: {bundle.arch.kind}",
        f"Battery: {bundle.capacity.capacity_nominal_kwh:.0f} kWh / "
        f"{bundle.ratings.p_discharge_max_kw:.0f} kW",
        f"Timesteps: {consistency['statistics']['timesteps']}",
        f"Duration: {consistency['statistics']['duration_hours']:.1f} hours",
        f"Resolution: {bundle.timegrid.dt_minutes} minutes",
        "",
    ]
    
    if bundle.arch.ancillary_services:
        lines.extend([
            "ANCILLARY SERVICES",
            "-" * 40,
        ])
        for service in bundle.arch.ancillary_services:
            lines.append(f"  - {service.name}: {service.direction}, "
                        f"{service.sustain_duration_hours:.1f}h sustain")
        lines.append("")
    
    lines.extend([
        "VALIDATION RESULTS",
        "-" * 40,
        f"Status: {'✓ VALID' if is_valid else '✗ INVALID'}",
        f"Errors: {len(errors)}",
        f"Warnings: {len(warnings)}",
        "",
    ])
    
    if errors:
        lines.extend([
            "ERRORS (must fix)",
            "-" * 40,
        ])
        for error in errors:
            lines.append(f"  ✗ {error}")
        lines.append("")
    
    if warnings:
        lines.extend([
            "WARNINGS (review)",
            "-" * 40,
        ])
        for warning in warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")
    
    if consistency['issues']:
        lines.extend([
            "CONSISTENCY ISSUES",
            "-" * 40,
        ])
        for issue in consistency['issues']:
            lines.append(f"  ⚠ {issue}")
        lines.append("")
    
    lines.extend([
        "STATISTICS",
        "-" * 40,
        f"Price spread: €{consistency['statistics'].get('price_spread_eur_mwh', 0):.1f}/MWh",
        f"Average load: {consistency['statistics'].get('avg_load_kw', 0):.0f} kW",
        f"Battery duration: {consistency['statistics'].get('battery_duration_hours', 0):.1f} hours",
        "",
        "=" * 60,
    ])
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.write_text(report)
        logger.info(f"Validation report saved to {output_path}")
    
    return report
# bess/data_model/milp_interface.py
"""
Clean interface between DataBundle and MILP solver.
Updated to handle generic ancillary services, frequency response activation,
enhanced action selection framework, and hybrid architecture.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from .schema import DataBundle, AncillaryService
from .registry import SYMBOLS

class MILPDataInterface:
    """Clean handoff from DataBundle to MILP solver with enhanced services framework and hybrid support"""
    
    def __init__(self, bundle: DataBundle):
        self.bundle = bundle
        self.T = len(bundle.timegrid.index())
        self._validate_bundle()
        
    def _validate_bundle(self):
        """Ensure bundle is ready for MILP"""
        try:
            self.bundle.validate_lengths()
            self.bundle.validate_architecture_requirements()
        except Exception as e:
            raise ValueError(f"Bundle validation failed: {e}")
    
    def to_milp_params(self) -> Dict[str, Any]:
        """Convert DataBundle to complete MILP parameter dictionary with generic services"""
        b = self.bundle
        
        params = {}
        
        # =================================================================
        # TIME AND INDEXING
        # =================================================================
        params.update({
            "T": self.T,
            "dt_hours": b.dt_hours,
            "time_index": b.timegrid.index(),
            "tz": b.timegrid.tz,
        })
        
        # =================================================================
        # EXOGENOUS TIME SERIES (guaranteed length T)
        # =================================================================
        params.update({
            # Energy quantities (kWh per timestep)
            "L_t": b.exogenous.load_ac,
            "a_t": b.exogenous.pv_dc,
            
            # Energy prices (€/kWh)
            "P_ret_t": b.tariffs.price_retail,
            "P_wh_t": b.tariffs.price_wholesale,
            
            # Capacity market
            "pi_cap_t": b.tariffs.price_cap,
            "Omega_cap": b.tariffs.cap_mask,  # Boolean mask
        })
        
        # =================================================================
        # GENERIC ANCILLARY SERVICES FRAMEWORK
        # =================================================================
        # Get service configurations
        services_by_direction = b.services_by_direction
        params["K"] = [service.name for service in b.arch.ancillary_services]  # All services
        params["K_up"] = services_by_direction["up"]      # K^↑
        params["K_down"] = services_by_direction["down"]  # K^↓
        
        # Service-specific parameters
        params["service_configs"] = {
            service.name: {
                "direction": service.direction,
                "sustain_duration_hours": service.sustain_duration_hours,
                "settlement_method": service.settlement_method,
                "activation_fraction_up": service.activation_fraction_up,
                "activation_fraction_down": service.activation_fraction_down,
            }
            for service in b.arch.ancillary_services
        }
        
        # Ancillary service prices (€/kW per period)
        params["pi_k_t"] = {}  # Availability prices by service
        for service_name in params["K"]:
            if service_name in b.tariffs.price_ancillary:
                params["pi_k_t"][service_name] = b.tariffs.price_ancillary[service_name]
            else:
                # Default to zero if not specified
                params["pi_k_t"][service_name] = np.zeros(self.T)
        
        # Activation settlement prices (€/kWh)
        params["u_k_up_t"] = {}   # Upward activation prices
        params["u_k_down_t"] = {} # Downward activation prices
        
        for service_name in params["K"]:
            service_config = params["service_configs"][service_name]
            
            # Set activation prices based on settlement method
            if service_config["settlement_method"] == "availability_only":
                # No activation settlement
                params["u_k_up_t"][service_name] = np.zeros(self.T)
                params["u_k_down_t"][service_name] = np.zeros(self.T)
            elif service_config["settlement_method"] == "wholesale_settled":
                # Use wholesale prices
                params["u_k_up_t"][service_name] = b.tariffs.price_wholesale.copy()
                params["u_k_down_t"][service_name] = b.tariffs.price_wholesale.copy()
            elif service_config["settlement_method"] == "explicit_prices":
                # Use explicit activation prices
                params["u_k_up_t"][service_name] = b.tariffs.price_activation_up.get(
                    service_name, np.zeros(self.T)
                )
                params["u_k_down_t"][service_name] = b.tariffs.price_activation_down.get(
                    service_name, np.zeros(self.T)
                )
        
        # Activation fractions (expected call rates)
        params["alpha_k_up_t"] = {}
        params["alpha_k_down_t"] = {}
        
        for service_name in params["K"]:
            service_config = params["service_configs"][service_name]
            
            # Use time-varying if available, otherwise constant from service config
            if service_name in b.tariffs.activation_up and b.tariffs.activation_up[service_name] is not None:
                params["alpha_k_up_t"][service_name] = b.tariffs.activation_up[service_name]
            elif service_config["activation_fraction_up"] is not None:
                params["alpha_k_up_t"][service_name] = np.full(self.T, service_config["activation_fraction_up"])
            else:
                params["alpha_k_up_t"][service_name] = np.zeros(self.T)  # No activation expected
            
            if service_name in b.tariffs.activation_down and b.tariffs.activation_down[service_name] is not None:
                params["alpha_k_down_t"][service_name] = b.tariffs.activation_down[service_name]
            elif service_config["activation_fraction_down"] is not None:
                params["alpha_k_down_t"][service_name] = np.full(self.T, service_config["activation_fraction_down"])
            else:
                params["alpha_k_down_t"][service_name] = np.zeros(self.T)  # No activation expected
        
        # Sustain duration requirements (hours)
        params["H_k_up"] = {}
        params["H_k_down"] = {}
        
        for service_name in params["K"]:
            service_config = params["service_configs"][service_name]
            duration = service_config["sustain_duration_hours"]
            
            if service_config["direction"] in ["up", "both"]:
                params["H_k_up"][service_name] = duration
            if service_config["direction"] in ["down", "both"]:
                params["H_k_down"][service_name] = duration
        
        # =================================================================
        # BATTERY PARAMETERS
        # =================================================================
        params.update({
            "C_nom": b.capacity.capacity_nominal_kwh,
            "SoC_min": b.bounds.soc_min,
            "SoC_max": b.bounds.soc_max,
            "eta_c": b.eff.eta_c,
            "eta_d": b.eff.eta_d,
            "delta": b.eff.self_discharge,
        })
        
        # Time-varying or constant usable capacity
        if b.capacity.usable_capacity_kwh is not None:
            params["C_t"] = b.capacity.usable_capacity_kwh
        else:
            params["C_t"] = np.full(self.T, b.capacity.capacity_nominal_kwh)
        
        # Optional throughput limits
        if b.bounds.cycles_per_day is not None:
            params["n_cycles"] = b.bounds.cycles_per_day
        
        # Initial SoC split factors (if provided)
        if b.bounds.initial_soc_split is not None:
            params["initial_soc_split"] = b.bounds.initial_soc_split
        
        # =================================================================
        # POWER RATINGS AND GRID LIMITS
        # =================================================================
        params.update({
            "P_ch_max": b.ratings.p_charge_max_kw,
            "P_dis_max": b.ratings.p_discharge_max_kw,
        })
        
        # Architecture-specific ratings
        if b.arch.kind == "ac_coupled":
            params.update({
                "P_inv_pv": b.ratings.p_inv_pv_kw,
                "P_inv_bess": b.ratings.p_inv_bess_kw,
                "architecture": "ac_coupled"
            })
        elif b.arch.kind == "dc_coupled":
            params.update({
                "P_inv_shared": b.ratings.p_inv_shared_kw,
                "architecture": "dc_coupled"
            })
        elif b.arch.kind == "hybrid":
            params.update({
                "P_inv_shared": b.ratings.p_inv_shared_kw,
                "architecture": "hybrid"
            })
            # Optional PV DC/DC rating for hybrid
            if b.ratings.p_dc_pv_kw is not None:
                params["P_dc_pv"] = b.ratings.p_dc_pv_kw
        
        # Grid connection limits
        if b.ratings.p_grid_import_max_kw is not None:
            params["P_grid_import_max"] = b.ratings.p_grid_import_max_kw
        if b.ratings.p_grid_export_max_kw is not None:
            params["P_grid_export_max"] = b.ratings.p_grid_export_max_kw
        
        # Fallback to general grid limit if specific import/export not set
        if b.arch.import_limit_kw is not None and "P_grid_import_max" not in params:
            params["P_grid_import_max"] = b.arch.import_limit_kw
        if b.arch.export_limit_kw is not None and "P_grid_export_max" not in params:
            params["P_grid_export_max"] = b.arch.export_limit_kw
        
        # =================================================================
        # EFFICIENCIES (architecture-aware)
        # =================================================================
        # Raw component efficiencies
        if b.arch.kind == "ac_coupled":
            params.update({
                "eta_pv_ac_from_dc": b.eff.eta_pv_ac_from_dc,
                "eta_bess_dc_from_ac": b.eff.eta_bess_dc_from_ac,
                "eta_bess_ac_from_dc": b.eff.eta_bess_ac_from_dc,
            })
        elif b.arch.kind in ["dc_coupled", "hybrid"]:
            params.update({
                "eta_shared_dc_from_ac": b.eff.eta_shared_dc_from_ac,
                "eta_shared_ac_from_dc": b.eff.eta_shared_ac_from_dc,
            })
        
        # Unified path efficiencies for objective function
        path_eff = b.path_efficiencies
        params.update({
            "eta_imp": path_eff["eta_imp"],      # Grid AC→DC for charging
            "eta_exp": path_eff["eta_exp"],      # Battery DC→AC for export
            "eta_pv_to_load": path_eff["eta_pv_to_load"],    # PV→load path
            "eta_pv_to_battery": path_eff["eta_pv_to_battery"], # PV→battery path
        })
        
        # =================================================================
        # MARKET SERVICES PARAMETERS
        # =================================================================
        params.update({
            "pi_dem": b.tariffs.price_demand,
            "H_cap": b.tariffs.cap_duration_hours,
        })
        
        # =================================================================
        # BOUNDARY CONDITIONS
        # =================================================================
        if b.bounds.soc_initial is not None:
            params["SoC_0"] = b.bounds.soc_initial
        else:
            # Default to mid-range
            params["SoC_0"] = 0.5 * (b.bounds.soc_min + b.bounds.soc_max)
        
        if b.bounds.soc_terminal_min is not None:
            params["SoC_T_min"] = b.bounds.soc_terminal_min
        if b.bounds.soc_terminal_max is not None:
            params["SoC_T_max"] = b.bounds.soc_terminal_max
        
        params["enforce_terminal_equality"] = b.bounds.enforce_terminal_equality
        
        # =================================================================
        # PERMISSIONS AND POLICY
        # =================================================================
        # Default permissions matrix
        default_permissions = self._build_default_permissions()
        
        # Override with explicit permissions if provided
        if b.arch.permissions is not None:
            permissions = {**default_permissions, **b.arch.permissions}
        else:
            permissions = default_permissions
        
        params["permissions"] = permissions
        
        # =================================================================
        # DEGRADATION
        # =================================================================
        params.update({
            "degradation_kind": b.degradation.kind,
            "phi_c": b.degradation.phi_c_eur_per_kwh,
            "phi_d": b.degradation.phi_d_eur_per_kwh,
        })
        
        if b.degradation.kind == "fce":
            params.update({
                "DoD_ref": b.degradation.DoD_ref,
                "f_cycle": b.degradation.f_cycle,
                "f_calendar": b.degradation.f_calendar,
                "RC": b.degradation.RC_eur_per_kwh,
                "SoH_EoL": b.degradation.SoH_EoL,
            })
        
        return params
    
    def _build_default_permissions(self) -> Dict[Tuple[str, str], int]:
        """Build default permissions matrix based on architecture settings"""
        permissions = {}
        
        # Sources: ret (retail), wh (wholesale), PV
        # Uses: load, wh (wholesale export)
        
        # Default: all sources can serve wholesale export
        for source in ["ret", "wh", "PV"]:
            permissions[("wh", source)] = 1
        
        # PV can always serve load directly
        permissions[("load", "PV")] = 1
        
        # Wholesale energy can serve load via battery
        permissions[("load", "wh")] = 1
        
        # Retail energy policy
        if self.bundle.arch.forbid_retail_to_load:
            permissions[("load", "ret")] = 0
        else:
            permissions[("load", "ret")] = 1
        
        return permissions
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get high-level scenario summary for logging/reporting"""
        b = self.bundle
        
        # Summarize ancillary services
        service_summary = {}
        for service in b.arch.ancillary_services:
            service_summary[service.name] = {
                "direction": service.direction,
                "duration_h": service.sustain_duration_hours,
                "settlement": service.settlement_method
            }
        
        return {
            "scenario_id": f"{b.arch.kind}_{self.T}steps_{b.timegrid.dt_minutes}min",
            "time_range": f"{b.timegrid.start} to {b.timegrid.end}",
            "timesteps": self.T,
            "resolution_minutes": b.timegrid.dt_minutes,
            "architecture": b.arch.kind,
            "battery_capacity_kwh": b.capacity.capacity_nominal_kwh,
            "battery_power_kw": f"{b.ratings.p_charge_max_kw}/{b.ratings.p_discharge_max_kw}",
            "soc_range": f"{b.bounds.soc_min:.1%} - {b.bounds.soc_max:.1%}",
            "policy_retail_to_load": "forbidden" if b.arch.forbid_retail_to_load else "allowed",
            "degradation_model": b.degradation.kind,
            "ancillary_services": service_summary,
            "num_services": len(b.arch.ancillary_services),
            "services_up": len(b.services_by_direction["up"]),
            "services_down": len(b.services_by_direction["down"]),
            "total_load_mwh": np.sum(b.exogenous.load_ac) / 1000,
            "total_pv_mwh": np.sum(b.exogenous.pv_dc) / 1000,
            "avg_retail_price": np.mean(b.tariffs.price_retail),
            "avg_wholesale_price": np.mean(b.tariffs.price_wholesale),
        }
    
    def validate_milp_readiness(self) -> List[str]:
        """Comprehensive validation for MILP solver compatibility"""
        errors = []
        b = self.bundle
        
        # Check time series lengths
        expected_length = self.T
        time_series_fields = [
            ("load_ac", b.exogenous.load_ac),
            ("pv_dc", b.exogenous.pv_dc),
            ("price_retail", b.tariffs.price_retail),
            ("price_wholesale", b.tariffs.price_wholesale),
            ("price_cap", b.tariffs.price_cap),
            ("cap_mask", b.tariffs.cap_mask),
        ]
        
        for name, series in time_series_fields:
            if len(series) != expected_length:
                errors.append(f"{name} length {len(series)} != expected {expected_length}")
        
        # Check ancillary service time series
        for service_name, price_array in b.tariffs.price_ancillary.items():
            if len(price_array) != expected_length:
                errors.append(f"price_ancillary[{service_name}] length != {expected_length}")
        
        # Check for NaN/inf values
        for name, series in time_series_fields:
            if np.any(~np.isfinite(series)):
                errors.append(f"{name} contains NaN or infinite values")
        
        # Check non-negativity constraints
        if np.any(b.exogenous.load_ac < 0):
            errors.append("load_ac contains negative values")
        if np.any(b.exogenous.pv_dc < 0):
            errors.append("pv_dc contains negative values")
        
        # Architecture-specific validation
        if b.arch.kind == "ac_coupled":
            required_ac = [
                (b.ratings.p_inv_pv_kw, "p_inv_pv_kw"),
                (b.ratings.p_inv_bess_kw, "p_inv_bess_kw"),
                (b.eff.eta_pv_ac_from_dc, "eta_pv_ac_from_dc"),
                (b.eff.eta_bess_dc_from_ac, "eta_bess_dc_from_ac"),
                (b.eff.eta_bess_ac_from_dc, "eta_bess_ac_from_dc"),
            ]
            for field, name in required_ac:
                if field is None:
                    errors.append(f"AC-coupled architecture requires {name}")
        
        elif b.arch.kind in ["dc_coupled", "hybrid"]:
            required_dc = [
                (b.ratings.p_inv_shared_kw, "p_inv_shared_kw"),
                (b.eff.eta_shared_dc_from_ac, "eta_shared_dc_from_ac"),
                (b.eff.eta_shared_ac_from_dc, "eta_shared_ac_from_dc"),
            ]
            for field, name in required_dc:
                if field is None:
                    errors.append(f"{b.arch.kind} architecture requires {name}")
        
        # Validate ancillary services configuration
        for service in b.arch.ancillary_services:
            # Check required price data exists
            if service.name not in b.tariffs.price_ancillary:
                errors.append(f"Missing availability price for service {service.name}")
            
            # Check activation prices for explicit settlement
            if service.settlement_method == "explicit_prices":
                if service.direction in ["up", "both"] and service.name not in b.tariffs.price_activation_up:
                    errors.append(f"Missing activation up price for service {service.name}")
                if service.direction in ["down", "both"] and service.name not in b.tariffs.price_activation_down:
                    errors.append(f"Missing activation down price for service {service.name}")
        
        # Check capacity consistency
        if b.capacity.usable_capacity_kwh is not None:
            if len(b.capacity.usable_capacity_kwh) != expected_length:
                errors.append(f"usable_capacity_kwh length != {expected_length}")
            if np.any(b.capacity.usable_capacity_kwh <= 0):
                errors.append("usable_capacity_kwh contains non-positive values")
        
        # Check power rating consistency
        if b.ratings.p_charge_max_kw <= 0 or b.ratings.p_discharge_max_kw <= 0:
            errors.append("Battery power ratings must be positive")
        
        # Check SoC bounds
        if not (0 <= b.bounds.soc_min < b.bounds.soc_max <= 1):
            errors.append("SoC bounds must satisfy 0 ≤ soc_min < soc_max ≤ 1")
        
        return errors
    
    def get_service_participation_summary(self) -> Dict[str, Any]:
        """Get summary of potential ancillary service participation"""
        b = self.bundle
        summary = {}
        
        for service in b.arch.ancillary_services:
            service_name = service.name
            
            # Calculate potential revenue
            if service_name in b.tariffs.price_ancillary:
                prices = b.tariffs.price_ancillary[service_name]
                
                # Estimate max capacity allocation based on direction
                if service.direction == "up":
                    max_capacity = b.ratings.p_discharge_max_kw
                elif service.direction == "down":
                    max_capacity = b.ratings.p_charge_max_kw
                else:  # both
                    max_capacity = min(b.ratings.p_charge_max_kw, b.ratings.p_discharge_max_kw)
                
                # Calculate potential revenue
                potential_revenue = np.sum(prices) * max_capacity * b.dt_hours
                avg_price = np.mean(prices)
                
                summary[service_name] = {
                    "direction": service.direction,
                    "max_capacity_kw": max_capacity,
                    "avg_price_eur_per_mw": avg_price,
                    "potential_revenue_eur": potential_revenue,
                    "settlement_method": service.settlement_method,
                    "sustain_duration_h": service.sustain_duration_hours,
                }
        
        return summary

# Convenience functions for common usage patterns
def load_for_milp(config_path: str) -> Dict[str, Any]:
    """One-liner to get MILP-ready parameters from config with enhanced services"""
    from .io import load_bundle_from_yaml
    
    bundle = load_bundle_from_yaml(config_path)
    interface = MILPDataInterface(bundle)
    
    # Validate before handing off
    errors = interface.validate_milp_readiness()
    if errors:
        raise ValueError(f"MILP validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return interface.to_milp_params()

def create_milp_interface(bundle: DataBundle) -> MILPDataInterface:
    """Factory function with validation"""
    interface = MILPDataInterface(bundle)
    
    errors = interface.validate_milp_readiness()
    if errors:
        raise ValueError(f"Bundle not ready for MILP:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return interface

def get_services_summary(bundle: DataBundle) -> Dict[str, Any]:
    """Get summary of configured ancillary services"""
    interface = MILPDataInterface(bundle)
    return interface.get_service_participation_summary()
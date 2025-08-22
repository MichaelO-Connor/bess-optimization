# bess/optimization/registry.py
"""
Registry mapping between schema fields and canonical notation from the mathematical formulation.
Maps DataBundle attributes to MILP parameter symbols.
"""

from typing import Dict, Any

# Symbol mapping from schema to canonical notation
SYMBOLS = {
    # ============================================================
    # TIME AND INDEXING
    # ============================================================
    "T": "Number of timesteps in optimization horizon",
    "dt_hours": "Δt - Timestep length in hours",
    "time_index": "t ∈ T = {1,...,T} - Time indices",
    "D": "Collection of daily subsets of T",
    "Omega_cap": "Ω^cap ⊆ T - Capacity obligation periods",
    
    # ============================================================
    # SETS
    # ============================================================
    "S": "S = {ret, wh, PV} - Charging source tags",
    "U": "U = {load, wh} - Discharge uses",
    "K": "K - Ancillary service products",
    "K_up": "K^↑ ⊆ K - Upward (discharge) products",
    "K_down": "K^↓ ⊆ K - Downward (charge) products",
    
    # ============================================================
    # BATTERY PARAMETERS
    # ============================================================
    "C_nom": "C_nom - Nominal energy capacity (kWh)",
    "C_t": "C_t - Usable capacity at time t (kWh)",
    "eta_c": "η_c - Charging efficiency",
    "eta_d": "η_d - Discharging efficiency",
    "delta": "δ - Self-discharge rate per period",
    "SoC_min": "SoC_min - Minimum SoC fraction",
    "SoC_max": "SoC_max - Maximum SoC fraction",
    "SoC_0": "SoC_0 - Initial SoC fraction",
    "n_cycles": "n_cycles - Max daily cycles",
    
    # ============================================================
    # EXOGENOUS SIGNALS
    # ============================================================
    "L_t": "L_t - Site load demand (kWh)",
    "a_t": "a_t - Available PV generation (kWh)",
    
    # ============================================================
    # PRICES
    # ============================================================
    "P_ret_t": "P^ret_t - Retail electricity price (€/kWh)",
    "P_wh_t": "P^wh_t - Wholesale electricity price (€/kWh)",
    "pi_k_t": "π^k_t - AS availability price (€/kW/period)",
    "u_k_up_t": "u^{k,↑}_t - AS upward activation price (€/kWh)",
    "u_k_down_t": "u^{k,↓}_t - AS downward activation price (€/kWh)",
    "pi_cap_t": "π^cap_t - Capacity payment price (€/kW/period)",
    "pi_dem": "π^dem - Demand charge (€/kW)",
    
    # ============================================================
    # ANCILLARY SERVICE PARAMETERS
    # ============================================================
    "H_k_up": "H^{k,↑}_t - Upward sustain duration (hours)",
    "H_k_down": "H^{k,↓}_t - Downward sustain duration (hours)",
    "alpha_k_up_t": "α^{k,↑}_t - Expected upward activation fraction",
    "alpha_k_down_t": "α^{k,↓}_t - Expected downward activation fraction",
    "H_cap": "H^cap - Capacity sustain duration (hours)",
    
    # ============================================================
    # POWER RATINGS
    # ============================================================
    "P_ch_max": "P̄^ch - Max charging power (kW)",
    "P_dis_max": "P̄^dis - Max discharging power (kW)",
    "P_grid_import_max": "P̄^grid - Max grid import (kW)",
    "P_grid_export_max": "P̄^grid - Max grid export (kW)",
    
    # AC-coupled specific
    "P_inv_pv": "P̄^{inv,PV} - PV inverter rating (kW)",
    "P_inv_bess": "P̄^{inv,B} - Battery inverter rating (kW)",
    
    # DC-coupled/hybrid specific
    "P_inv_shared": "P̄^inv - Shared inverter rating (kW)",
    "P_dc_pv": "P̄^{dc,PV} - PV DC/DC port rating (kW)",
    
    # ============================================================
    # EFFICIENCIES
    # ============================================================
    # AC-coupled
    "eta_pv_ac_from_dc": "η^{ac←dc}_{PV} - PV DC→AC efficiency",
    "eta_bess_dc_from_ac": "η^{dc←ac}_{B} - Battery AC→DC efficiency",
    "eta_bess_ac_from_dc": "η^{ac←dc}_{B} - Battery DC→AC efficiency",
    
    # DC-coupled/hybrid
    "eta_shared_dc_from_ac": "η^{dc←ac} - Shared PCS AC→DC efficiency",
    "eta_shared_ac_from_dc": "η^{ac←dc} - Shared PCS DC→AC efficiency",
    
    # Path efficiencies
    "eta_imp": "η_imp - Grid import to battery efficiency",
    "eta_exp": "η_exp - Battery to grid export efficiency",
    "eta_pv_to_load": "η_{PV→load} - PV to load efficiency",
    "eta_pv_to_battery": "η_{PV→batt} - PV to battery efficiency",
    
    # ============================================================
    # DEGRADATION
    # ============================================================
    "phi_c": "φ_c - Charge degradation cost (€/kWh)",
    "phi_d": "φ_d - Discharge degradation cost (€/kWh)",
    "DoD_ref": "DoD_ref - Reference depth of discharge",
    "f_cycle": "f_cycle - Cycle degradation factor",
    "f_calendar": "f_calendar - Calendar degradation factor",
    "RC": "RC - Replacement cost (€/kWh)",
    "SoH_EoL": "SoH_EoL - End-of-life SoH threshold",
    
    # ============================================================
    # DECISION VARIABLES
    # ============================================================
    # Energy flows
    "c_ret": "c^ret_t - Charging from retail (kWh)",
    "c_wh": "c^wh_t - Charging from wholesale (kWh)",
    "c_pv": "c^PV_t - Charging from PV (kWh)",
    "c_freq": "c^freq_t - Charging from frequency response (kWh)",
    "c_k": "c^k_t - Charging for service k (kWh)",
    
    "d_load": "d^load_t - Discharge to load (kWh)",
    "d_wh": "d^wh_t - Discharge to wholesale (kWh)",
    "d_freq": "d^freq_t - Discharge for frequency response (kWh)",
    "d_k": "d^k_t - Discharge for service k (kWh)",
    
    "d_load_s": "d^{load,s}_t - Tagged discharge to load (kWh)",
    "d_wh_s": "d^{wh,s}_t - Tagged discharge to wholesale (kWh)",
    
    # State variables
    "soc": "SoC_t - Total state of charge (kWh)",
    "soc_s": "SoC^s_t - Tagged state of charge (kWh)",
    
    # PV flows
    "pv_load": "PV^load_t - PV to load (kWh)",
    "pv_curt": "PV^curt_t - PV curtailed (kWh)",
    "g": "g_t - Grid import to load (kWh)",
    
    # Market participation
    "r_k": "r^k_t - Reserve capacity for service k (kW)",
    "r_up": "r^↑_t - Total upward reserves (kW)",
    "r_down": "r^↓_t - Total downward reserves (kW)",
    "Q_cap": "Q^cap - Capacity commitment (kW)",
    "dc": "DC_t - Degradation cost (€)",
    
    # Binaries
    "y_ch_ret": "y^{ch,ret}_t - Retail charging binary",
    "y_ch_wh": "y^{ch,wh}_t - Wholesale charging binary",
    "y_ch_pv": "y^{ch,pv}_t - PV charging binary",
    "y_dis_load": "y^{dis,load}_t - Load discharge binary",
    "y_dis_wh": "y^{dis,wh}_t - Wholesale discharge binary",
    "y_idle": "y^{idle}_t - Idle mode binary",
}

# Reverse mapping for looking up schema fields by symbol
SYMBOL_TO_FIELD = {symbol: field for field, symbol in SYMBOLS.items()}

def get_symbol(field_name: str) -> str:
    """
    Get the canonical symbol for a schema field.
    
    Args:
        field_name: Name of the field in the schema
        
    Returns:
        Canonical notation symbol with description
    """
    return SYMBOLS.get(field_name, f"Unknown field: {field_name}")

def get_field(symbol: str) -> str:
    """
    Get the schema field name for a canonical symbol.
    
    Args:
        symbol: Canonical notation symbol
        
    Returns:
        Schema field name
    """
    # First try direct lookup
    if symbol in SYMBOL_TO_FIELD:
        return SYMBOL_TO_FIELD[symbol]
    
    # Try to find by partial match
    for field, sym_desc in SYMBOLS.items():
        if symbol in sym_desc:
            return field
    
    return f"Unknown symbol: {symbol}"

def describe_mapping() -> str:
    """
    Generate a human-readable description of the mapping.
    
    Returns:
        Formatted string describing the schema-to-notation mapping
    """
    lines = ["Schema to Canonical Notation Mapping", "=" * 40]
    
    categories = [
        ("Time and Indexing", ["T", "dt_hours", "time_index"]),
        ("Battery Parameters", ["C_nom", "C_t", "eta_c", "eta_d", "SoC_min", "SoC_max"]),
        ("Exogenous Signals", ["L_t", "a_t"]),
        ("Prices", ["P_ret_t", "P_wh_t", "pi_k_t"]),
        ("Power Ratings", ["P_ch_max", "P_dis_max", "P_inv_pv", "P_inv_bess"]),
        ("Decision Variables", ["c_ret", "c_wh", "d_load", "d_wh", "soc", "r_k"]),
    ]
    
    for category, fields in categories:
        lines.append(f"\n{category}:")
        lines.append("-" * len(category))
        for field in fields:
            if field in SYMBOLS:
                lines.append(f"  {field:20s} → {SYMBOLS[field]}")
    
    return "\n".join(lines)

# Export the mapping for use in validation
__all__ = ["SYMBOLS", "SYMBOL_TO_FIELD", "get_symbol", "get_field", "describe_mapping"]
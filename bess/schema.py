# bess/data_model/schema.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Dict, Tuple, List
import numpy as np
import pandas as pd

# Updated to include hybrid
ArchitectureKind = Literal["ac_coupled", "dc_coupled", "hybrid"]

class TimeGrid(BaseModel):
    start: str                         # ISO, tz-aware recommended
    end: str
    dt_minutes: int = Field(gt=0)      # resolution
    tz: str = "UTC"

    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq=f"{self.dt_minutes}min", tz=self.tz, inclusive="left")

class AncillaryService(BaseModel):
    """Definition of a single ancillary service product"""
    
    name: str                                    # Service identifier (e.g., "ru", "rd", "spin", "fcr")
    direction: Literal["up", "down", "both"]     # K^↑, K^↓, or both
    sustain_duration_hours: float               # H^{k,↑}_t or H^{k,↓}_t
    
    # Optional activation parameters
    activation_fraction_up: Optional[float] = None    # α^{k,↑}_t (constant) 
    activation_fraction_down: Optional[float] = None  # α^{k,↓}_t (constant)
    
    # Settlement pricing approach
    settlement_method: Literal["availability_only", "wholesale_settled", "explicit_prices"] = "availability_only"
    
    @validator("sustain_duration_hours")
    def positive_duration(cls, v):
        assert v > 0, "Sustain duration must be positive"
        return v
    
    @validator("activation_fraction_up", "activation_fraction_down")
    def valid_activation_fraction(cls, v):
        if v is not None:
            assert 0 <= v <= 1, "Activation fraction must be in [0,1]"
        return v

class Ratings(BaseModel):
    p_charge_max_kw: float
    p_discharge_max_kw: float
    
    # AC-coupled ratings
    p_inv_pv_kw: Optional[float] = None       # P̄^{inv,PV}
    p_inv_bess_kw: Optional[float] = None     # P̄^{inv,B}
    
    # DC-coupled and hybrid ratings (shared PCS)
    p_inv_shared_kw: Optional[float] = None   # P̄^{inv}
    
    # Hybrid-specific: Optional PV DC/DC port rating
    p_dc_pv_kw: Optional[float] = None        # P̄^{dc,PV} for hybrid PV DC/DC
    
    # Grid connection limits (optional)
    p_grid_import_max_kw: Optional[float] = None  # P̄^{grid} import
    p_grid_export_max_kw: Optional[float] = None  # P̄^{grid} export
    
    @validator("*")
    def positive_ratings(cls, v):
        if v is not None:
            assert v > 0, "All power ratings must be positive"
        return v

class Efficiencies(BaseModel):
    eta_c: float                              # η_c battery charge efficiency
    eta_d: float                              # η_d battery discharge efficiency  
    self_discharge: float = 0.0               # δ self-discharge per timestep
    
    # AC-coupled efficiencies
    eta_pv_ac_from_dc: Optional[float] = None      # η^{ac←dc}_{PV}
    eta_bess_dc_from_ac: Optional[float] = None    # η^{dc←ac}_{B} 
    eta_bess_ac_from_dc: Optional[float] = None    # η^{ac←dc}_{B}
    
    # DC-coupled and hybrid efficiencies (shared PCS)
    eta_shared_dc_from_ac: Optional[float] = None  # η^{dc←ac}
    eta_shared_ac_from_dc: Optional[float] = None  # η^{ac←dc}

    @validator("eta_c", "eta_d")
    def battery_eff_bounds(cls, v):
        assert 0 < v <= 1, "Battery efficiencies must be in (0,1]"
        return v
        
    @validator("self_discharge")
    def self_discharge_bounds(cls, v):
        assert 0 <= v < 1, "Self-discharge must be in [0,1)"
        return v
        
    @validator("eta_pv_ac_from_dc", "eta_bess_dc_from_ac", "eta_bess_ac_from_dc", 
               "eta_shared_dc_from_ac", "eta_shared_ac_from_dc")
    def inverter_eff_bounds(cls, v):
        if v is not None:
            assert 0 < v <= 1, "Inverter efficiencies must be in (0,1]"
        return v

    def get_path_efficiencies(self, arch_kind: ArchitectureKind) -> Dict[str, float]:
        """Compute unified path efficiencies for MILP objective function"""
        if arch_kind == "ac_coupled":
            if self.eta_bess_dc_from_ac is None or self.eta_bess_ac_from_dc is None:
                raise ValueError("AC-coupled architecture requires BESS inverter efficiencies")
            return {
                "eta_imp": self.eta_bess_dc_from_ac,     # AC→DC on grid charging
                "eta_exp": self.eta_bess_ac_from_dc,     # DC→AC on export
                "eta_pv_to_load": self.eta_pv_ac_from_dc,    # PV DC→AC for load
                "eta_pv_to_battery": self.eta_bess_dc_from_ac,  # PV AC→DC via BESS PCS
            }
        elif arch_kind == "dc_coupled":
            if self.eta_shared_dc_from_ac is None or self.eta_shared_ac_from_dc is None:
                raise ValueError("DC-coupled architecture requires shared PCS efficiencies")
            return {
                "eta_imp": self.eta_shared_dc_from_ac,   # AC→DC on grid charging
                "eta_exp": self.eta_shared_ac_from_dc,   # DC→AC on export
                "eta_pv_to_load": self.eta_shared_ac_from_dc,  # PV DC→AC for load
                "eta_pv_to_battery": 1.0,               # Direct DC coupling, no conversion
            }
        elif arch_kind == "hybrid":
            # Hybrid uses shared PCS like DC-coupled
            if self.eta_shared_dc_from_ac is None or self.eta_shared_ac_from_dc is None:
                raise ValueError("Hybrid architecture requires shared PCS efficiencies")
            return {
                "eta_imp": self.eta_shared_dc_from_ac,   # AC→DC on grid charging
                "eta_exp": self.eta_shared_ac_from_dc,   # DC→AC on export
                "eta_pv_to_load": self.eta_shared_ac_from_dc,  # PV DC→AC for load via shared PCS
                "eta_pv_to_battery": 1.0,               # Direct DC coupling via DC/DC, no AC conversion
            }
        else:
            raise ValueError(f"Unknown architecture: {arch_kind}")

class Bounds(BaseModel):
    soc_min: float                           # SoC_min
    soc_max: float                           # SoC_max
    cycles_per_day: Optional[float] = None   # n_cycles daily throughput limit
    
    # Boundary conditions for optimization horizon
    soc_initial: Optional[float] = None      # SoC_0 (as fraction of capacity)
    soc_terminal_min: Optional[float] = None # S_T lower bound
    soc_terminal_max: Optional[float] = None # S_T upper bound  
    enforce_terminal_equality: bool = True   # SoC_T = SoC_0
    
    # Initial split factors for tagged SoC (optional, must sum to 1)
    initial_soc_split: Optional[Dict[str, float]] = None  # {'ret': 0.33, 'wh': 0.33, 'pv': 0.34}

    @validator("soc_min", "soc_max", "soc_initial", "soc_terminal_min", "soc_terminal_max")
    def soc_in_unit_interval(cls, v):
        if v is not None:
            assert 0 <= v <= 1, "SoC values must be in [0,1]"
        return v
        
    @validator("soc_max")
    def max_ge_min(cls, v, values):
        if "soc_min" in values:
            assert v >= values["soc_min"], "soc_max must be >= soc_min"
        return v
        
    @validator("cycles_per_day") 
    def positive_cycles(cls, v):
        if v is not None:
            assert v > 0, "cycles_per_day must be positive"
        return v
    
    @validator("initial_soc_split")
    def valid_split(cls, v):
        if v is not None:
            # Check it contains the right keys
            required_keys = {'ret', 'wh', 'pv'}
            if set(v.keys()) != required_keys:
                raise ValueError(f"initial_soc_split must have keys {required_keys}")
            # Check they sum to 1
            total = sum(v.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"initial_soc_split values must sum to 1, got {total}")
            # Check all non-negative
            if any(val < 0 for val in v.values()):
                raise ValueError("initial_soc_split values must be non-negative")
        return v

class Degradation(BaseModel):
    kind: Literal["throughput", "fce"] = "throughput"
    
    # Throughput-based degradation (€/kWh)
    phi_c_eur_per_kwh: float = 0.0          # φ_c charge degradation cost
    phi_d_eur_per_kwh: float = 0.0          # φ_d discharge degradation cost
    
    # FCE-based degradation  
    DoD_ref: float = 0.8                    # Reference depth of discharge
    f_cycle: float = 0.0                    # Cycle degradation factor
    f_calendar: float = 0.0                 # Calendar degradation factor  
    RC_eur_per_kwh: float = 0.0            # Replacement cost €/kWh
    SoH_EoL: float = 0.8                   # End-of-life SoH threshold

    @validator("DoD_ref", "SoH_EoL")
    def unit_fractions(cls, v):
        assert 0 < v <= 1, "DoD_ref and SoH_EoL must be in (0,1]"
        return v

class Tariffs(BaseModel):
    # Energy prices (€/kWh) - time series aligned to time grid
    price_retail: np.ndarray                # P^{ret}_t
    price_wholesale: np.ndarray             # P^{wh}_t
    
    # Ancillary service availability prices (€/kW) - time series by service name
    price_ancillary: Dict[str, np.ndarray] = {}    # π^k_t for each service k∈K
    
    # Ancillary service activation settlement prices (€/kWh) - time series by service name  
    price_activation_up: Dict[str, np.ndarray] = {}     # u^{k,↑}_t for each service k∈K^↑
    price_activation_down: Dict[str, np.ndarray] = {}   # u^{k,↓}_t for each service k∈K^↓
    
    # Capacity market (€/kW per period) - time series
    price_cap: np.ndarray                   # π^{cap}_t
    cap_mask: np.ndarray                    # Ω^{cap} boolean obligation hours
    
    # Scalar prices  
    price_demand: float = 0.0               # π^{dem} demand charge €/kW
    
    # Service parameters
    cap_duration_hours: float = 4.0         # H^{cap} required discharge duration
    
    # Optional: time-varying activation fractions (defaults in AncillaryService if constant)
    activation_up: Dict[str, Optional[np.ndarray]] = {}     # α^{k,↑}_t by service
    activation_down: Dict[str, Optional[np.ndarray]] = {}   # α^{k,↓}_t by service

class Exogenous(BaseModel):
    load_ac: np.ndarray                     # L_t site load (kWh per timestep)
    pv_dc: np.ndarray                       # a_t PV generation (kWh per timestep)

class Capacity(BaseModel):
    capacity_nominal_kwh: float             # C_nom nameplate capacity
    usable_capacity_kwh: Optional[np.ndarray] = None  # C_t time-varying usable capacity

    @validator("capacity_nominal_kwh")
    def positive_capacity(cls, v):
        assert v > 0, "Nominal capacity must be positive"
        return v

class Architecture(BaseModel):
    kind: ArchitectureKind                   # Now includes "hybrid"
    forbid_retail_to_load: bool = True      # Policy: no retail energy to load via battery
    
    # Ancillary services configuration
    ancillary_services: List[AncillaryService] = []
    
    # Explicit permissions matrix A_{u,s}: (use, source) -> {0,1}
    permissions: Optional[Dict[Tuple[str, str], int]] = None
    
    # Grid connection limits (alternative to ratings specification)
    import_limit_kw: Optional[float] = None  # P̄^{grid} maximum import power  
    export_limit_kw: Optional[float] = None  # P̄^{grid} maximum export power
    
    def get_services_by_direction(self) -> Dict[str, List[str]]:
        """Get service names grouped by direction for constraint building"""
        services_up = []
        services_down = []
        
        for service in self.ancillary_services:
            if service.direction in ["up", "both"]:
                services_up.append(service.name)
            if service.direction in ["down", "both"]:
                services_down.append(service.name)
        
        return {
            "up": services_up,      # K^↑
            "down": services_down   # K^↓
        }
    
    def get_service_by_name(self, name: str) -> Optional[AncillaryService]:
        """Get service configuration by name"""
        for service in self.ancillary_services:
            if service.name == name:
                return service
        return None

class DataBundle(BaseModel):
    timegrid: TimeGrid
    arch: Architecture
    ratings: Ratings
    eff: Efficiencies
    bounds: Bounds
    degradation: Degradation
    tariffs: Tariffs
    exogenous: Exogenous
    capacity: Capacity

    @validator("tariffs", "exogenous", pre=True)
    def ensure_numpy_arrays(cls, v):
        """Convert lists/Series to numpy arrays"""
        if isinstance(v, dict):
            result = {}
            for k, val in v.items():
                if hasattr(val, "__len__") and not isinstance(val, (str, np.ndarray)):
                    result[k] = np.asarray(val)
                else:
                    result[k] = val
            return result
        return v

    def validate_lengths(self):
        """Ensure all time series have consistent length"""
        T = len(self.timegrid.index())
        
        # Core time series
        time_series = {
            "price_retail": self.tariffs.price_retail,
            "price_wholesale": self.tariffs.price_wholesale, 
            "price_cap": self.tariffs.price_cap,
            "cap_mask": self.tariffs.cap_mask,
            "load_ac": self.exogenous.load_ac,
            "pv_dc": self.exogenous.pv_dc,
        }
        
        for name, arr in time_series.items():
            if len(arr) != T:
                raise ValueError(f"{name} length {len(arr)} != expected {T}")
        
        # Ancillary service time series
        for service_name, price_array in self.tariffs.price_ancillary.items():
            if len(price_array) != T:
                raise ValueError(f"price_ancillary[{service_name}] length != {T}")
        
        for service_name, price_array in self.tariffs.price_activation_up.items():
            if len(price_array) != T:
                raise ValueError(f"price_activation_up[{service_name}] length != {T}")
        
        for service_name, price_array in self.tariffs.price_activation_down.items():
            if len(price_array) != T:
                raise ValueError(f"price_activation_down[{service_name}] length != {T}")
        
        # Optional time series
        if self.capacity.usable_capacity_kwh is not None:
            if len(self.capacity.usable_capacity_kwh) != T:
                raise ValueError(f"usable_capacity_kwh length != {T}")
        
        for service_name, activation_array in self.tariffs.activation_up.items():
            if activation_array is not None and len(activation_array) != T:
                raise ValueError(f"activation_up[{service_name}] length != {T}")
        
        for service_name, activation_array in self.tariffs.activation_down.items():
            if activation_array is not None and len(activation_array) != T:
                raise ValueError(f"activation_down[{service_name}] length != {T}")

    def validate_architecture_requirements(self):
        """Ensure architecture-specific parameters are present"""
        if self.arch.kind == "ac_coupled":
            required_fields = [
                (self.ratings.p_inv_pv_kw, "p_inv_pv_kw"),
                (self.ratings.p_inv_bess_kw, "p_inv_bess_kw"),
                (self.eff.eta_pv_ac_from_dc, "eta_pv_ac_from_dc"),
                (self.eff.eta_bess_dc_from_ac, "eta_bess_dc_from_ac"),
                (self.eff.eta_bess_ac_from_dc, "eta_bess_ac_from_dc"),
            ]
            missing = [name for field, name in required_fields if field is None]
            if missing:
                raise ValueError(f"AC-coupled architecture requires: {missing}")
                
        elif self.arch.kind in ["dc_coupled", "hybrid"]:
            # Both DC-coupled and hybrid need shared PCS
            required_fields = [
                (self.ratings.p_inv_shared_kw, "p_inv_shared_kw"),
                (self.eff.eta_shared_dc_from_ac, "eta_shared_dc_from_ac"),
                (self.eff.eta_shared_ac_from_dc, "eta_shared_ac_from_dc"),
            ]
            missing = [name for field, name in required_fields if field is None]
            if missing:
                raise ValueError(f"{self.arch.kind} architecture requires: {missing}")
            
            # Hybrid optionally has PV DC/DC rating
            if self.arch.kind == "hybrid" and self.ratings.p_dc_pv_kw is not None:
                if self.ratings.p_dc_pv_kw <= 0:
                    raise ValueError("p_dc_pv_kw must be positive if specified")
        
        # Validate ancillary services have corresponding price data
        for service in self.arch.ancillary_services:
            if service.name not in self.tariffs.price_ancillary:
                raise ValueError(f"Missing availability price for service {service.name}")
            
            # Check activation prices if settlement method requires them
            if service.settlement_method == "explicit_prices":
                if service.direction in ["up", "both"] and service.name not in self.tariffs.price_activation_up:
                    raise ValueError(f"Missing activation up price for service {service.name}")
                if service.direction in ["down", "both"] and service.name not in self.tariffs.price_activation_down:
                    raise ValueError(f"Missing activation down price for service {service.name}")

    @property
    def path_efficiencies(self) -> Dict[str, float]:
        """Get architecture-specific path efficiencies for MILP"""
        return self.eff.get_path_efficiencies(self.arch.kind)
    
    @property  
    def dt_hours(self) -> float:
        """Timestep length in hours (Δt from MILP)"""
        return self.timegrid.dt_minutes / 60.0
        
    @property
    def T(self) -> int:
        """Number of timesteps in optimization horizon"""
        return len(self.timegrid.index())
    
    @property
    def services_by_direction(self) -> Dict[str, List[str]]:
        """Get ancillary services grouped by direction (K^↑, K^↓)"""
        return self.arch.get_services_by_direction()

    def __post_init__(self):
        """Run all validations after construction"""
        self.validate_lengths()
        self.validate_architecture_requirements()
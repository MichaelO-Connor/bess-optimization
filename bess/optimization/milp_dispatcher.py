# bess/optimization/milp_dispatcher_fixed.py
"""
Fixed MILP Dispatcher implementing Section 8 formulation from the PDF.
Supports AC-coupled, DC-coupled, and hybrid architectures with 
generic ancillary services and multi-revenue streams.

FIXES:
1. Terminal SoC uses correct capacity index
2. Initial SoC not arbitrarily split across tags
3. Daily throughput uses time-varying capacity
4. Hybrid architecture fully implemented
5. Capacity deliverability includes t=0
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResults:
    """Container for optimization results"""
    objective_value: float
    solve_time: float
    status: str
    
    # Energy flows
    charge_retail: np.ndarray      # c^ret_t
    charge_wholesale: np.ndarray   # c^wh_t
    charge_pv: np.ndarray          # c^PV_t
    charge_freq: np.ndarray        # c^freq_t
    discharge_load: np.ndarray     # d^load_t
    discharge_wholesale: np.ndarray # d^wh_t
    discharge_freq: np.ndarray     # d^freq_t
    
    # Tagged discharge flows
    discharge_load_by_source: Dict[str, np.ndarray]  # d^{load,s}_t
    discharge_wh_by_source: Dict[str, np.ndarray]    # d^{wh,s}_t
    
    # State variables
    soc_total: np.ndarray          # SoC_t
    soc_by_source: Dict[str, np.ndarray]  # SoC^s_t
    
    # PV flows
    pv_to_load: np.ndarray         # PV^load_t
    pv_curtailed: np.ndarray       # PV^curt_t
    grid_to_load: np.ndarray       # g_t
    
    # Market participation
    reserve_capacity: Dict[str, np.ndarray]  # r^k_t
    activation_up: Dict[str, np.ndarray]     # d^k_t
    activation_down: Dict[str, np.ndarray]   # c^k_t
    capacity_commitment: float      # Q^cap
    
    # Binaries
    action_binaries: Dict[str, np.ndarray]
    
    # Financial metrics
    energy_cost: float
    degradation_cost: float
    ancillary_revenue: float
    capacity_revenue: float
    total_revenue: float

class MILPDispatcher:
    """
    Main MILP optimization dispatcher implementing Section 8 formulation.
    Handles all architectures including hybrid, and revenue streams.
    """
    
    def __init__(self, params: Dict[str, Any], solver: str = "gurobi"):
        """
        Initialize MILP dispatcher with parameters from MILPDataInterface.
        
        Args:
            params: Dictionary of MILP parameters from MILPDataInterface.to_milp_params()
            solver: Solver backend ("gurobi" or "cplex")
        """
        self.params = params
        self.solver = solver
        self.T = params["T"]
        self.dt = params["dt_hours"]
        self.architecture = params["architecture"]
        
        # Initialize model
        self.model = gp.Model("BESS_Optimization")
        self.model.setParam('OutputFlag', 0)  # Suppress output
        
        # Variable containers
        self.vars = {}
        
        # Build the optimization problem
        self._build_variables()
        self._build_objective()
        self._build_common_constraints()
        self._build_architecture_constraints()
        
    def _build_variables(self):
        """Create all decision variables per Section 8"""
        T = self.T
        K = self.params["K"]  # All ancillary services
        
        # ========== Energy Flows ==========
        # Charging by source (c^s_t)
        self.vars['c_ret'] = self.model.addVars(T, lb=0, name="c_ret")
        self.vars['c_wh'] = self.model.addVars(T, lb=0, name="c_wh")
        self.vars['c_pv'] = self.model.addVars(T, lb=0, name="c_pv")
        
        # Frequency response aggregates
        self.vars['c_freq'] = self.model.addVars(T, lb=0, name="c_freq")
        self.vars['d_freq'] = self.model.addVars(T, lb=0, name="d_freq")
        
        # Service-specific activation energy
        self.vars['c_k'] = self.model.addVars(K, T, lb=0, name="c_k")
        self.vars['d_k'] = self.model.addVars(K, T, lb=0, name="d_k")
        
        # Tagged discharge flows (d^{u,s}_t)
        sources = ['ret', 'wh', 'pv']
        self.vars['d_load_s'] = self.model.addVars(sources, T, lb=0, name="d_load_s")
        self.vars['d_wh_s'] = self.model.addVars(sources, T, lb=0, name="d_wh_s")
        
        # Total discharge by use
        self.vars['d_load'] = self.model.addVars(T, lb=0, name="d_load")
        self.vars['d_wh'] = self.model.addVars(T, lb=0, name="d_wh")
        
        # ========== State Variables ==========
        # Tagged SoC (SoC^s_t)
        self.vars['soc_s'] = self.model.addVars(sources, T, lb=0, name="soc_s")
        # Total SoC
        self.vars['soc'] = self.model.addVars(T, lb=0, name="soc")
        
        # ========== PV Dispatch ==========
        self.vars['pv_load'] = self.model.addVars(T, lb=0, name="pv_load")
        self.vars['pv_curt'] = self.model.addVars(T, lb=0, name="pv_curt")
        
        # Grid import for load
        self.vars['g'] = self.model.addVars(T, lb=0, name="g")
        
        # ========== Market Participation ==========
        # Reserve capacity by service (r^k_t)
        self.vars['r_k'] = self.model.addVars(K, T, lb=0, name="r_k")
        
        # Aggregate reserves by direction
        self.vars['r_up'] = self.model.addVars(T, lb=0, name="r_up")
        self.vars['r_down'] = self.model.addVars(T, lb=0, name="r_down")
        
        # Capacity commitment
        self.vars['Q_cap'] = self.model.addVar(lb=0, name="Q_cap")
        
        # ========== Action Selection Binaries ==========
        self.vars['y_ch_ret'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_ch_ret")
        self.vars['y_ch_wh'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_ch_wh")
        self.vars['y_ch_pv'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_ch_pv")
        self.vars['y_dis_load'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_dis_load")
        self.vars['y_dis_wh'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_dis_wh")
        self.vars['y_idle'] = self.model.addVars(T, vtype=GRB.BINARY, name="y_idle")
        
        # ========== Degradation ==========
        if self.params['degradation_kind'] == 'throughput':
            self.vars['dc'] = self.model.addVars(T, lb=0, name="dc")
        
    def _build_objective(self):
        """Build objective function per equation (7) in Section 8"""
        T = self.T
        dt = self.dt
        p = self.params
        
        obj = gp.LinExpr()
        
        for t in range(T):
            # Retail import cost for residual load
            obj += (p['L_t'][t] - self.vars['d_load'][t] - self.vars['pv_load'][t]) * p['P_ret_t'][t]
            
            # Retail charging cost
            obj += self.vars['c_ret'][t] * p['P_ret_t'][t]
            
            # Wholesale charging cost
            obj += self.vars['c_wh'][t] * p['P_wh_t'][t]
            
            # Wholesale export revenue (negative cost)
            obj -= self.vars['d_wh'][t] * p['P_wh_t'][t]
            
            # Degradation cost
            if p['degradation_kind'] == 'throughput':
                self.vars['dc'][t] = (
                    p['phi_c'] * (self.vars['c_ret'][t] + self.vars['c_wh'][t] + 
                                  self.vars['c_pv'][t] + self.vars['c_freq'][t]) +
                    p['phi_d'] * (self.vars['d_load'][t] + self.vars['d_wh'][t] + 
                                  self.vars['d_freq'][t])
                )
                obj += self.vars['dc'][t]
            
            # Ancillary service availability revenue (negative cost)
            for k in p['K']:
                obj -= p['pi_k_t'][k][t] * self.vars['r_k'][k, t]
                
                # Activation settlement
                obj += self.vars['c_k'][k, t] * p['u_k_down_t'][k][t]
                obj -= self.vars['d_k'][k, t] * p['u_k_up_t'][k][t]
        
        # Capacity payments (negative cost)
        cap_mask = p['Omega_cap']
        for t in range(T):
            if cap_mask[t]:
                obj -= p['pi_cap_t'][t] * self.vars['Q_cap']
        
        self.model.setObjective(obj, GRB.MINIMIZE)
        
    def _build_common_constraints(self):
        """Build common constraints (equations 1-14) - FIXED VERSION"""
        T = self.T
        dt = self.dt
        p = self.params
        
        # ========== (1) Boundary Conditions - FIXED ==========
        if p.get('enforce_terminal_equality', True):
            # FIX 1: Use correct capacity index (T-1) for terminal SoC
            # Option A: Fraction-based (maintains same fraction of capacity)
            self.model.addConstr(
                self.vars['soc'][T-1] / p['C_t'][T-1] == p['SoC_0'],
                name="terminal_soc_fraction"
            )
            # Option B (alternative): Absolute kWh if capacity is stable
            # self.model.addConstr(
            #     self.vars['soc'][T-1] == p['SoC_0'] * p['C_t'][T-1],
            #     name="terminal_soc"
            # )
        else:
            # Band constraints with correct indices
            if 'SoC_T_min' in p:
                self.model.addConstr(
                    self.vars['soc'][T-1] >= p['SoC_T_min'] * p['C_t'][T-1],
                    name="terminal_soc_min"
                )
            if 'SoC_T_max' in p:
                self.model.addConstr(
                    self.vars['soc'][T-1] <= p['SoC_T_max'] * p['C_t'][T-1],
                    name="terminal_soc_max"
                )
        
        # ========== (3) Aggregation Constraints ==========
        sources = ['ret', 'wh', 'pv']
        
        for t in range(T):
            # (3a) Total SoC
            self.model.addConstr(
                self.vars['soc'][t] == gp.quicksum(self.vars['soc_s'][s, t] for s in sources),
                name=f"soc_total_{t}"
            )
            
            # (3b) Total discharge to load
            self.model.addConstr(
                self.vars['d_load'][t] == gp.quicksum(self.vars['d_load_s'][s, t] for s in sources),
                name=f"d_load_total_{t}"
            )
            
            # (3c) Total discharge to wholesale
            self.model.addConstr(
                self.vars['d_wh'][t] == gp.quicksum(self.vars['d_wh_s'][s, t] for s in sources),
                name=f"d_wh_total_{t}"
            )
            
            # (3d) Frequency response aggregation
            self.model.addConstr(
                self.vars['c_freq'][t] == gp.quicksum(self.vars['c_k'][k, t] for k in p['K']),
                name=f"c_freq_total_{t}"
            )
            self.model.addConstr(
                self.vars['d_freq'][t] == gp.quicksum(self.vars['d_k'][k, t] for k in p['K']),
                name=f"d_freq_total_{t}"
            )
        
        # ========== (4) AC-side Load Balance ==========
        for t in range(T):
            self.model.addConstr(
                p['L_t'][t] == self.vars['pv_load'][t] + self.vars['d_load'][t] + self.vars['g'][t],
                name=f"load_balance_{t}"
            )
        
        # ========== (5) Action Exclusivity ==========
        for t in range(T):
            # At most one mode active
            binaries = [
                self.vars['y_ch_ret'][t],
                self.vars['y_ch_wh'][t],
                self.vars['y_ch_pv'][t],
                self.vars['y_dis_load'][t],
                self.vars['y_dis_wh'][t],
                self.vars['y_idle'][t]
            ]
            # (5b) Exactly one mode active
            self.model.addConstr(
                gp.quicksum(binaries) == 1,
                name=f"action_completeness_{t}"
            )
        
        # ========== (6) Action-specific Power Gating ==========
        for t in range(T):
            self.model.addConstr(
                self.vars['c_ret'][t] <= p['P_ch_max'] * dt * self.vars['y_ch_ret'][t],
                name=f"gate_c_ret_{t}"
            )
            self.model.addConstr(
                self.vars['c_wh'][t] <= p['P_ch_max'] * dt * self.vars['y_ch_wh'][t],
                name=f"gate_c_wh_{t}"
            )
            self.model.addConstr(
                self.vars['c_pv'][t] <= p['P_ch_max'] * dt * self.vars['y_ch_pv'][t],
                name=f"gate_c_pv_{t}"
            )
            self.model.addConstr(
                self.vars['d_load'][t] <= p['P_dis_max'] * dt * self.vars['y_dis_load'][t],
                name=f"gate_d_load_{t}"
            )
            self.model.addConstr(
                self.vars['d_wh'][t] <= p['P_dis_max'] * dt * self.vars['y_dis_wh'][t],
                name=f"gate_d_wh_{t}"
            )
        
        # ========== (7) Policy Constraints ==========
        for t in range(T):
            # (7a) Wholesale can't serve load
            self.model.addConstr(self.vars['d_load_s']['wh', t] == 0, name=f"no_wh_to_load_{t}")
            # (7b) Retail can't be exported
            self.model.addConstr(self.vars['d_wh_s']['ret', t] == 0, name=f"no_ret_export_{t}")
            # (7c) PV can't be exported via battery
            self.model.addConstr(self.vars['d_wh_s']['pv', t] == 0, name=f"no_pv_export_{t}")
        
        # ========== (8) Activation Bounds ==========
        for k in p['K']:
            for t in range(T):
                # Upward activation
                self.model.addConstr(
                    self.vars['d_k'][k, t] <= p['alpha_k_up_t'][k][t] * self.vars['r_k'][k, t] * dt,
                    name=f"activation_up_{k}_{t}"
                )
                # Downward activation
                self.model.addConstr(
                    self.vars['c_k'][k, t] <= p['alpha_k_down_t'][k][t] * self.vars['r_k'][k, t] * dt,
                    name=f"activation_down_{k}_{t}"
                )
        
        # ========== (9) Reserve Direction Sums ==========
        for t in range(T):
            self.model.addConstr(
                self.vars['r_up'][t] == gp.quicksum(
                    self.vars['r_k'][k, t] for k in p['K_up']
                ),
                name=f"r_up_total_{t}"
            )
            self.model.addConstr(
                self.vars['r_down'][t] == gp.quicksum(
                    self.vars['r_k'][k, t] for k in p['K_down']
                ),
                name=f"r_down_total_{t}"
            )
        
        # ========== (10) SoC Dynamics with FR Energy - FIXED ==========
        # FIX 2: Don't pre-split initial SoC arbitrarily
        # Add initial total SoC constraint (do once)
        self.model.addConstr(
            self.vars['soc'][0] == p['SoC_0'] * p['C_t'][0],
            name="initial_soc_total"
        )
        
        # Handle initial split if provided
        if 'initial_soc_split' in p and p['initial_soc_split'] is not None:
            # Use provided split factors
            for s in sources:
                self.model.addConstr(
                    self.vars['soc_s'][s, 0] == p['SoC_0'] * p['C_t'][0] * p['initial_soc_split'][s],
                    name=f"initial_soc_{s}"
                )
        
        for s in sources:
            for t in range(T):
                if t == 0:
                    soc_prev = self.vars['soc_s'][s, 0]
                else:
                    soc_prev = self.vars['soc_s'][s, t-1]
                
                # Charging energy
                if s == 'ret':
                    charge = self.vars['c_ret'][t]
                elif s == 'wh':
                    charge = self.vars['c_wh'][t] + self.vars['c_freq'][t]
                else:  # pv
                    charge = self.vars['c_pv'][t]
                
                # Discharging energy
                if s == 'wh':
                    discharge = (self.vars['d_load_s'][s, t] + self.vars['d_wh_s'][s, t] + 
                                self.vars['d_freq'][t])
                else:
                    discharge = self.vars['d_load_s'][s, t] + self.vars['d_wh_s'][s, t]
                
                # SoC dynamics equation (skip for t=0 as we handle it differently)
                if t > 0:
                    self.model.addConstr(
                        self.vars['soc_s'][s, t] == 
                        soc_prev * (1 - p['delta']) + 
                        p['eta_c'] * charge - 
                        discharge / p['eta_d'],
                        name=f"soc_dynamics_{s}_{t}"
                    )
        
        # ========== (11) SoC Bounds ==========
        for t in range(T):
            self.model.addConstr(
                self.vars['soc'][t] >= p['SoC_min'] * p['C_t'][t],
                name=f"soc_min_{t}"
            )
            self.model.addConstr(
                self.vars['soc'][t] <= p['SoC_max'] * p['C_t'][t],
                name=f"soc_max_{t}"
            )
        
        # ========== (12) Reserve Deliverability ==========
        # FIX: Start from t=0 for proper coverage
        for t in range(T):
            # (12a) Upward deliverability 
            if t == 0:
                # Use initial SoC for t=0
                soc_for_reserve = p['SoC_0'] * p['C_t'][0]
                self.model.addConstr(
                    p['eta_d'] * soc_for_reserve >= 
                    gp.quicksum(p['H_k_up'][k] * self.vars['r_k'][k, t] 
                               for k in p['K_up'] if k in p['H_k_up']),
                    name=f"reserve_up_energy_init"
                )
            else:
                for k in p['K_up']:
                    if k in p['H_k_up']:
                        self.model.addConstr(
                            p['eta_d'] * self.vars['soc'][t-1] >= 
                            p['H_k_up'][k] * self.vars['r_k'][k, t],
                            name=f"reserve_up_energy_{k}_{t}"
                        )
            
            # (12b) Downward deliverability
            if t == 0:
                soc_for_reserve = p['SoC_0'] * p['C_t'][0]
                self.model.addConstr(
                    p['SoC_max'] * p['C_t'][t] - soc_for_reserve >=
                    gp.quicksum((p['H_k_down'][k] / p['eta_c']) * self.vars['r_k'][k, t]
                               for k in p['K_down'] if k in p['H_k_down']),
                    name=f"reserve_down_headroom_init"
                )
            else:
                for k in p['K_down']:
                    if k in p['H_k_down']:
                        self.model.addConstr(
                            p['SoC_max'] * p['C_t'][t] - self.vars['soc'][t-1] >=
                            (p['H_k_down'][k] / p['eta_c']) * self.vars['r_k'][k, t],
                            name=f"reserve_down_headroom_{k}_{t}"
                        )
        
        # ========== (13) Capacity Deliverability - FIXED ==========
        # (13a) Power constraint
        self.model.addConstr(
            self.vars['Q_cap'] <= p['P_dis_max'],
            name="capacity_power_limit"
        )
        
        # (13b) Energy constraint for capacity periods
        # FIX: Include t=0 in the loop
        cap_mask = p['Omega_cap']
        for t in range(T):  # Start from 0, not 1
            if cap_mask[t]:
                if t == 0:
                    # Use initial SoC for t=0
                    soc_for_cap = p['SoC_0'] * p['C_t'][0]
                    self.model.addConstr(
                        p['eta_d'] * soc_for_cap >= p['H_cap'] * self.vars['Q_cap'],
                        name=f"capacity_energy_init"
                    )
                else:
                    self.model.addConstr(
                        p['eta_d'] * self.vars['soc'][t-1] >= p['H_cap'] * self.vars['Q_cap'],
                        name=f"capacity_energy_{t}"
                    )
        
        # ========== (14) Daily Throughput Caps (Optional) - FIXED ==========
        # FIX 3: Use time-varying capacity instead of nominal
        if p.get('n_cycles'):
            # Create daily subsets
            days = []
            steps_per_day = int(24 / dt)
            for d in range(0, T, steps_per_day):
                days.append(list(range(d, min(d + steps_per_day, T))))
            
            for d_idx, day_steps in enumerate(days):
                # Use minimum capacity over the day (conservative approach)
                # Alternatives: mean, end-of-day, or max
                day_capacity = min(p['C_t'][t] for t in day_steps)
                
                # (14a) Discharge limit
                self.model.addConstr(
                    gp.quicksum(
                        self.vars['d_load'][t] + self.vars['d_wh'][t] + self.vars['d_freq'][t]
                        for t in day_steps
                    ) <= (p['SoC_max'] - p['SoC_min']) * day_capacity * p['n_cycles'],
                    name=f"daily_discharge_limit_{d_idx}"
                )
                
                # (14b) Charge limit
                self.model.addConstr(
                    gp.quicksum(
                        self.vars['c_ret'][t] + self.vars['c_wh'][t] + 
                        self.vars['c_pv'][t] + self.vars['c_freq'][t]
                        for t in day_steps
                    ) <= (p['SoC_max'] - p['SoC_min']) * day_capacity * p['n_cycles'] / p['eta_c'],
                    name=f"daily_charge_limit_{d_idx}"
                )
    
    def _build_architecture_constraints(self):
        """Build architecture-specific constraints based on system type"""
        if self.architecture == "ac_coupled":
            self._build_ac_coupled_constraints()
        elif self.architecture == "dc_coupled":
            self._build_dc_coupled_constraints()
        elif self.architecture == "hybrid":
            self._build_hybrid_constraints()  # FIX 4: Add hybrid support
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def _build_ac_coupled_constraints(self):
        """Build AC-coupled specific constraints (equations 15-19)"""
        T = self.T
        dt = self.dt
        p = self.params
        
        for t in range(T):
            # ========== (15) PV Energy Split ==========
            # (15a) PV balance
            self.model.addConstr(
                p['eta_pv_ac_from_dc'] * (p['a_t'][t] - self.vars['pv_curt'][t]) ==
                self.vars['pv_load'][t] + self.vars['c_pv'][t] / p['eta_bess_dc_from_ac'],
                name=f"pv_balance_{t}"
            )
            
            # (15b) PV to load limit
            self.model.addConstr(
                self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"pv_load_limit_{t}"
            )
            
            # (15c) Combined load limit
            self.model.addConstr(
                self.vars['d_load'][t] + self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"combined_load_limit_{t}"
            )
            
            # ========== (16) PV Inverter Limit ==========
            self.model.addConstr(
                self.vars['pv_load'][t] + self.vars['c_pv'][t] / p['eta_bess_dc_from_ac'] <=
                p['P_inv_pv'] * dt,
                name=f"pv_inverter_limit_{t}"
            )
            
            # ========== (17) Battery PCS Charging ==========
            self.model.addConstr(
                (self.vars['c_ret'][t] + self.vars['c_wh'][t] + 
                 self.vars['c_pv'][t] + self.vars['c_freq'][t]) / p['eta_bess_dc_from_ac'] +
                self.vars['r_down'][t] * dt <=
                p['P_inv_bess'] * dt,
                name=f"bess_pcs_charge_{t}"
            )
            
            # ========== (18) Battery PCS Discharging ==========
            self.model.addConstr(
                p['eta_bess_ac_from_dc'] * (self.vars['d_load'][t] + self.vars['d_wh'][t] + 
                                           self.vars['d_freq'][t]) / p['eta_d'] +
                self.vars['r_up'][t] * dt <=
                p['P_inv_bess'] * dt,
                name=f"bess_pcs_discharge_{t}"
            )
            
            # ========== (18a) Reserve Stacking Cap ==========
            self.model.addConstr(
                gp.quicksum(self.vars['r_k'][k, t] for k in p['K']) <= p['P_inv_bess'],
                name=f"reserve_stack_limit_{t}"
            )
            
            # ========== (19) POI Import/Export Limits ==========
            if 'P_grid_import_max' in p:
                # (19a) Import limit
                self.model.addConstr(
                    self.vars['g'][t] + 
                    (self.vars['c_ret'][t] + self.vars['c_wh'][t] + self.vars['c_freq'][t]) / 
                    p['eta_bess_dc_from_ac'] <=
                    p['P_grid_import_max'] * dt,
                    name=f"poi_import_{t}"
                )
            
            if 'P_grid_export_max' in p:
                # (19b) Export limit
                self.model.addConstr(
                    p['eta_bess_ac_from_dc'] * (self.vars['d_wh'][t] + self.vars['d_freq'][t]) <=
                    p['P_grid_export_max'] * dt,
                    name=f"poi_export_{t}"
                )
    
    def _build_dc_coupled_constraints(self):
        """Build DC-coupled specific constraints (equations 20-21)"""
        T = self.T
        dt = self.dt
        p = self.params
        
        for t in range(T):
            # ========== (20) PV DC-bus Balance ==========
            # (20a) PV allocation
            self.model.addConstr(
                p['a_t'][t] == 
                self.vars['c_pv'][t] + 
                self.vars['pv_load'][t] / p['eta_shared_ac_from_dc'] +
                self.vars['pv_curt'][t],
                name=f"pv_dc_balance_{t}"
            )
            
            # (20b) PV to load limit
            self.model.addConstr(
                self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"pv_load_limit_{t}"
            )
            
            # (20c) Combined load limit
            self.model.addConstr(
                self.vars['d_load'][t] + self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"combined_load_limit_{t}"
            )
            
            # ========== (21) Shared PCS Throughput ==========
            self.model.addConstr(
                (self.vars['c_ret'][t] + self.vars['c_wh'][t] + self.vars['c_freq'][t]) / 
                p['eta_shared_dc_from_ac'] +
                self.vars['pv_load'][t] +
                p['eta_shared_ac_from_dc'] * (self.vars['d_load'][t] + self.vars['d_wh'][t] + 
                                             self.vars['d_freq'][t]) +
                self.vars['r_up'][t] * dt + self.vars['r_down'][t] * dt <=
                p['P_inv_shared'] * dt,
                name=f"shared_pcs_limit_{t}"
            )
            
            # ========== (21c) Reserve Stacking Cap ==========
            self.model.addConstr(
                gp.quicksum(self.vars['r_k'][k, t] for k in p['K']) <= p['P_inv_shared'],
                name=f"reserve_stack_limit_{t}"
            )
            
            # ========== (21a,b) POI Import/Export Limits ==========
            if 'P_grid_import_max' in p:
                self.model.addConstr(
                    self.vars['g'][t] + 
                    (self.vars['c_ret'][t] + self.vars['c_wh'][t] + self.vars['c_freq'][t]) / 
                    p['eta_shared_dc_from_ac'] <=
                    p['P_grid_import_max'] * dt,
                    name=f"poi_import_{t}"
                )
            
            if 'P_grid_export_max' in p:
                self.model.addConstr(
                    p['eta_shared_ac_from_dc'] * (self.vars['d_wh'][t] + self.vars['d_freq'][t]) <=
                    p['P_grid_export_max'] * dt,
                    name=f"poi_export_{t}"
                )
    
    def _build_hybrid_constraints(self):
        """
        FIX 4: Build Hybrid architecture constraints (equations 22-25)
        Hybrid = Shared PCS + PV DC/DC (Bidirectional PCS)
        """
        T = self.T
        dt = self.dt
        p = self.params
        
        # Check for hybrid-specific parameters
        if 'eta_shared_dc_from_ac' not in p or 'eta_shared_ac_from_dc' not in p:
            raise ValueError("Hybrid architecture requires shared PCS efficiencies")
        if 'P_inv_shared' not in p:
            raise ValueError("Hybrid architecture requires P_inv_shared rating")
        
        for t in range(T):
            # ========== (22) PV DC-bus Balance (no PV export) ==========
            # (22a) PV allocation at DC bus
            self.model.addConstr(
                p['a_t'][t] == 
                self.vars['c_pv'][t] +  # Direct DC charging
                self.vars['pv_load'][t] / p['eta_shared_ac_from_dc'] +  # PV to AC load
                self.vars['pv_curt'][t],  # Curtailed
                name=f"hybrid_pv_dc_balance_{t}"
            )
            
            # (22b) PV to load limit
            self.model.addConstr(
                self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"hybrid_pv_load_limit_{t}"
            )
            
            # (22c) Combined load limit
            self.model.addConstr(
                self.vars['d_load'][t] + self.vars['pv_load'][t] <= p['L_t'][t],
                name=f"hybrid_combined_load_limit_{t}"
            )
            
            # ========== (23) Shared PCS Throughput (with frequency response) ==========
            # All AC exchanges share the PCS capacity
            self.model.addConstr(
                # AC→DC charging (including reg-down)
                (self.vars['c_ret'][t] + self.vars['c_wh'][t] + self.vars['c_freq'][t]) / 
                p['eta_shared_dc_from_ac'] +
                # PV→AC for load
                self.vars['pv_load'][t] +
                # DC→AC discharging (including reg-up)
                p['eta_shared_ac_from_dc'] * (self.vars['d_load'][t] + self.vars['d_wh'][t] + 
                                             self.vars['d_freq'][t]) <=
                p['P_inv_shared'] * dt,
                name=f"hybrid_shared_pcs_limit_{t}"
            )
            
            # ========== (23a) Reserve Stacking Cap ==========
            self.model.addConstr(
                gp.quicksum(self.vars['r_k'][k, t] for k in p['K']) <= p['P_inv_shared'],
                name=f"hybrid_reserve_stack_limit_{t}"
            )
            
            # ========== (24) Optional PV DC/DC Port Rating ==========
            # Only apply if the DC/DC rating is specified
            if 'P_dc_pv' in p:
                self.model.addConstr(
                    self.vars['c_pv'][t] + 
                    self.vars['pv_load'][t] / p['eta_shared_ac_from_dc'] <=
                    p['P_dc_pv'] * dt,
                    name=f"hybrid_pv_dcdc_limit_{t}"
                )
            
            # ========== (25) POI Import/Export Limits ==========
            if 'P_grid_import_max' in p:
                # (25a) Grid import limit
                self.model.addConstr(
                    self.vars['g'][t] + 
                    (self.vars['c_ret'][t] + self.vars['c_wh'][t] + self.vars['c_freq'][t]) / 
                    p['eta_shared_dc_from_ac'] <=
                    p['P_grid_import_max'] * dt,
                    name=f"hybrid_poi_import_{t}"
                )
            
            if 'P_grid_export_max' in p:
                # (25b) Grid export limit
                self.model.addConstr(
                    p['eta_shared_ac_from_dc'] * (self.vars['d_wh'][t] + self.vars['d_freq'][t]) <=
                    p['P_grid_export_max'] * dt,
                    name=f"hybrid_poi_export_{t}"
                )
    
    def solve(self, time_limit: Optional[float] = None, 
              mip_gap: Optional[float] = 0.001) -> OptimizationResults:
        """
        Solve the MILP problem and return results.
        
        Args:
            time_limit: Maximum solve time in seconds
            mip_gap: MIP optimality gap tolerance
            
        Returns:
            OptimizationResults object with solution
        """
        # Set solver parameters
        if time_limit:
            self.model.setParam('TimeLimit', time_limit)
        if mip_gap:
            self.model.setParam('MIPGap', mip_gap)
        
        # Solve
        self.model.optimize()
        
        if self.model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            raise RuntimeError(f"Optimization failed with status {self.model.Status}")
        
        # Extract results
        return self._extract_results()
    
    def _extract_results(self) -> OptimizationResults:
        """Extract optimization results into structured format"""
        T = self.T
        p = self.params
        
        # Extract variable values
        def get_var_values(var_dict, indices=None):
            if indices is None:
                return np.array([var_dict[t].X for t in range(T)])
            else:
                return np.array([var_dict[idx, t].X for t in range(T)])
        
        # Energy flows
        c_ret = get_var_values(self.vars['c_ret'])
        c_wh = get_var_values(self.vars['c_wh'])
        c_pv = get_var_values(self.vars['c_pv'])
        c_freq = get_var_values(self.vars['c_freq'])
        d_load = get_var_values(self.vars['d_load'])
        d_wh = get_var_values(self.vars['d_wh'])
        d_freq = get_var_values(self.vars['d_freq'])
        
        # Tagged flows
        sources = ['ret', 'wh', 'pv']
        d_load_by_source = {s: get_var_values(self.vars['d_load_s'], s) for s in sources}
        d_wh_by_source = {s: get_var_values(self.vars['d_wh_s'], s) for s in sources}
        
        # State variables
        soc = get_var_values(self.vars['soc'])
        soc_by_source = {s: get_var_values(self.vars['soc_s'], s) for s in sources}
        
        # PV flows
        pv_load = get_var_values(self.vars['pv_load'])
        pv_curt = get_var_values(self.vars['pv_curt'])
        g = get_var_values(self.vars['g'])
        
        # Market participation
        reserve_capacity = {k: get_var_values(self.vars['r_k'], k) for k in p['K']}
        activation_up = {k: get_var_values(self.vars['d_k'], k) for k in p['K']}
        activation_down = {k: get_var_values(self.vars['c_k'], k) for k in p['K']}
        Q_cap = self.vars['Q_cap'].X
        
        # Binaries
        action_binaries = {
            'y_ch_ret': get_var_values(self.vars['y_ch_ret']),
            'y_ch_wh': get_var_values(self.vars['y_ch_wh']),
            'y_ch_pv': get_var_values(self.vars['y_ch_pv']),
            'y_dis_load': get_var_values(self.vars['y_dis_load']),
            'y_dis_wh': get_var_values(self.vars['y_dis_wh']),
            'y_idle': get_var_values(self.vars['y_idle']),
        }
        
        # Calculate financial metrics
        energy_cost = np.sum(
            (p['L_t'] - d_load - pv_load) * p['P_ret_t'] +
            c_ret * p['P_ret_t'] +
            c_wh * p['P_wh_t'] -
            d_wh * p['P_wh_t']
        )
        
        if p['degradation_kind'] == 'throughput':
            degradation_cost = np.sum([self.vars['dc'][t].X for t in range(T)])
        else:
            degradation_cost = 0
        
        ancillary_revenue = sum(
            np.sum(p['pi_k_t'][k] * reserve_capacity[k]) for k in p['K']
        )
        
        capacity_revenue = Q_cap * np.sum(p['pi_cap_t'][p['Omega_cap']])
        
        total_revenue = -energy_cost - degradation_cost + ancillary_revenue + capacity_revenue
        
        return OptimizationResults(
            objective_value=self.model.ObjVal,
            solve_time=self.model.Runtime,
            status=self.model.Status,
            charge_retail=c_ret,
            charge_wholesale=c_wh,
            charge_pv=c_pv,
            charge_freq=c_freq,
            discharge_load=d_load,
            discharge_wholesale=d_wh,
            discharge_freq=d_freq,
            discharge_load_by_source=d_load_by_source,
            discharge_wh_by_source=d_wh_by_source,
            soc_total=soc,
            soc_by_source=soc_by_source,
            pv_to_load=pv_load,
            pv_curtailed=pv_curt,
            grid_to_load=g,
            reserve_capacity=reserve_capacity,
            activation_up=activation_up,
            activation_down=activation_down,
            capacity_commitment=Q_cap,
            action_binaries=action_binaries,
            energy_cost=energy_cost,
            degradation_cost=degradation_cost,
            ancillary_revenue=ancillary_revenue,
            capacity_revenue=capacity_revenue,
            total_revenue=total_revenue
        )


def optimize_bess(data_bundle, **solver_kwargs) -> OptimizationResults:
    """
    High-level interface to optimize BESS operations.
    
    Args:
        data_bundle: DataBundle object with all input data
        **solver_kwargs: Additional solver parameters
        
    Returns:
        OptimizationResults with complete solution
    """
    from ..data_model.milp_interface import MILPDataInterface
    
    # Convert data bundle to MILP parameters
    interface = MILPDataInterface(data_bundle)
    params = interface.to_milp_params()
    
    # Create and solve MILP
    dispatcher = MILPDispatcher(params)
    results = dispatcher.solve(**solver_kwargs)
    
    # Log summary
    logger.info(f"Optimization complete: {results.status}")
    logger.info(f"Total revenue: €{results.total_revenue:,.2f}")
    logger.info(f"Solve time: {results.solve_time:.2f}s")
    
    return results
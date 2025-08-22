"""
BESS Optimization Framework
Battery Energy Storage System optimization with MILP.
"""

__version__ = "0.1.0"

# Main API exports
from .optimize import BESSOptimizer, quick_optimize
from .schema import DataBundle, TimeGrid, Architecture, AncillaryService
from .schema import Ratings, Efficiencies, Bounds, Degradation, Tariffs, Exogenous, Capacity
from .io import DataLoader, DataWriter, generate_template

# Convenience imports
from .optimization.milp_interface import MILPDataInterface
from .utils.validators import validate_data_bundle, generate_validation_report

__all__ = [
    "BESSOptimizer",
    "quick_optimize", 
    "DataBundle",
    "TimeGrid",
    "Architecture",
    "AncillaryService",
    "Ratings",
    "Efficiencies", 
    "Bounds",
    "Degradation",
    "Tariffs",
    "Exogenous",
    "Capacity",
    "DataLoader",
    "DataWriter",
    "generate_template",
    "MILPDataInterface",
    "validate_data_bundle",
    "generate_validation_report",
]

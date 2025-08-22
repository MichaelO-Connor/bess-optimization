

# BESS Optimization Framework

A comprehensive Mixed Integer Linear Programming (MILP) framework for optimizing Battery Energy Storage System (BESS) dispatch strategies across multiple revenue streams.

## Features

- **Multi-Revenue Optimization**: Energy arbitrage, ancillary services, capacity markets, and behind-the-meter applications
- **Flexible Architecture Support**: AC-coupled, DC-coupled, and hybrid configurations
- **Real Market Data**: Integration with ENTSO-E for European electricity markets
- **CSV Data Input**: Load your own time series data from CSV files
- **Advanced Modeling**: Battery degradation, reserve deliverability, and tagged energy accounting
- **Comprehensive Validation**: Built-in data validation and consistency checks
- **Command-Line Interface**: Easy-to-use CLI for common operations
- **Professional Package**: Proper Python packaging with testing and documentation

## Quick Start

### Installation

1. **Install Gurobi** (required for optimization)
   - Get a license from [Gurobi](https://www.gurobi.com/downloads/)
   - Academic licenses are free
   - Set up your license file

2. **Install the Package**
```bash
git clone https://github.com/yourusername/bess-optimization.git
cd bess-optimization
pip install -e .
```

3. **Set Environment Variables**
```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
export ENTSOE_API_KEY=your-key-here  # Optional, for market data
```

### Basic Usage with CSV Data

```python
from bess import BESSOptimizer

# Generate template files
from bess.io import generate_template
generate_template('my_config.yaml', architecture='ac_coupled')

# Load configuration and optimize
optimizer = BESSOptimizer.from_config('my_config.yaml')
results = optimizer.optimize()

print(f"Total revenue: €{results.total_revenue:.2f}")
```

### CSV File Format

Your CSV file should have the following columns:

| Column | Description | Units | Required |
|--------|-------------|-------|----------|
| timestamp | Datetime index | ISO format | Yes |
| site_load_kw | Site electricity demand | kW | Yes |
| retail_price | Import electricity price | €/kWh | Yes |
| wholesale_price | Export/spot price | €/kWh | Yes |
| pv_kw | Solar generation | kW | No |
| fcr_price | FCR reserve price | €/MW/h | No |
| afrr_up_price | aFRR up price | €/MW/h | No |
| capacity_price | Capacity market price | €/kW | No |

Example CSV:
```csv
timestamp,site_load_kw,retail_price,wholesale_price,pv_kw
2024-01-01T00:00:00,100,0.15,0.10,0
2024-01-01T01:00:00,95,0.14,0.09,0
2024-01-01T02:00:00,90,0.13,0.08,0
...
```

## Command Line Interface

### Generate Templates

```bash
# Generate both config and CSV templates
python -m bess.cli generate-template --type both -o my_project/

# Generate only configuration template
python -m bess.cli generate-template --type config -o templates/
```

### Run Optimization

```bash
# Optimize from configuration file
python -m bess.cli optimize config.yaml --csv data.csv -o results.csv

# With time limit and verbose output
python -m bess.cli optimize config.yaml --time-limit 300 --verbose
```

### Validate Data

```bash
# Validate CSV data
python -m bess.cli validate data.csv --report validation.txt

# Validate configuration
python -m bess.cli validate config.yaml --strict
```

## Architecture Types

### AC-Coupled Systems
```yaml
architecture:
  kind: 'ac_coupled'
  forbid_retail_to_load: true

ratings:
  p_inv_pv_kw: 3000      # PV inverter rating
  p_inv_bess_kw: 2500    # Battery inverter rating

efficiencies:
  eta_pv_ac_from_dc: 0.98
  eta_bess_dc_from_ac: 0.97
  eta_bess_ac_from_dc: 0.97
```

### DC-Coupled Systems
```yaml
architecture:
  kind: 'dc_coupled'

ratings:
  p_inv_shared_kw: 3000  # Shared inverter rating

efficiencies:
  eta_shared_dc_from_ac: 0.97
  eta_shared_ac_from_dc: 0.97
```

### Hybrid Systems
```yaml
architecture:
  kind: 'hybrid'

ratings:
  p_inv_shared_kw: 3000  # Shared inverter rating
  p_dc_pv_kw: 3500       # Optional PV DC/DC converter

efficiencies:
  eta_shared_dc_from_ac: 0.97
  eta_shared_ac_from_dc: 0.97
```

## Ancillary Services Configuration

```yaml
architecture:
  ancillary_services:
    - name: 'fcr'
      direction: 'both'            # up, down, or both
      sustain_duration_hours: 0.25 # 15 minutes for FCR
      settlement_method: 'availability_only'
      
    - name: 'afrr_up'
      direction: 'up'
      sustain_duration_hours: 1.0
      activation_fraction_up: 0.05  # 5% expected activation
      settlement_method: 'wholesale_settled'
```

## Real Market Data with ENTSO-E

### Quick Example
```python
from bess.data_sources.entsoe_to_bess import quick_entsoe_optimization

# Requires ENTSOE_API_KEY environment variable
results = quick_entsoe_optimization(
    country='germany',
    days=1,
    capacity_kwh=1000,
    power_kw=250
)
```

### Detailed Example
```python
from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer
from datetime import datetime
import numpy as np

# Create site-specific load profile (REQUIRED - not from ENTSO-E!)
from bess.data_sources.load_profiles import create_commercial_load_profile

start = datetime(2024, 6, 1)
end = datetime(2024, 6, 2)

site_load = create_commercial_load_profile(
    start=start,
    end=end,
    base_load_kw=100,
    peak_load_kw=500
)

optimizer = ENTSOEBESSOptimizer()

results = optimizer.optimize_with_entsoe(
    country='germany',
    start=start,
    end=end,
    bess_config={
        'capacity_kwh': 1000,
        'power_kw': 250,
        'efficiency': 0.90
    },
    load_profile=site_load,  # YOUR site load - required!
    retail_tariff=0.25       # Your retail tariff
)
```

### Supported Countries
- Germany (`'germany'`): FCR, aFRR
- Netherlands (`'netherlands'`): FCR, aFRR  
- France (`'france'`): FCR, aFRR, mFRR
- Belgium (`'belgium'`): FCR, aFRR
- Great Britain (`'gb'`): FFR, Dynamic Containment

## Advanced Configuration

### Multiple Revenue Streams
```python
from bess import BESSOptimizer
from bess.schema import AncillaryService

# Define services
services = [
    AncillaryService(
        name='fcr',
        direction='both',
        sustain_duration_hours=0.25,
        settlement_method='availability_only'
    ),
    AncillaryService(
        name='afrr_up',
        direction='up', 
        sustain_duration_hours=1.0,
        activation_fraction_up=0.05,
        settlement_method='wholesale_settled'
    )
]

# Create configuration with services
# ... (see examples/ for complete code)
```

### Custom Load Profiles
```python
from bess.data_sources.load_profiles import (
    create_commercial_load_profile,
    create_industrial_load_profile,
    create_residential_load_profile
)

# Commercial building
load = create_commercial_load_profile(
    start=start,
    end=end,
    base_load_kw=50,
    peak_load_kw=200,
    business_hours=(7, 19)
)

# Industrial facility
load = create_industrial_load_profile(
    start=start,
    end=end,
    base_load_kw=500,
    production_load_kw=2000,
    shifts=[(6, 14), (14, 22)]
)
```

## Data Validation

```python
from bess.utils.validators import validate_data_bundle, generate_validation_report

# Validate configuration
is_valid, errors, warnings = validate_data_bundle(bundle)

if errors:
    for error in errors:
        print(f"Error: {error}")

# Generate detailed report
report = generate_validation_report(bundle, 'validation_report.txt')
```

## Results Analysis

```python
# Get key performance indicators
kpis = optimizer.get_kpis()

print(f"Total revenue: €{kpis['total_revenue']:.2f}")
print(f"Daily cycles: {kpis['avg_daily_cycles']:.2f}")
print(f"Utilization: {kpis['utilization']:.1%}")

# Access detailed results
results.charge_retail      # Retail charging (kWh)
results.discharge_wholesale # Wholesale export (kWh)
results.soc_total          # State of charge (kWh)
results.reserve_capacity   # Ancillary reserves by service (kW)

# Save results
optimizer.save_results('results.csv', format='csv')
```

## Project Structure

```
bess-optimization/
├── bess/                   # Main package
│   ├── __init__.py         # Package exports
│   ├── schema.py           # Data models (Pydantic)
│   ├── optimize.py         # High-level optimizer
│   ├── io.py              # Data I/O utilities
│   ├── cli.py             # Command-line interface
│   │
│   ├── optimization/       # MILP solver core
│   │   ├── milp_interface.py    # DataBundle → MILP
│   │   └── milp_dispatcher.py   # Gurobi MILP solver
│   │
│   ├── data_sources/       # Data input modules
│   │   ├── csv_loader.py        # CSV data loading
│   │   ├── entsoe_integration.py # ENTSO-E API client
│   │   ├── entsoe_to_bess.py    # ENTSO-E → BESS integration
│   │   └── load_profiles.py     # Synthetic load generators
│   │
│   └── utils/              # Utilities
│       └── validators.py        # Data validation
│
├── examples/               # Usage examples
│   ├── basic_optimization.py    # Simple arbitrage
│   ├── csv_example.py           # CSV workflows
│   └── entsoe_example.py        # Real market data
│
├── config/                 # Configuration templates
│   ├── base_config.yaml         # Full configuration example
│   └── examples/
│       └── ac_coupled.yaml      # AC-coupled example
│
├── data/
│   └── templates/              # CSV templates
│       └── timeseries_template.csv
│
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Examples

### Basic Energy Arbitrage
```python
# See examples/basic_optimization.py
python examples/basic_optimization.py
```

### CSV Data Workflow
```python
# See examples/csv_example.py
python examples/csv_example.py
```

### Real Market Data
```python
# See examples/entsoe_example.py  
# Requires ENTSO-E API key
python examples/entsoe_example.py
```

## Requirements

**Core Dependencies:**
- Python 3.8+
- Gurobi 10.0+ (with valid license)
- NumPy, Pandas, Pydantic, PyYAML

**Optional:**
- `entsoe-py` (for ENTSO-E market data)
- `matplotlib` (for plotting)

Install optional dependencies:
```bash
pip install -e ".[entsoe,viz]"    # ENTSO-E + visualization
pip install -e ".[dev]"           # Development tools
```

## Getting Help

### Documentation
- **Quick Start**: `docs/quickstart.md`
- **ENTSO-E Guide**: `docs/entsoe_guide.md`
- **API Reference**: Generated from docstrings

### Examples
- `examples/basic_optimization.py` - Start here
- `examples/csv_example.py` - Working with CSV data
- `examples/entsoe_example.py` - Real market data

### Common Issues

1. **"Gurobi not found"**
   - Install Gurobi and set `GRB_LICENSE_FILE`
   - Verify: `python -c "import gurobipy"`

2. **"No module named bess"**
   - Run `pip install -e .` from the repository root

3. **"Optimization infeasible"**
   - Check battery capacity vs. load requirements
   - Review SoC bounds and ancillary service requirements
   - Run validation: `python -m bess.cli validate config.yaml`

4. **"ENTSO-E API errors"**
   - Verify API key: `echo $ENTSOE_API_KEY`
   - Check date range (some data has 5-day lag)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest`
5. Submit a pull request

---

**Note**: This framework is designed for research and commercial applications. The optimization models are based on established mathematical formulations for battery dispatch optimization with multiple revenue streams.

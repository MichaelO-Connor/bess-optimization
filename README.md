# BESS Optimization Framework

A comprehensive Mixed Integer Linear Programming (MILP) framework for optimizing Battery Energy Storage System (BESS) dispatch strategies across multiple revenue streams.

## Features

- **Multi-Revenue Optimization**: Energy arbitrage, ancillary services, capacity markets, and behind-the-meter applications
- **Flexible Architecture Support**: AC-coupled, DC-coupled, and hybrid configurations
- **Real Market Data**: Integration with ENTSO-E for European electricity markets
- **CSV Data Input**: Load your own time series data from CSV files
- **Advanced Modeling**: Battery degradation, reserve deliverability, and tagged energy accounting
- **Comprehensive Validation**: Built-in data validation and consistency checks

## Quick Start

### Installation

1. **Install Gurobi** (required for optimization)
   - Get a license from [Gurobi](https://www.gurobi.com/downloads/)
   - Academic licenses are free
   - Set up your license file

2. **Clone and Install Package**
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
from bess.data_sources.csv_loader import CSVDataLoader

# Load your time series data
data = CSVDataLoader.load_from_file('data/timeseries.csv')

# Create optimizer
optimizer = BESSOptimizer.from_csv(
    data,
    battery_kwh=1000,  # 1 MWh battery
    battery_kw=250      # 250 kW power
)

# Run optimization
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
| capacity_price | Capacity market price | €/kW | No |

Example CSV:
```csv
timestamp,site_load_kw,retail_price,wholesale_price,pv_kw
2024-01-01T00:00:00,100,0.15,0.10,0
2024-01-01T01:00:00,95,0.14,0.09,0
2024-01-01T02:00:00,90,0.13,0.08,0
...
```

## Advanced Configuration

### Using YAML Configuration

```yaml
# config.yaml
timegrid:
  start: '2024-01-01T00:00:00'
  end: '2024-01-08T00:00:00'
  dt_minutes: 60
  tz: 'UTC'

architecture:
  kind: 'ac_coupled'
  ancillary_services:
    - name: 'fcr'
      direction: 'both'
      sustain_duration_hours: 0.25

ratings:
  p_charge_max_kw: 2500
  p_discharge_max_kw: 2500
  p_inv_bess_kw: 2500

capacity:
  capacity_nominal_kwh: 10000

data_source:
  type: 'csv'
  csv:
    filepath: 'data/timeseries.csv'
```

Load and optimize:
```python
from bess import BESSOptimizer

optimizer = BESSOptimizer.from_config('config.yaml')
results = optimizer.optimize()
```

### Multiple CSV Files

```python
from bess.data_sources.csv_loader import CSVDataLoader

# Load from separate files
data = CSVDataLoader.load_from_multiple_files({
    'load': 'data/site_load.csv',
    'prices': 'data/prices.csv',
    'pv': 'data/pv_generation.csv',
    'ancillary': 'data/ancillary_prices.csv'
})

# Continue with optimization...
```

## Data Validation

### Validate Before Optimization

```python
from bess.utils.validators import validate_data_bundle, generate_validation_report

# Create your data bundle
bundle = create_bundle_from_csv(data)

# Validate
is_valid, errors, warnings = validate_data_bundle(bundle)

if is_valid:
    print("✓ Data validation passed")
else:
    print(f"✗ {len(errors)} errors found")
    for error in errors:
        print(f"  - {error}")

# Generate detailed report
report = generate_validation_report(bundle, 'validation_report.txt')
```

## Examples

### Example 1: Energy Arbitrage Only

```python
from bess import quick_optimize

# Simple arbitrage optimization
results = quick_optimize(
    csv_file='data/prices.csv',
    battery_kwh=1000,
    battery_kw=250,
    output_file='results.csv'
)
```

### Example 2: With Ancillary Services

```python
from bess import BESSOptimizer, AncillaryService

# Configure services
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

# Optimize with services
optimizer = BESSOptimizer.from_csv(
    data,
    battery_kwh=1000,
    battery_kw=250,
    ancillary_services=services
)

results = optimizer.optimize()
```

### Example 3: ENTSO-E Market Data (Optional)

```python
from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer
from datetime import datetime

# Requires ENTSO-E API key
optimizer = ENTSOEBESSOptimizer()

# Use real market data
results = optimizer.optimize_with_entsoe(
    country='germany',
    start=datetime(2024, 6, 1),
    end=datetime(2024, 6, 2),
    bess_config={
        'capacity_kwh': 1000,
        'power_kw': 250,
        'efficiency': 0.90
    },
    load_profile=your_site_load  # Must provide site load!
)
```

## Output and Results

### Access Optimization Results

```python
# Energy flows (kWh per timestep)
results.charge_retail      # Retail charging
results.charge_wholesale   # Wholesale charging
results.discharge_load     # Serving site load
results.discharge_wholesale # Export to grid

# Ancillary services (kW)
results.reserve_capacity['fcr']    # FCR commitment
results.reserve_capacity['afrr_up'] # aFRR up commitment

# State of charge (kWh)
results.soc_total

# Financial metrics
results.total_revenue      # Net revenue (€)
results.energy_cost        # Energy costs (€)
results.ancillary_revenue  # AS revenue (€)
```

### Save Results

```python
# Save to CSV
optimizer.save_results('results.csv')

# Get KPIs
kpis = optimizer.get_kpis()
print(f"Daily cycles: {kpis['avg_daily_cycles']:.2f}")
print(f"Utilization: {kpis['utilization']:.1%}")

# Generate plots
optimizer.plot_dispatch('dispatch.png')
```

## Project Structure

```
bess-optimization/
├── bess/               # Main package
│   ├── schema.py       # Data models
│   ├── optimize.py     # High-level optimizer
│   ├── io.py          # Data I/O
│   ├── optimization/   # MILP implementation
│   └── data_sources/   # Data loaders
├── examples/           # Example scripts
├── data/              # Data directory
│   └── templates/     # CSV templates
├── config/            # Configuration files
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Troubleshooting

### Common Issues

1. **"Gurobi not found"**
   - Install Gurobi and set `GRB_LICENSE_FILE` environment variable
   - Verify installation: `python -c "import gurobipy"`

2. **"Invalid CSV format"**
   - Check column names match expected format
   - Ensure timestamp column is properly formatted
   - Verify no missing required columns

3. **"Optimization infeasible"**
   - Check battery capacity vs load requirements
   - Verify SoC bounds are reasonable
   - Review ancillary service requirements

4. **"Validation errors"**
   - Run validation report for detailed diagnostics
   - Check time series lengths are consistent
   - Verify all prices are non-negative

## Requirements

- Python 3.8+
- Gurobi 10.0+ (with valid license)
- NumPy, Pandas, PyYAML
- Optional: matplotlib (for plots), entsoe-py (for market data)

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [Read the Docs](https://bess-optimization.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/yourusername/bess-optimization/issues)
- Examples: See `examples/` directory

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bess_optimization,
  title = {BESS Optimization Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bess-optimization}
}
```
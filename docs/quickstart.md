# BESS Optimization Quick Start

## Installation

1. **Install Gurobi** (required)
   - Get license from [Gurobi Downloads](https://www.gurobi.com/downloads/)
   - Academic licenses are free
   - Set `GRB_LICENSE_FILE` environment variable

2. **Install Package**
   ```bash
   pip install -e .
   ```

## Basic Usage

### 1. CSV Data Optimization

```python
from bess import BESSOptimizer

# Load from configuration
optimizer = BESSOptimizer.from_config('config/examples/ac_coupled.yaml')

# Run optimization  
results = optimizer.optimize()

print(f"Total revenue: €{results.total_revenue:.2f}")
```

### 2. CSV Data Format

Your CSV should have columns:
- `timestamp`: ISO datetime
- `site_load_kw`: Site electricity demand (kW) 
- `retail_price`: Import price (€/kWh)
- `wholesale_price`: Export price (€/kWh)
- `pv_kw`: Solar generation (kW, optional)

See `data/templates/timeseries_template.csv` for example format.

### 3. Command Line

```bash
# Generate templates
bess-generate-template --type both -o my_project/

# Run optimization
bess-optimize config.yaml --csv data.csv -o results.csv

# Validate data
bess-validate data.csv --report validation.txt
```

## Examples

Check the `examples/` directory for:
- `basic_optimization.py` - Simple energy arbitrage
- `csv_example.py` - CSV data workflows
- `entsoe_example.py` - Real European market data (requires API key)

## Next Steps

1. Create your own CSV data file
2. Copy and modify `config/examples/ac_coupled.yaml`  
3. Run optimization with your data
4. Analyze results and iterate

See the main README.md for comprehensive documentation.

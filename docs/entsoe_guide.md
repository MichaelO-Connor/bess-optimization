# ENTSO-E Integration Quick Start Guide

## Installation

Add to your `requirements.txt`:
```
entsoe-py>=0.5.10
pandas>=1.3.0
numpy>=1.21.0
```

## Get Your API Key (Free)

1. Register at https://transparency.entsoe.eu/
2. Email `transparency@entsoe.eu` with subject "Restful API access"  
3. Once approved, go to Account Settings → Generate Security Token
4. Set environment variable:
```bash
export ENTSOE_API_KEY='your-token-here'
```

## Quick Start Examples

### 1. One-Line Optimization

```python
from bess.data_sources.entsoe_to_bess import quick_entsoe_optimization

# Optimize 1 MWh battery for tomorrow's German market
results = quick_entsoe_optimization(
    country='germany',
    days=1,
    capacity_kwh=1000,
    power_kw=250
)
```

### 2. Custom Country & Services

```python
from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer
from datetime import datetime

optimizer = ENTSOEBESSOptimizer()

# Define BESS
bess_config = {
    'capacity_kwh': 2000,
    'power_kw': 500,
    'efficiency': 0.90,
    'soc_min': 0.1,
    'soc_max': 0.9
}

# Run for Netherlands with default services
results = optimizer.optimize_with_entsoe(
    country='netherlands',
    start=datetime(2024, 6, 1),
    end=datetime(2024, 6, 2),
    bess_config=bess_config
)
```

### 3. Direct ENTSO-E Data Access

```python
from bess.data_sources.entsoe_integration import ENTSOEDataFetcher
import pandas as pd

# Create fetcher for Germany
fetcher = ENTSOEDataFetcher(
    api_key='your-key',
    bidding_zone='DE_LU',
    timezone='Europe/Berlin'
)

# Fetch MILP-ready data
start = pd.Timestamp('2024-06-01', tz='Europe/Berlin')
end = pd.Timestamp('2024-06-02', tz='Europe/Berlin')

data = fetcher.fetch_milp_data(
    start=start,
    end=end,
    dt_minutes=15,
    ancillary_services=[
        {'name': 'fcr', 'direction': 'both', 'sustain_duration_hours': 0.25},
        {'name': 'afrr_up', 'direction': 'up', 'sustain_duration_hours': 1.0}
    ]
)

# Access formatted data
wholesale_prices = data['P_wh_t']  # EUR/kWh
fcr_prices = data['pi_k_t']['fcr']  # EUR/kW/period
```

## Supported Countries

| Country | Code | Bidding Zone | Services |
|---------|------|--------------|----------|
| Germany | `'germany'` | `'DE_LU'` | FCR, aFRR, mFRR |
| Netherlands | `'netherlands'` | `'NL'` | FCR, aFRR |
| France | `'france'` | `'FR'` | FCR, aFRR, mFRR |
| Belgium | `'belgium'` | `'BE'` | FCR, aFRR |
| Great Britain | `'gb'` | `'GB'` | FFR, DC |
| Austria | `'AT'` | `'AT'` | FCR, aFRR |
| Switzerland | `'CH'` | `'CH'` | FCR, aFRR |

## Data Retrieved from ENTSO-E

### ✅ Automatically Fetched from ENTSO-E
- **Day-ahead wholesale prices** (`P_wh_t`) - EUR/kWh
- **Ancillary service availability prices** (`pi_k_t`) - EUR/kW/period
- **Activation settlement prices** (`u_k_t`) - EUR/kWh (or derived from wholesale)
- **Historical activation rates** (`alpha_k_t`) - fraction [0,1] estimated from history

### ❌ NOT Available from ENTSO-E (Must Provide)
- **Site-specific load** (`L_t`) - Your behind-the-meter load profile (kWh)
  - This is YOUR building/facility load, not grid-wide load!
  - Must come from meter data, forecasts, or synthetic profiles
- **Retail tariff** (`P_ret_t`) - Your specific electricity supply tariff
- **PV generation** (`a_t`) - Your rooftop/on-site solar forecast
- **Service sustain durations** (`H_k`) - From market rules (e.g., FCR = 15 min)

### ⚠️ Common Mistake
ENTSO-E provides **system-wide grid load** (total country/zone demand in MW), which is NOT the same as your **site-specific load** (your building's consumption in kW). The MILP needs site-specific load!

## Complete Example: German Market with Site Load

```python
from bess.data_sources.entsoe_to_bess import ENTSOEBESSOptimizer
from datetime import datetime, timedelta
import numpy as np
import os

# Set API key
os.environ['ENTSOE_API_KEY'] = 'your-key-here'

# Initialize
optimizer = ENTSOEBESSOptimizer()

# Configure 10 MWh / 2.5 MW battery
bess_config = {
    'capacity_kwh': 10000,
    'power_kw': 2500,
    'efficiency': 0.90,
    'soc_min': 0.1,
    'soc_max': 0.9,
    'cycles_per_day': 2.0,
    'degradation_cost': 0.02  # 20 EUR/MWh
}

# IMPORTANT: Define YOUR site-specific load profile (kW)
# Example: Industrial facility with 2 MW base load
hours_per_day = 96  # 15-min intervals
days = 7
total_hours = hours_per_day * days

# Create realistic industrial load pattern
time_array = np.arange(total_hours)
hour_of_day = (time_array * 0.25) % 24  # Convert to hours

site_load_kw = np.zeros(total_hours)
for i in range(total_hours):
    h = hour_of_day[i]
    if 6 <= h < 22:  # Production hours
        site_load_kw[i] = 2000 + 500 * np.sin(np.pi * (h - 6) / 16)
    else:  # Night/weekend
        site_load_kw[i] = 800  # Base load only

# Define services (German FCR market)
services = [
    {
        'name': 'fcr',
        'direction': 'both',
        'sustain_duration_hours': 0.25,  # 15 min
        'settlement_method': 'availability_only',
        'entsoe_product_type': 'FCR'
    }
]

# Optimize for next week
start = datetime.now() + timedelta(days=1)
end = start + timedelta(days=7)

results = optimizer.optimize_with_entsoe(
    country='germany',
    start=start,
    end=end,
    bess_config=bess_config,
    dt_minutes=15,  # 15-min resolution for FCR
    custom_services=services,
    load_profile=site_load_kw,  # YOUR SITE LOAD!
    retail_tariff=0.25  # Your retail tariff EUR/kWh
)

print(f"Weekly revenue: €{results.total_revenue:.2f}")
```

### Alternative: Load from CSV

```python
import pandas as pd

# Load your meter data
meter_data = pd.read_csv('site_meter_data.csv', 
                         index_col='timestamp', 
                         parse_dates=True)

# Extract load profile for optimization period
site_load_kw = meter_data.loc[start:end, 'load_kw'].values

# Use in optimization
results = optimizer.optimize_with_entsoe(
    country='germany',
    start=start,
    end=end,
    bess_config=bess_config,
    load_profile=site_load_kw  # From your meter!
)
```

## Understanding the Results

The optimization returns energy flows at each timestep:

```python
# Energy flows (kWh per timestep)
results.charge_wholesale    # Wholesale charging
results.discharge_wholesale # Wholesale export
results.charge_pv          # PV charging
results.discharge_load     # Serving local load

# Ancillary services (kW)
results.reserve_capacity['fcr']     # FCR capacity commitment
results.reserve_capacity['afrr_up'] # aFRR up commitment

# State of charge (kWh)
results.soc_total

# Financial metrics
results.energy_cost         # Cost of energy purchases
results.ancillary_revenue   # Revenue from AS participation
results.total_revenue       # Net revenue
```

## Troubleshooting

### "No data found"
- Check your bidding zone code is correct
- Verify date range (ENTSO-E has 5-day lag for some data)
- Some markets don't publish all data types

### "Invalid security token"
- Verify API key is correct: `echo $ENTSOE_API_KEY`
- Check key is approved (can take 1-2 days after email)

### Missing ancillary prices
- Not all markets publish AS prices via ENTSO-E
- Try different service names or product types
- Check ENTSO-E transparency platform manually

### Rate limiting
- ENTSO-E allows 400 requests/minute
- Built-in rate limiting at 0.15s between requests
- Use caching for development

## API Data Mapping

| MILP Parameter | ENTSO-E API Call | Document Type | Notes |
|----------------|------------------|---------------|-------|
| `P_wh_t` | `query_day_ahead_prices()` | A44 | Wholesale energy prices |
| `pi_fcr_t` | `query_contracted_reserve_prices()` | A81 | FCR availability prices |
| `pi_afrr_t` | `query_procured_balancing_capacity()` | A15 | aFRR capacity prices |
| `u_k_t` | `query_imbalance_prices()` | A85 | Activation/imbalance prices |
| `alpha_k_t` | Calculated from historical activation/capacity | A83/A81 | Estimated from past data |
| ~~`L_t`~~ | ❌ NOT AVAILABLE | - | **Site load must be user-provided!** |
| System Load | `query_load_forecast()` | A65 | Grid-wide MW (not for MILP) |

## Next Steps

1. **Add your own data**: Combine ENTSO-E with your meter/forecast data
2. **Extend to other markets**: Add custom API clients for your TSO
3. **Optimize dispatch**: Use MPC with rolling horizon
4. **Backtest strategies**: Download historical data for validation
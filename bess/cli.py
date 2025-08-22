#!/usr/bin/env python3
"""
Command-line interface for BESS optimization.
Provides convenient commands for common operations.
"""

import click
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

from bess import BESSOptimizer, __version__
from bess.data_sources.csv_loader import CSVDataLoader, create_sample_csv
from bess.utils.validators import validate_data_bundle, generate_validation_report
from bess.io import generate_template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="bess-optimization")
def cli():
    """BESS Optimization Framework - Battery dispatch optimization with MILP."""
    pass


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--csv', 'csv_file', type=click.Path(exists=True),
              help='CSV file with time series data')
@click.option('--output', '-o', type=click.Path(),
              default='results.csv', help='Output file for results')
@click.option('--time-limit', type=int, default=300,
              help='Maximum solve time in seconds')
@click.option('--mip-gap', type=float, default=0.001,
              help='MIP optimality gap')
@click.option('--verbose', '-v', is_flag=True,
              help='Show detailed output')
def optimize(config_file, csv_file, output, time_limit, mip_gap, verbose):
    """
    Run BESS optimization from configuration file.
    
    Example:
        bess-optimize config.yaml --csv data.csv -o results.csv
    """
    try:
        # Set logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        click.echo(f"Loading configuration from {config_file}...")
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override CSV file if provided
        if csv_file:
            if 'data_source' not in config:
                config['data_source'] = {}
            config['data_source']['csv'] = {'filepath': csv_file}
        
        # Create optimizer
        if csv_file:
            click.echo(f"Loading time series from {csv_file}...")
            data = CSVDataLoader.load_from_file(csv_file)
            # TODO: Create bundle from config and CSV data
            click.echo("Creating optimizer...")
        else:
            click.echo("Creating optimizer from config...")
            optimizer = BESSOptimizer.from_config(config_file)
        
        # Run optimization
        click.echo(f"Running optimization (time limit: {time_limit}s)...")
        results = optimizer.optimize(
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose
        )
        
        # Save results
        click.echo(f"Saving results to {output}...")
        optimizer.save_results(output)
        
        # Display summary
        click.echo("\n" + "="*50)
        click.echo("OPTIMIZATION RESULTS")
        click.echo("="*50)
        click.echo(f"Status: {results.status}")
        click.echo(f"Total Revenue: €{results.total_revenue:,.2f}")
        click.echo(f"Energy Cost: €{results.energy_cost:,.2f}")
        click.echo(f"Ancillary Revenue: €{results.ancillary_revenue:,.2f}")
        click.echo(f"Degradation Cost: €{results.degradation_cost:,.2f}")
        click.echo(f"Solve Time: {results.solve_time:.2f}s")
        
        # Get KPIs
        kpis = optimizer.get_kpis()
        click.echo("\nKey Performance Indicators:")
        click.echo(f"  Daily Cycles: {kpis['avg_daily_cycles']:.2f}")
        click.echo(f"  Utilization: {kpis['utilization']:.1%}")
        click.echo(f"  PV Curtailed: {kpis['total_pv_curtailed']:.1f} kWh")
        
        click.echo(f"\n✓ Results saved to {output}")
        
    except Exception as e:
        click.echo(f"✗ Optimization failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--config', type=click.Path(exists=True),
              help='Configuration file for validation settings')
@click.option('--report', '-r', type=click.Path(),
              help='Save validation report to file')
@click.option('--strict/--no-strict', default=True,
              help='Fail on validation errors')
def validate(input_file, config, report, strict):
    """
    Validate input data for BESS optimization.
    
    Example:
        bess-validate data.csv --report validation.txt
    """
    try:
        click.echo(f"Loading data from {input_file}...")
        
        # Determine file type
        input_path = Path(input_file)
        
        if input_path.suffix == '.csv':
            # Load CSV data
            data = CSVDataLoader.load_from_file(input_path)
            click.echo(f"Loaded {data.n_timesteps} timesteps")
            
            # Create basic bundle for validation
            # TODO: Use config if provided
            from examples.csv_example import create_bundle_from_csv
            bundle = create_bundle_from_csv(data)
            
        elif input_path.suffix in ['.yaml', '.yml']:
            # Load from config
            from bess import DataLoader
            bundle = DataLoader.load_bundle(input_path)
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        
        # Run validation
        click.echo("Running validation...")
        is_valid, errors, warnings = validate_data_bundle(bundle, strict=False)
        
        # Display results
        if errors:
            click.echo(f"\n✗ Found {len(errors)} errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
        
        if warnings:
            click.echo(f"\n⚠ Found {len(warnings)} warnings:")
            for warning in warnings:
                click.echo(f"  - {warning}")
        
        if is_valid:
            click.echo("\n✓ Data validation passed")
        else:
            click.echo("\n✗ Data validation failed", err=True)
        
        # Generate report if requested
        if report:
            click.echo(f"\nGenerating validation report...")
            report_text = generate_validation_report(bundle, report)
            click.echo(f"Report saved to {report}")
        
        # Exit with error if strict and not valid
        if strict and not is_valid:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"✗ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command('generate-template')
@click.option('--type', 'template_type',
              type=click.Choice(['config', 'csv', 'both']),
              default='both', help='Type of template to generate')
@click.option('--output-dir', '-o', type=click.Path(),
              default='.', help='Output directory')
@click.option('--days', type=int, default=7,
              help='Number of days for CSV template')
@click.option('--timestep', type=int, default=60,
              help='Timestep in minutes for CSV template')
@click.option('--architecture', 
              type=click.Choice(['ac_coupled', 'dc_coupled', 'hybrid']),
              default='ac_coupled', help='System architecture')
def generate_template_cmd(template_type, output_dir, days, timestep, architecture):
    """
    Generate template configuration and data files.
    
    Example:
        bess-generate-template --type both -o templates/
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if template_type in ['config', 'both']:
            config_path = output_path / 'config_template.yaml'
            click.echo(f"Generating configuration template...")
            generate_template(config_path, architecture=architecture)
            click.echo(f"✓ Config template saved to {config_path}")
        
        if template_type in ['csv', 'both']:
            csv_path = output_path / 'data_template.csv'
            click.echo(f"Generating CSV template ({days} days, {timestep}min)...")
            create_sample_csv(
                csv_path,
                days=days,
                dt_minutes=timestep,
                include_pv=True,
                include_ancillary=True
            )
            click.echo(f"✓ CSV template saved to {csv_path}")
        
        click.echo("\nTemplates generated successfully!")
        click.echo("Edit the templates with your data and run:")
        click.echo(f"  bess-optimize {output_path}/config_template.yaml "
                  f"--csv {output_path}/data_template.csv")
        
    except Exception as e:
        click.echo(f"✗ Template generation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--format', 'output_format',
              type=click.Choice(['summary', 'detailed', 'json']),
              default='summary', help='Output format')
def analyze(results_file, output_format):
    """
    Analyze optimization results.
    
    Example:
        bess-analyze results.csv --format detailed
    """
    try:
        import pandas as pd
        import json
        
        click.echo(f"Loading results from {results_file}...")
        
        # Load results based on file type
        results_path = Path(results_file)
        
        if results_path.suffix == '.csv':
            df = pd.read_csv(results_path, index_col=0, parse_dates=True)
        elif results_path.suffix == '.parquet':
            df = pd.read_parquet(results_path)
        else:
            raise ValueError(f"Unsupported file type: {results_path.suffix}")
        
        # Calculate statistics
        stats = {
            'timesteps': len(df),
            'total_charge': df[[c for c in df.columns if 'charge' in c]].sum().sum(),
            'total_discharge': df[[c for c in df.columns if 'discharge' in c]].sum().sum(),
            'max_soc': df['soc_total'].max() if 'soc_total' in df else None,
            'min_soc': df['soc_total'].min() if 'soc_total' in df else None,
        }
        
        if output_format == 'json':
            click.echo(json.dumps(stats, indent=2, default=str))
        elif output_format == 'detailed':
            click.echo("\n" + "="*50)
            click.echo("DETAILED RESULTS ANALYSIS")
            click.echo("="*50)
            click.echo(df.describe())
            click.echo("\nColumn Summary:")
            for col in df.columns:
                click.echo(f"  {col}: mean={df[col].mean():.2f}, "
                          f"std={df[col].std():.2f}")
        else:  # summary
            click.echo("\n" + "="*50)
            click.echo("RESULTS SUMMARY")
            click.echo("="*50)
            for key, value in stats.items():
                if value is not None:
                    click.echo(f"{key}: {value:.2f}" if isinstance(value, float) 
                              else f"{key}: {value}")
        
    except Exception as e:
        click.echo(f"✗ Analysis failed: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
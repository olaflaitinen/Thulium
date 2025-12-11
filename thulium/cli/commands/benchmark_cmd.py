"""
Benchmark CLI command for Thulium.

Provides command-line interface for running benchmarks.
"""

import typer
from pathlib import Path
from typing import Optional, List
import logging

app = typer.Typer(help="Run HTR benchmarks")
logger = logging.getLogger(__name__)


@app.command()
def run(
    config: Path = typer.Argument(
        ...,
        help="Path to benchmark configuration YAML file"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output path for results (auto-detects format from extension)"
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Output format: markdown, csv, json"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output"
    ),
    include_samples: bool = typer.Option(
        False,
        "--include-samples",
        help="Include per-sample results in output"
    ),
):
    """
    Run a benchmark evaluation from a configuration file.
    
    Example:
        thulium benchmark run config/eval/iam_en.yaml
        thulium benchmark run config/eval/global_mixed.yaml -o results.md
    """
    from thulium.evaluation.benchmarking import run_benchmark
    from thulium.evaluation.reporting import (
        generate_markdown_report,
        generate_csv_report,
        generate_json_report,
        save_report
    )
    
    if not config.exists():
        typer.echo(f"Error: Config file not found: {config}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"Running benchmark: {config}")
    
    result = run_benchmark(str(config), verbose=verbose)
    
    if format == "markdown":
        report = generate_markdown_report(result, include_samples=include_samples)
    elif format == "csv":
        report = generate_csv_report(result, include_samples=include_samples)
    elif format == "json":
        report = generate_json_report(result, include_samples=include_samples)
    else:
        typer.echo(f"Unknown format: {format}", err=True)
        raise typer.Exit(1)
    
    if output:
        save_report(result, output, format=format, include_samples=include_samples)
        typer.echo(f"Results saved to: {output}")
    else:
        typer.echo(report)
    
    typer.echo("")
    typer.echo(f"Summary: CER={result.aggregate_cer*100:.2f}%, WER={result.aggregate_wer*100:.2f}%")


@app.command("list")
def list_configs():
    """List available benchmark configurations."""
    config_dir = Path("config/eval")
    if not config_dir.exists():
        typer.echo("No benchmark configs found in config/eval/")
        return
    
    typer.echo("Available benchmark configurations:")
    for config_file in sorted(config_dir.glob("*.yaml")):
        typer.echo(f"  - {config_file}")


if __name__ == "__main__":
    app()

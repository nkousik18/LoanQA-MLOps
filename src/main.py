#!/usr/bin/env python3
"""
Student Loan Intelligence System - Main Entry Point

This module provides the main CLI interface for the Student Loan Intelligence System,
supporting both API server and CLI operations.


"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from src.api.main import app
from src.core.config import settings
from src.core.logging import setup_logging
from src.extraction.cli import process_documents
from src.utils.version import get_version_info

# Console for rich output
console = Console()

# Main CLI application
cli_app = typer.Typer(
    name="sli",
    help="Student Loan Intelligence System - AI-powered loan processing platform",
    add_completion=False,
)

# Subcommands
api_app = typer.Typer(help="API server commands")
process_app = typer.Typer(help="Document processing commands")
cli_app.add_typer(api_app, name="api")
cli_app.add_typer(process_app, name="process")


@cli_app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    version: bool = typer.Option(
        False, "--version", help="Show version information"
    ),
) -> None:
    """
    Student Loan Intelligence System
    
    A comprehensive AI-powered platform for student loan document processing,
    comparison analysis, and educational guidance.
    """
    if version:
        version_info = get_version_info()
        console.print(Panel.fit(f"[bold blue]{version_info}[/bold blue]"))
        raise typer.Exit()
    
    # Setup logging based on verbosity
    log_level = "DEBUG" if verbose else settings.LOG_LEVEL
    setup_logging(log_level=log_level)
    
    # Welcome message
    if verbose:
        console.print(
            Panel(
                "[bold green]Student Loan Intelligence System[/bold green]\n"
                f"Environment: {settings.ENVIRONMENT}\n"
                f"Debug Mode: {settings.DEBUG}",
                title="System Status"
            )
        )


@api_app.command("serve")
def serve_api(
    host: str = typer.Option(
        settings.HOST, "--host", help="Host address to bind"
    ),
    port: int = typer.Option(
        settings.PORT, "--port", help="Port to bind"
    ),
    reload: bool = typer.Option(
        settings.DEBUG, "--reload", help="Enable auto-reload in development"
    ),
    workers: int = typer.Option(
        1, "--workers", help="Number of worker processes"
    ),
) -> None:
    """Start the FastAPI server."""
    
    console.print(
        Panel(
            f"[bold green]Starting API Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Workers: {workers}\n"
            f"Reload: {reload}",
            title="Server Configuration"
        )
    )
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,  # Single worker with reload
            log_level=settings.LOG_LEVEL.lower(),
            access_log=settings.DEBUG,
        )
    except Exception as e:
        console.print(f"[bold red]Failed to start server: {e}[/bold red]")
        raise typer.Exit(1)


@api_app.command("docs")
def serve_docs(
    port: int = typer.Option(
        8080, "--port", help="Documentation port"
    ),
) -> None:
    """Start documentation server."""
    
    console.print(
        Panel(
            f"[bold blue]Starting Documentation Server[/bold blue]\n"
            f"URL: http://localhost:{port}/docs",
            title="Documentation"
        )
    )
    
    # Run uvicorn with docs-only mode
    uvicorn.run(
        app,
        host="localhost",
        port=port,
        log_level="info",
        access_log=True,
    )


@process_app.command("extract")
def extract_documents(
    input_path: Path = typer.Argument(
        ..., help="Path to input file or directory"
    ),
    output_path: Path = typer.Option(
        Path("output"), "--output", "-o", help="Output directory"
    ),
    format_type: str = typer.Option(
        "json", "--format", "-f", help="Output format (json, csv, xlsx)"
    ),
    ocr_engine: str = typer.Option(
        "auto", "--ocr", help="OCR engine (tesseract, easyocr, auto)"
    ),
    batch_size: int = typer.Option(
        10, "--batch-size", help="Batch processing size"
    ),
) -> None:
    """Extract data from documents using OCR and AI."""
    
    console.print(
        Panel(
            f"[bold green]Document Extraction[/bold green]\n"
            f"Input: {input_path}\n"
            f"Output: {output_path}\n"
            f"Format: {format_type}\n"
            f"OCR Engine: {ocr_engine}\n"
            f"Batch Size: {batch_size}",
            title="Processing Configuration"
        )
    )
    
    try:
        results = asyncio.run(
            process_documents(
                input_path=input_path,
                output_path=output_path,
                format_type=format_type,
                ocr_engine=ocr_engine,
                batch_size=batch_size,
            )
        )
        
        # Display results
        table = Table(title="Extraction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in results.items():
            if isinstance(value, (int, float)):
                table.add_row(key, str(value))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Extraction failed: {e}[/bold red]")
        raise typer.Exit(1)


@process_app.command("compare")
def compare_loans(
    loan_file1: Path = typer.Option(
        ..., "--loan1", help="First loan file"
    ),
    loan_file2: Path = typer.Option(
        ..., "--loan2", help="Second loan file"
    ),
    output_path: Path = typer.Option(
        Path("comparison_output.json"), 
        "--output", "-o", 
        help="Comparison output file"
    ),
) -> None:
    """Compare two loan offers."""
    
    console.print(
        Panel(
            f"[bold green]Loan Comparison[/bold green]\n"
            f"Loan 1: {loan_file1}\n"
            f"Loan 2: {loan_file2}\n"
            f"Output: {output_path}",
            title="Comparison Configuration"
        )
    )
    
    try:
        # TODO: Implement loan comparison logic
        console.print("[bold yellow]Loan comparison feature coming soon![/bold yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Comparison failed: {e}[/bold red]")
        raise typer.Exit(1)


@cli_app.command("status")
def show_status() -> None:
    """Show system status and configuration."""
    
    # Create status table
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    # System information
    version_info = get_version_info()
    table.add_row("Version", version_info, "Current system version")
    table.add_row("Environment", settings.ENVIRONMENT, "Deployment environment")
    table.add_row("Debug Mode", str(settings.DEBUG), "Development mode status")
    
    # API Keys status (masked for security)
    api_status = "✓ Configured" if settings.ANTHROPIC_API_KEY else "✗ Missing"
    table.add_row("Anthropic API", api_status, "Claude AI service")
    
    openai_status = "✓ Configured" if settings.OPENAI_API_KEY else "✗ Missing"
    table.add_row("OpenAI API", openai_status, "OpenAI GPT service")
    
    # Database status
    db_status = "✓ Configured" if settings.DATABASE_URL else "✗ Missing"
    table.add_row("Database", db_status, settings.DATABASE_URL or "Not configured")
    
    # Redis status
    redis_status = "✓ Configured" if settings.REDIS_URL else "✗ Missing"
    table.add_row("Redis Cache", redis_status, settings.REDIS_URL or "Not configured")
    
    console.print(table)


@cli_app.command("config")
def show_config() -> None:
    """Show current system configuration."""
    
    console.print(
        Panel(
            "[bold yellow]Current Configuration[/bold yellow]\n"
            "Environment variables and system settings",
            title="Configuration"
        )
    )
    
    # Display configuration
    config_table = Table(title="Environment Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    important_settings = [
        ("APP_NAME", settings.APP_NAME),
        ("ENVIRONMENT", settings.ENVIRONMENT),
        ("DEBUG", str(settings.DEBUG)),
        ("LOG_LEVEL", settings.LOG_LEVEL),
        ("PORT", str(settings.PORT)),
        ("HOST", settings.HOST),
        ("DEFAULT_MODEL", settings.DEFAULT_MODEL),
        ("MAX_TOKENS", str(settings.MAX_TOKENS)),
    ]
    
    for setting, value in important_settings:
        config_table.add_row(setting, value)
    
    console.print(config_table)
    
    console.print(
        Panel(
            "[bold blue]For sensitive configuration (API keys, database URLs), "
            "please check your .env file.[/bold blue]",
            title="Security Note"
        )
    )


@cli_app.command("test")
def run_tests(
    coverage: bool = typer.Option(True, "--coverage", help="Generate coverage report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose test output"),
    test_path: Optional[str] = typer.Option(None, "--path", help="Specific test path"),
) -> None:
    """Run the test suite."""
    
    import pytest
    
    console.print(
        Panel(
            f"[bold blue]Running Test Suite[/bold blue]\n"
            f"Coverage: {'Enabled' if coverage else 'Disabled'}\n"
            f"Verbose: {'Enabled' if verbose else 'Disabled'}",
            title="Testing"
        )
    )
    
    # Build pytest command
    pytest_args = []
    
    if coverage:
        pytest_args.extend(["--cov=src", "--cov-report=term-missing"])
    
    if verbose:
        pytest_args.append("-v")
    
    if test_path:
        pytest_args.append(test_path)
    
    try:
        exit_code = pytest.main(pytest_args)
        if exit_code != 0:
            console.print("[bold red]Tests failed![/bold red]")
            raise typer.Exit(1)
        else:
            console.print("[bold green]All tests passed![/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]Test execution failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    cli_app()

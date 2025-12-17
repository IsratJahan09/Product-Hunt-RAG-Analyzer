"""
Command-line interface for Product Hunt RAG Analyzer.

This module provides a CLI with subcommands for building indices,
running analysis, viewing statistics, and starting the API server.
"""

import sys
import argparse
import io
from pathlib import Path
from typing import Optional, Dict, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Fix Unicode encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from src.utils.config import ConfigManager
from src.utils.logger import Logger, get_logger
from src.utils.index_builder import IndexBuilder
from src.main import AnalysisPipeline
from src.modules.report_generation import ReportGenerator


# Create console with legacy_windows=False to use UTF-8 mode
console = Console(legacy_windows=False, force_terminal=True)
logger = None  # Will be initialized after argument parsing


class CLIError(Exception):
    """Exception raised for CLI-specific errors."""
    pass


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging for CLI.
    
    Args:
        verbose: If True, set log level to DEBUG
    """
    global logger
    log_level = "DEBUG" if verbose else "INFO"
    Logger.setup(log_level=log_level)
    logger = get_logger(__name__)


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ConfigManager instance
        
    Raises:
        CLIError: If config loading fails
    """
    try:
        if config_path:
            console.print(f"[cyan]Loading config from: {config_path}[/cyan]")
            config = ConfigManager(config_path=config_path)
        else:
            console.print("[cyan]Loading default configuration[/cyan]")
            config = ConfigManager()
        
        return config
        
    except Exception as e:
        raise CLIError(f"Failed to load configuration: {e}")


def build_index_command(args: argparse.Namespace) -> int:
    """
    Execute build-index command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("\n[bold cyan]Building FAISS Indices[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # Validate dataset path
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise CLIError(f"Dataset path does not exist: {dataset_path}")
        
        # Load configuration
        config = load_config(args.config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[green]✓[/green] Dataset path: {dataset_path}")
        console.print(f"[green]✓[/green] Output directory: {output_dir}")
        console.print(f"[green]✓[/green] Index type: {args.index_type}")
        
        # Initialize index builder
        console.print("\n[cyan]Initializing index builder...[/cyan]")
        builder = IndexBuilder()
        
        # Build indices
        console.print("\n[cyan]Building indices...[/cyan]")
        
        build_config = {
            "products_file": "products.jsonl",
            "reviews_file": "reviews.jsonl",
            "output_dir": str(output_dir),
            "product_index_type": args.index_type,
            "review_index_type": args.index_type,
            "batch_size": config.get("processing.batch_size", 32)
        }
        
        results = builder.build_all_indices(
            dataset_path=str(dataset_path),
            config=build_config
        )
        
        # Display results
        console.print("\n[bold green]✓ Index Building Complete![/bold green]")
        console.print("=" * 60)
        
        # Create results table
        table = Table(title="Build Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Products", justify="right", style="green")
        table.add_column("Reviews", justify="right", style="green")
        
        if results.get("products"):
            prod_result = results["products"]
            table.add_row(
                "Total Items",
                str(prod_result.get("total_products", 0)),
                str(results.get("reviews", {}).get("total_reviews", 0))
            )
            table.add_row(
                "Indexed Items",
                str(prod_result.get("indexed_products", 0)),
                str(results.get("reviews", {}).get("indexed_reviews", 0))
            )
            table.add_row(
                "Skipped Items",
                str(prod_result.get("skipped_products", 0)),
                str(results.get("reviews", {}).get("skipped_reviews", 0))
            )
            table.add_row(
                "Build Time",
                prod_result.get("build_time_formatted", "N/A"),
                results.get("reviews", {}).get("build_time_formatted", "N/A")
            )
        
        console.print(table)
        
        if results.get("errors"):
            console.print("\n[yellow]⚠ Warnings:[/yellow]")
            for error in results["errors"]:
                console.print(f"  [yellow]•[/yellow] {error}")
        
        console.print(f"\n[green]Total time: {results.get('total_time_formatted', 'N/A')}[/green]")
        console.print(f"[green]Indices saved to: {output_dir}[/green]\n")
        
        return 0
        
    except CLIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n", style="red")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


def analyze_command(args: argparse.Namespace) -> int:
    """
    Execute analyze command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("\n[bold cyan]Running Competitive Analysis[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # Validate product idea
        if not args.product_idea or not args.product_idea.strip():
            raise CLIError("Product idea cannot be empty")
        
        if len(args.product_idea) < 10:
            raise CLIError("Product idea must be at least 10 characters")
        
        # Load configuration
        config = load_config(args.config)
        
        # Display parameters
        console.print(f"[cyan]Product Idea:[/cyan] {args.product_idea}")
        console.print(f"[cyan]Max Competitors:[/cyan] {args.max_competitors}")
        console.print(f"[cyan]Output Format:[/cyan] {args.output_format}")
        
        # Initialize pipeline
        console.print("\n[cyan]Initializing analysis pipeline...[/cyan]")
        pipeline = AnalysisPipeline(config=config)
        
        # Load indices
        indices_dir = Path(config.get("storage.indices_dir", "./data/indices"))
        product_index_path = str(indices_dir / "products")
        review_index_path = str(indices_dir / "reviews")
        
        console.print(f"[cyan]Loading indices from: {indices_dir}[/cyan]")
        
        try:
            pipeline.load_indices(
                product_index_path=product_index_path,
                review_index_path=review_index_path
            )
            console.print("[green]✓[/green] Indices loaded successfully")
        except FileNotFoundError:
            raise CLIError(
                f"Index files not found in {indices_dir}. "
                f"Run 'build-index' command first."
            )
        
        # Run analysis
        console.print("\n[cyan]Running analysis...[/cyan]")
        console.print("[dim]This may take a few minutes...[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing competitors...", total=None)
            
            result = pipeline.run_analysis(
                product_idea=args.product_idea,
                max_competitors=args.max_competitors,
                output_format=args.output_format
            )
            
            progress.update(task, completed=True)
        
        # Check status
        if result.get("status") == "failed":
            raise CLIError(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Display results summary
        console.print("\n[bold green]✓ Analysis Complete![/bold green]")
        console.print("=" * 60)
        
        # Create summary table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Analysis ID", result.get("analysis_id", "N/A"))
        table.add_row("Status", result.get("status", "N/A"))
        table.add_row(
            "Competitors Identified",
            str(len(result.get("competitors_identified", [])))
        )
        table.add_row(
            "Confidence Score",
            f"{result.get('confidence_score', 0):.2f}"
        )
        table.add_row(
            "Processing Time",
            f"{result.get('processing_time_ms', 0)}ms"
        )
        
        console.print(table)
        
        # Display competitors
        competitors = result.get("competitors_identified", [])
        if competitors:
            console.print("\n[cyan]Competitors Identified:[/cyan]")
            for i, comp in enumerate(competitors, 1):
                console.print(f"  {i}. {comp}")
        
        # Display warnings
        if result.get("warnings"):
            console.print("\n[yellow]⚠ Warnings:[/yellow]")
            for warning in result["warnings"]:
                console.print(f"  [yellow]•[/yellow] {warning}")
        
        # Save report if output file specified
        if args.output_file:
            console.print(f"\n[cyan]Saving report to: {args.output_file}[/cyan]")
            
            report_generator = ReportGenerator()
            output_path = Path(args.output_file)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get full report
            full_report = result.get("full_report", {})
            
            # Export based on format
            if args.output_format == "json":
                report_generator.export_json(full_report, str(output_path))
            elif args.output_format == "markdown":
                report_generator.export_markdown(full_report, str(output_path))
            elif args.output_format == "pdf":
                report_generator.export_pdf(full_report, str(output_path))
            
            console.print(f"[green]✓[/green] Report saved to: {output_path}")
        else:
            console.print("\n[dim]No output file specified. Report not saved.[/dim]")
        
        console.print()
        return 0
        
    except CLIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n", style="red")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


def stats_command(args: argparse.Namespace) -> int:
    """
    Execute stats command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("\n[bold cyan]Dataset Statistics[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # Validate indices directory
        indices_dir = Path(args.indices_dir)
        if not indices_dir.exists():
            raise CLIError(f"Indices directory does not exist: {indices_dir}")
        
        console.print(f"[cyan]Indices directory:[/cyan] {indices_dir}")
        
        # Initialize pipeline
        console.print("\n[cyan]Loading indices...[/cyan]")
        pipeline = AnalysisPipeline()
        
        product_index_path = str(indices_dir / "products")
        review_index_path = str(indices_dir / "reviews")
        
        try:
            load_result = pipeline.load_indices(
                product_index_path=product_index_path,
                review_index_path=review_index_path
            )
            console.print("[green]✓[/green] Indices loaded successfully")
        except FileNotFoundError:
            raise CLIError(
                f"Index files not found in {indices_dir}. "
                f"Run 'build-index' command first."
            )
        
        # Get statistics
        stats = pipeline.get_dataset_stats()
        
        # Display statistics
        console.print("\n[bold green]Dataset Statistics[/bold green]")
        console.print("=" * 60)
        
        # Create statistics table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Indices Loaded", "✓" if stats.get("indices_loaded") else "✗")
        table.add_row("Product Index Size", str(stats.get("product_index_size", 0)))
        table.add_row("Review Index Size", str(stats.get("review_index_size", 0)))
        table.add_row("Product Index Path", stats.get("product_index_path", "N/A"))
        table.add_row("Review Index Path", stats.get("review_index_path", "N/A"))
        
        if load_result:
            table.add_row(
                "Load Time",
                f"{load_result.get('load_time_ms', 0):.2f}ms"
            )
        
        console.print(table)
        console.print()
        
        return 0
        
    except CLIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n", style="red")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


def serve_command(args: argparse.Namespace) -> int:
    """
    Execute serve command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("\n[bold cyan]Starting FastAPI Server[/bold cyan]")
    console.print("=" * 60)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Get server settings
        host = args.host
        port = args.port
        
        console.print(f"[cyan]Host:[/cyan] {host}")
        console.print(f"[cyan]Port:[/cyan] {port}")
        console.print(f"[cyan]Docs:[/cyan] http://{host}:{port}/docs")
        console.print(f"[cyan]ReDoc:[/cyan] http://{host}:{port}/redoc")
        
        console.print("\n[green]Starting server...[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        # Import and run uvicorn
        import uvicorn
        
        uvicorn.run(
            "src.api.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Server stopped by user[/yellow]\n")
        return 0
    except CLIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n", style="red")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """
    Execute evaluate command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("\n[bold cyan]Running System Evaluation[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from src.evaluation.runner import EvaluationRunner
        
        # Initialize runner
        console.print("[cyan]Initializing evaluation runner...[/cyan]")
        runner = EvaluationRunner()
        
        # Display parameters
        console.print(f"[cyan]Evaluation Type:[/cyan] {args.eval_type}")
        console.print(f"[cyan]Evaluation Data:[/cyan] {args.eval_dir}")
        console.print(f"[cyan]Output Format:[/cyan] {args.output_format}")
        console.print(f"[cyan]Output Directory:[/cyan] {args.output_dir}")
        
        # Run evaluation based on type
        console.print("\n[cyan]Running evaluation...[/cyan]")
        console.print("[dim]This may take several minutes...[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating system...", total=None)
            
            if args.eval_type == "full":
                results = runner.run_full_evaluation(
                    eval_dir=args.eval_dir,
                    indices_dir=args.indices_dir,
                    output_format=args.output_format,
                    output_dir=args.output_dir
                )
            elif args.eval_type == "retrieval":
                results = runner.run_retrieval_evaluation(
                    eval_dir=args.eval_dir,
                    indices_dir=args.indices_dir,
                    output_format=args.output_format,
                    output_dir=args.output_dir
                )
            elif args.eval_type == "sentiment":
                results = runner.run_sentiment_evaluation(
                    eval_dir=args.eval_dir,
                    output_format=args.output_format,
                    output_dir=args.output_dir
                )
            elif args.eval_type == "feature_gaps":
                results = runner.run_feature_gaps_evaluation(
                    eval_dir=args.eval_dir,
                    output_format=args.output_format,
                    output_dir=args.output_dir
                )
            else:
                raise CLIError(f"Unknown evaluation type: {args.eval_type}")
            
            progress.update(task, completed=True)
        
        # Display results summary
        summary = results.get("summary", {})
        overall_status = summary.get("overall_status", "UNKNOWN")
        
        console.print("\n[bold green]✓ Evaluation Complete![/bold green]")
        console.print("=" * 60)
        
        # Create summary table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Overall Status", overall_status)
        table.add_row("Total Validations", str(summary.get("total_validations", 0)))
        table.add_row("Passed", str(summary.get("passed", 0)))
        table.add_row("Failed", str(summary.get("failed", 0)))
        table.add_row(
            "Duration",
            f"{summary.get('total_duration_seconds', 0):.2f}s"
        )
        
        console.print(table)
        
        # Display validation details
        validations = summary.get("validations", [])
        if validations:
            console.print("\n[cyan]Validation Details:[/cyan]")
            for validation in validations:
                status_symbol = "✓" if validation["passed"] else "✗"
                status_color = "green" if validation["passed"] else "red"
                console.print(
                    f"  [{status_color}]{status_symbol}[/{status_color}] "
                    f"[{validation['level']}] {validation['name']}: {validation['status']}"
                )
        
        # Display report location
        console.print(f"\n[green]Report saved to: {args.output_dir}[/green]")
        
        # Return appropriate exit code
        console.print()
        return 0 if overall_status == "PASS" else 1
        
    except CLIError as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}\n", style="red")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser with all subcommands.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="product-hunt-rag-analyzer",
        description="AI-powered competitive analysis for Product Hunt products",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build indices from dataset
  python -m src.cli build-index --dataset-path dataset --index-type flat
  
  # Run analysis
  python -m src.cli analyze --product-idea "A task management app with AI" --max-competitors 5
  
  # View statistics
  python -m src.cli stats --indices-dir data/indices
  
  # Start API server
  python -m src.cli serve --host 0.0.0.0 --port 8000
  
  # Run evaluation
  python -m src.cli evaluate --eval-type full --output-format html
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True
    )
    
    # ===== build-index command =====
    build_parser = subparsers.add_parser(
        "build-index",
        help="Build FAISS indices from Product Hunt dataset",
        description="Build both product and review FAISS indices from dataset"
    )
    
    build_parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to Product Hunt dataset directory or JSON file"
    )
    
    build_parser.add_argument(
        "--output-dir",
        type=str,
        default="data/indices",
        help="Directory to save FAISS indices (default: data/indices)"
    )
    
    build_parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type (default: flat)"
    )
    
    build_parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    # ===== analyze command =====
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run competitive analysis for a product idea",
        description="Identify competitors and generate competitive intelligence report"
    )
    
    analyze_parser.add_argument(
        "--product-idea",
        type=str,
        required=True,
        help="Product idea description (minimum 10 characters)"
    )
    
    analyze_parser.add_argument(
        "--max-competitors",
        type=int,
        default=5,
        help="Number of competitors to analyze (default: 5)"
    )
    
    analyze_parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "markdown", "pdf"],
        help="Report output format (default: json)"
    )
    
    analyze_parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save report file (optional)"
    )
    
    analyze_parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    # ===== stats command =====
    stats_parser = subparsers.add_parser(
        "stats",
        help="Display dataset statistics from indices",
        description="Show statistics about loaded FAISS indices"
    )
    
    stats_parser.add_argument(
        "--indices-dir",
        type=str,
        default="data/indices",
        help="Directory containing FAISS indices (default: data/indices)"
    )
    
    # ===== serve command =====
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start FastAPI web service",
        description="Start the REST API server using uvicorn"
    )
    
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    serve_parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file"
    )
    
    # ===== evaluate command =====
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run system evaluation and validation",
        description="Run evaluation procedures to validate system quality and performance"
    )
    
    evaluate_parser.add_argument(
        "--eval-type",
        type=str,
        required=True,
        choices=["full", "retrieval", "sentiment", "feature_gaps"],
        help="Type of evaluation to run (full, retrieval, sentiment, feature_gaps)"
    )
    
    evaluate_parser.add_argument(
        "--eval-dir",
        type=str,
        default="dataset/evaluation",
        help="Path to evaluation data directory (default: dataset/evaluation)"
    )
    
    evaluate_parser.add_argument(
        "--indices-dir",
        type=str,
        default="data/indices",
        help="Path to FAISS indices directory (default: data/indices)"
    )
    
    evaluate_parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "html", "markdown"],
        help="Report output format (default: json)"
    )
    
    evaluate_parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/evaluation",
        help="Directory to save evaluation reports (default: reports/evaluation)"
    )
    
    return parser


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Create parser
    parser = create_parser()
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Display banner
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Product Hunt RAG Analyzer[/bold cyan]\n"
            "[dim]AI-powered competitive analysis for Product Hunt products[/dim]",
            border_style="cyan"
        )
    )
    
    # Execute command
    try:
        if args.command == "build-index":
            return build_index_command(args)
        elif args.command == "analyze":
            return analyze_command(args)
        elif args.command == "stats":
            return stats_command(args)
        elif args.command == "serve":
            return serve_command(args)
        elif args.command == "evaluate":
            return evaluate_command(args)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Operation cancelled by user[/yellow]\n")
        return 1
    except Exception as e:
        console.print(f"\n[bold red]✗ Fatal error:[/bold red] {e}\n", style="red")
        if args.verbose:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())

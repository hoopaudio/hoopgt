import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from typing import Optional
from .types import TargetHardware, OptimizationLevel
from .engine import OptimizationEngine

console = Console()
app = typer.Typer(
    name="hoopgt",
    help="🏀 HoopGT SDK - Model Optimization Platform",
    rich_markup_mode="rich",
)


@app.command()
def optimize(
    model_path: str = typer.Argument(..., help="Path to the model to optimize"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for optimized model"
    ),
    level: OptimizationLevel = typer.Option(
        "balanced", "--level", "-l", help="Optimization level"
    ),
    target: TargetHardware = typer.Option(
        ..., "--target", "-t", help="Target hardware platform (e.g., apple-silicon)"
    ),
    quantize: bool = typer.Option(
        False, "--quantize", "-q", help="Enable quantization optimization"
    ),
):
    """
    🔧 Optimize a model for a specific target hardware.
    """
    rprint(f"[bold green]🏀 HoopGT Optimizer[/bold green]")
    table = Table(box=None, show_header=False)
    table.add_column(style="blue")
    table.add_column()
    table.add_row("Model Path:", model_path)
    table.add_row("Target:", target.value)
    table.add_row("Level:", level.value)
    table.add_row("Quantize:", str(quantize))
    console.print(table)

    try:
        engine = OptimizationEngine()
        results = engine.run(
            model_path=model_path,
            target=target,
            level=level,
            quantize=quantize,
            output_path=output,
        )

        rprint("\n[bold green]📊 Optimization Results[/bold green]")
        results_table = Table(show_header=False)
        results_table.add_column(style="cyan")
        results_table.add_column()

        results_table.add_row("Final Target", results["target"])

        if results["quantization_enabled"]:
            results_table.add_row(
                "Quantization Method",
                f"{results['quantization_method']} (auto-selected)",
            )
            stats = results["quantization_stats"]
            if stats:
                results_table.add_row(
                    "Size Reduction", f"{stats['reduction_ratio']:.2f}x"
                )
                results_table.add_row(
                    "Original Size", f"{stats['original_size_mb']:.3f} MB"
                )
                results_table.add_row(
                    "Optimized Size", f"{stats['quantized_size_mb']:.3f} MB"
                )
        else:
            results_table.add_row("Quantization", "Not enabled")

        if results["output_path"]:
            results_table.add_row("Saved To", results["output_path"])
        else:
            results_table.add_row("Saved To", "Not saved (no --output path provided)")

        console.print(results_table)
        rprint("\n[bold green]🎉 Optimization complete![/bold green]")

    except Exception as e:
        rprint(f"\n[bold red]❌ An error occurred during optimization:[/bold red]")
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(code=1)


@app.command()
def deploy(
    model_name: str = typer.Argument(..., help="Name of the model to deploy"),
    port: int = typer.Option(3000, "--port", "-p", help="Port for the service"),
    target: TargetHardware = typer.Option(
        ..., "--target", "-t", help="Target hardware for deployment"
    ),
):
    """
    🚀 Deploy a model (placeholder for your implementation).
    """
    rprint(f"[bold green]🏀 HoopGT Deployer[/bold green]")
    rprint(f"[blue]Model:[/blue] {model_name}")
    rprint(f"[blue]Target:[/blue] {target.value}")
    rprint(f"[blue]Port:[/blue] {port}")

    # TODO: Add your deployment logic here
    rprint("[yellow]⚠️  Deployment logic not implemented yet[/yellow]")
    rprint("[dim]This is a boilerplate - add your implementation![/dim]")


@app.command()
def list():
    """
    📋 List available models (placeholder for your implementation).
    """
    rprint("[bold green]🏀 Available Models[/bold green]")

    # Create a sample table
    table = Table(title="Model Registry")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Target", style="blue")
    table.add_column("Status", style="green")

    # TODO: Replace with your actual model registry
    table.add_row(
        "sample-model", "PyTorch", TargetHardware.APPLE_SILICON.value, "Not Implemented"
    )
    table.add_row(
        "bert-base", "Transformer", TargetHardware.X86_SERVER.value, "Not Implemented"
    )

    console.print(table)
    rprint("[dim]This is a boilerplate - add your model registry![/dim]")


@app.command()
def info():
    """
    ℹ️  Show system information and supported targets.
    """
    rprint("[bold green]🏀 HoopGT System Info[/bold green]")

    # Detect current hardware
    import platform

    machine = platform.machine()
    system = platform.system()

    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Version", "0.1.0")
    table.add_row("System", f"{system} {machine}")
    table.add_row("Python", "3.9+")
    table.add_row("Status", "MVP Boilerplate")

    # Show supported targets
    if system == "Darwin" and machine == "arm64":
        table.add_row("Detected Target", TargetHardware.APPLE_SILICON.value)
    elif machine in ["x86_64", "AMD64"]:
        table.add_row("Detected Target", TargetHardware.X86_SERVER.value)
    else:
        table.add_row("Detected Target", "unknown")

    console.print(table)

    # Show all supported options
    rprint("\n[bold green]📋 Supported Options[/bold green]")

    targets_table = Table(title="Target Hardware")
    targets_table.add_column("Target", style="cyan")
    targets_table.add_column("Description", style="white")

    target_descriptions = {
        TargetHardware.APPLE_SILICON: "Apple M1/M2/M3/M4 chips",
        TargetHardware.X86_SERVER: "Intel/AMD x86 servers",
        TargetHardware.ARM_MOBILE: "ARM mobile/embedded devices",
        TargetHardware.NVIDIA_JETSON: "NVIDIA Jetson devices",
    }

    for target, desc in target_descriptions.items():
        targets_table.add_row(target.value, desc)

    console.print(targets_table)

    levels_table = Table(title="Optimization Levels")
    levels_table.add_column("Level", style="cyan")
    levels_table.add_column("Description", style="white")

    levels_table.add_row(
        OptimizationLevel.LIGHT.value, "Minimal optimization, fastest compilation"
    )
    levels_table.add_row(
        OptimizationLevel.BALANCED.value, "Balanced optimization and compilation time"
    )
    levels_table.add_row(
        OptimizationLevel.AGGRESSIVE.value, "Maximum optimization, slower compilation"
    )

    console.print(levels_table)


if __name__ == "__main__":
    app()

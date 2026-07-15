from pathlib import Path

import typer

app = typer.Typer(help="Inspect model checkpoints.")


@app.command("list")
def list_checkpoints(directory: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """List checkpoint files below DIRECTORY."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        typer.echo("No checkpoint files found.")
        return
    for path in files:
        typer.echo(path.relative_to(directory))

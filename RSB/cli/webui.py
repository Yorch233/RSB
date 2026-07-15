import typer

app = typer.Typer(help="Launch or inspect the optional web interface.")


@app.command()
def status() -> None:
    """Report whether the optional web interface is available."""
    typer.echo("The optional Web UI is not bundled with this repository.")

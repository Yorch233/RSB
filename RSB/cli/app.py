import typer

from RSB.cli import checkpoint, config, dataset, inference, metric, train, webui
from RSB.cli.tui import ThemedTyperGroup

app = typer.Typer(
    name="rsb",
    help="Regularized Schrodinger Bridge command line interface.",
    cls=ThemedTyperGroup,
)
app.command("config")(config.configure)
app.add_typer(checkpoint.app, name="checkpoint")
app.add_typer(inference.app, name="inference")
app.add_typer(train.app, name="train")
app.add_typer(dataset.app, name="dataset")
app.command("metric")(metric.calculate)
app.add_typer(webui.app, name="webui")


@app.callback()
def main() -> None:
    """Run RSB training, inference, evaluation, and demos."""


if __name__ == "__main__":
    app()

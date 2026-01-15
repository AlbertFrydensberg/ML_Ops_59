from typing import Annotated

import typer

from ml_ops_59.data import save_raw_csv, data_loader, validate_data

app = typer.Typer(help="ML Ops 59 CLI")


@app.command()
def data_download(
    out: Annotated[
        str,
        typer.Option("--out", "-o", help="Where to save the CSV (repo path)"),
    ] = "data/raw/wine.csv",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Overwrite file if it already exists"),
    ] = False,
):
    """Download wine dataset and save it under data/raw/."""
    save_raw_csv(out_path=out, overwrite=overwrite)
    typer.echo(f"Saved dataset to: {out}")


@app.command()
def data_check():
    """Load the dataset and run simple validation checks."""
    df = data_loader()
    validate_data(df)
    typer.echo(f"OK: loaded {df.shape[0]} rows, {df.shape[1]} columns")


def main():
    app()


if __name__ == "__main__":
    app()

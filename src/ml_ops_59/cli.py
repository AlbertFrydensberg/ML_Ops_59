import typer
from ml_ops_59.data import save_raw_csv, data_loader, validate_data

app = typer.Typer()


@app.command()
def data_download(
    out: str = "data/raw/wine.csv",
    overwrite: bool = False,
):
    """Download wine dataset and save to data/raw."""
    save_raw_csv(out_path=out, overwrite=overwrite)
    typer.echo(f"Saved dataset to {out}")

@app.command()
def data_check():
    df = data_loader()
    validate_data(df)
    typer.echo(f"✅ OK: loaded {df.shape[0]} rows, {df.shape[1]} columns")



if __name__ == "__main__":
    app()

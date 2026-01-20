"""Compute simple dataset statistics for M19."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def main() -> None:
    if not DATA_DIR.exists():
        print("No data directory found.")
        return

    csv_files = list(DATA_DIR.rglob("*.csv"))

    if not csv_files:
        print("No CSV files found in data directory.")
        return

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        df = pd.read_csv(csv_file)

        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print("Columns:")
        print(list(df.columns))

        print("Missing values per column:")
        print(df.isna().sum())


if __name__ == "__main__":
    main()

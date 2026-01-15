from ml_ops_59.data import data_loader
import numpy as np
import pandas as pd


def test_data_loads_as_dataframe():
    df = data_loader()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_expected_columns_present():
    """
    Wine datasets typically contain a 'class' label and several numeric features.
    This test checks the most common column names, but stays flexible:
    - requires 'class'
    - requires at least a reasonable number of feature columns
    """
    df = data_loader()
    assert "class" in df.columns, "Expected a 'class' column (target label)."
    assert df.shape[1] >= 5, "Expected at least 5 columns (features + target)."


def test_no_missing_values_in_critical_columns():
    """
    For our ML training, missing values in the target or core features break training.
    """
    df = data_loader()
    # target must not be missing
    assert df["class"].notna().all(), "Found missing values in 'class'."

    assert df.isna().sum().sum() == 0, "Dataset contains missing values."


def test_class_is_numeric_and_in_reasonable_range():
    """
    We test that our labels are integers.
    """
    df = data_loader()

    assert pd.api.types.is_numeric_dtype(df["class"]), "'class' must be numeric."

    qmin = df["class"].min()
    qmax = df["class"].max()
    assert qmin >= 0, f"Unexpected minimum class: {qmin}"
    assert qmax <= 10, f"Unexpected maximum class: {qmax}"


def test_feature_columns_are_numeric():
    """
    Most wine datasets are fully numeric. This ensures your model won't choke on strings.
    """
    df = data_loader()

    feature_cols = [c for c in df.columns if c != "class"]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    assert len(feature_cols) > 0, "No feature columns found."
    assert not non_numeric, f"Non-numeric feature columns found: {non_numeric}"


def test_no_duplicate_rows():
    """
    Duplicates can bias training/validation splits and metrics.
    """
    df = data_loader()
    dup_count = df.duplicated().sum()
    assert dup_count == 0, f"Found {dup_count} duplicate rows."


# src/ml_ops_59/sweep_entry.py

from ml_ops_59.train import main

if __name__ == "__main__":
    # Force the sweep code path, regardless of what's in config.yaml
    main(overrides=["task=sweep"])

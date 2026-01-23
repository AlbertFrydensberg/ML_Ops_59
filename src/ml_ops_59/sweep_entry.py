# src/ml_ops_59/sweep_entry.py
import sys

from ml_ops_59.train import main

if __name__ == "__main__":
    # Hydra reads CLI overrides from sys.argv
    # Ensure we always run the sweep path
    sys.argv.append("task=sweep")
    main()

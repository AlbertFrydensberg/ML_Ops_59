# Import Hydra for config file
import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ml_ops_59.data import data_loader
from ml_ops_59.evaluate import compute_confusion_matrix, compute_metrics
from ml_ops_59.model import create_model
from ml_ops_59.visualize import plot_confusion_matrix
from ml_ops_59.visualize import generate_shap_explanations
from ml_ops_59.wandb_logger import WandBLogger

"""
Training a KNN on wine data (single run)
"""


def train(n_neighbors: int = 5, test_size: float = 0.2, seed: int = 42) -> float:
    """Train a KNN on wine data (single run) and return accuracy."""

    np.random.seed(seed)

    df = data_loader()
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = create_model(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    generate_shap_explanations(model, X_train, X_test, X.columns.tolist())

    return float(acc)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train_hydra(cfg: DictConfig):
    acc = train(
        n_neighbors=cfg.model.n_neighbors,
        test_size=cfg.data.test_size,
        seed=cfg.data.seed,
    )

    print(f"Validation Accuracy: {acc:.4f}")

    return acc


"""
One WandB sweep trial (called by wandb.agent)
"""


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def sweep_trial(cfg: DictConfig):
    """
    One WandB sweep trial (called by wandb.agent)
    Does Stratified K-Fold CV and logs mean/std metrics + confusion matrix from OOF preds.
    W&B will override cfg.model.n_neighbors with sweep values
    """
    wandb_logger = WandBLogger(project_name=cfg.wandb.project_name, enabled=cfg.wandb.enabled)

    wandb_config = wandb_logger.config
    k = int(wandb_config.get("K", cfg.model.n_neighbors))  # W&B sweep or Hydra default
    cv_folds = int(wandb_config.get("cv_folds", cfg.training.cv_folds))  # W&B or Hydra
    trial_seed = int(wandb_config.get("seed", cfg.data.seed))  # W&B or Hydra
    weights = str(wandb_config.get("weights", cfg.model.weights)).lower()  # W&B or Hydra
    p = int(wandb_config.get("p", cfg.model.p))  # W&B or Hydra

    # Validation
    if weights not in {"uniform", "distance"}:
        print(f"Warning: invalid weights='{weights}', defaulting to 'uniform'")
        weights = "uniform"

    if p not in {1, 2}:
        print(f"Warning: invalid p='{p}', defaulting to 2")
        p = 2

    # Log all config to W&B
    wandb_logger.log_config(
        {
            "K": k,
            "weights": weights,
            "p": p,
            "cv_folds": cv_folds,
            "seed": trial_seed,
            "cv_strategy": "StratifiedKFold(shuffle=True)",
            "scaler": "StandardScaler",
            "model": "KNeighborsClassifier",
            "metric": "minkowski",
        }
    )

    # Set random seed
    np.random.seed(trial_seed)

    # Load data
    df = data_loader()
    X = df.drop(columns=["class"]).values
    y = df["class"].values

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=trial_seed)

    fold_accs = []
    oof_preds = np.empty_like(y)
    fold_rows = []

    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train model
        model = create_model(n_neighbors=k, weights=weights, p=p)
        model.fit(X_train, y_train)

        # Predict and evaluate
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds

        acc = accuracy_score(y_val, preds)
        fold_accs.append(acc)

        # Compute metrics
        num_classes = len(np.unique(y))
        m = compute_metrics(y_val, preds, num_classes)

        wandb_logger.log_metrics(
            {
                "fold": fold_idx,
                "fold_accuracy": acc,
                "fold_precision": m["avg_precision"],
                "fold_recall": m["avg_recall"],
                "fold_f1": m["avg_f1"],
            },
            step=fold_idx,
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "accuracy": acc,
                "precision": m["avg_precision"],
                "recall": m["avg_recall"],
                "f1": m["avg_f1"],
            }
        )

    # Table of folds (optional but nice)
    wandb_logger.log_table("cv_folds_table", fold_rows)

    acc_mean = float(np.mean(fold_accs))
    acc_std = float(np.std(fold_accs))

    wandb_logger.log_metrics(
        {
            "cv_accuracy_mean": acc_mean,
            "cv_accuracy_std": acc_std,
        }
    )

    num_classes = cfg.training.num_classes  # From Hydra config
    overall = compute_metrics(y, oof_preds, num_classes)
    wandb_logger.log_metrics(
        {
            "cv_precision_oof": overall["avg_precision"],
            "cv_recall_oof": overall["avg_recall"],
            "cv_f1_oof": overall["avg_f1"],
        }
    )

    class_names = cfg.training.class_names  # From Hydra config
    confusion = compute_confusion_matrix(y, oof_preds, class_names=class_names)
    plot_confusion_matrix(confusion, class_names, "confusion_matrix.png")

    wandb_logger.log_image("confusion_matrix", "confusion_matrix.png")
    wandb_logger.log_confusion_matrix(y, oof_preds, class_names)

    wandb_logger.finish()
    return acc_mean


if __name__ == "__main__":
    train_hydra()

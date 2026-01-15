import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_ops_59.wandb_logger import WandBLogger
from ml_ops_59.data import data_loader
from ml_ops_59.evaluate import compute_confusion_matrix, compute_metrics
from ml_ops_59.model import create_model
from ml_ops_59.visualize import plot_confusion_matrix

"""
Training a KNN on wine data (single run)
"""


def train(n_neighbors=5, test_size=0.2, seed=42):
    df = data_loader()

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = create_model(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Validation Accuracy: {acc:.4f}")
    return acc


"""
One WandB sweep trial (called by wandb.agent)
"""

def sweep_trial(project_name="MLops_59", seed=42):
    np.random.seed(seed)

    df = data_loader()
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logger owns wandb.init()
    wandb_logger = WandBLogger(project_name=project_name, enabled=True)

    config = wandb_logger.config
    k = int(config.get("K", 5))

    model = create_model(n_neighbors=k)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    wandb_logger.log_metrics({"val_accuracy": acc})

    num_classes = 3
    metrics = compute_metrics(y_test, preds, num_classes)
    wandb_logger.log_metrics(
        {
            "test_precision": metrics["avg_precision"],
            "test_recall": metrics["avg_recall"],
            "test_f1": metrics["avg_f1"],
        }
    )

    class_names = [1, 2, 3]
    confusion = compute_confusion_matrix(y_test, preds, class_names=class_names)
    plot_confusion_matrix(confusion, class_names, "confusion_matrix.png")

    wandb_logger.log_image("confusion_matrix", "confusion_matrix.png")
    wandb_logger.log_confusion_matrix(y_test, preds, class_names)

    wandb_logger.finish()
    return acc

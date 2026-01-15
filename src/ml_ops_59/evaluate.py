import numpy as np
# import matplotlib.pyplot as plt


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1

    return confusion


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """
    Compute various classification metrics

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        num_classes: Number of classes

    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1)
    """
    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)

    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))

        # Precision: TP / (TP + FP)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_precision": np.mean(precision),
        "avg_recall": np.mean(recall),
        "avg_f1": np.mean(f1),
    }

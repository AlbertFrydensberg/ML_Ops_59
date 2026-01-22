# import matplotlib.pyplot as plt
import numpy as np


def compute_confusion_matrix(y_true, y_pred, class_names=None, num_classes=None):
    """
    Compute confusion matrix that works with labels like {1,2,3} or {0,1,2}.
    If class_names is provided, it defines the label order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(class_names)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels) if num_classes is None else num_classes

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


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


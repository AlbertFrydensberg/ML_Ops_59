"""
Utility functions for evaluation and visualization
Including accuracy computation, confusion matrix, and plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_confusion_matrix(confusion_matrix: np.ndarray, 
                         class_names: Optional[list] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix to plot
        class_names: List of class names
        save_path: Path to save the plot (if None, display only)
    """
    num_classes = confusion_matrix.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if confusion_matrix[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


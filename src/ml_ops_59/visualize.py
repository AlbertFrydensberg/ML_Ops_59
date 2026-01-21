"""
Utility functions for evaluation and visualization
Including accuracy computation, confusion matrix, and plotting
"""

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path
import warnings
import logging



def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: Optional[list] = None, save_path: Optional[str] = None
) -> None:
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
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


"""
Implementation of SHAP
"""
# Suppress warnings and verbose SHAP logging (it has lots of annoy)
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('shap').setLevel(logging.WARNING)


# Ignore SHAP's numpy RNG warnings (it is not important, it is just a reminder for future package updates)
warnings.filterwarnings('ignore', message='.*NumPy global RNG.*')

def generate_shap_explanations(
    model,
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[list] = None,
    n_background: int = 50,
    n_explain: int = 10,
    output_dir: str = "reports/figures",
    save_plots: bool = True
) -> dict:
    """
    Generate SHAP explanations for a trained model.
    """

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        if feature_names is None:
            feature_names = X_train.columns.tolist()
        X_train = X_train.values
    
    if isinstance(X_test, pd.DataFrame):
        if feature_names is None:
            feature_names = X_test.columns.tolist()
        X_test = X_test.values
    
    # Sample background data (for computational efficiency)
    n_background = min(n_background, len(X_train))
    background_indices = np.random.choice(len(X_train), n_background, replace=False)
    X_background = X_train[background_indices]
    
    # Sample test data to explain
    n_explain = min(n_explain, len(X_test))
    X_explain = X_test[:n_explain]
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict_proba, X_background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_explain, silent=True)
    
    results = {
        'shap_values': shap_values,
        'explainer': explainer,
        'feature_names': feature_names,
        'X_explain': X_explain
    }
    
    # Save plots if requested
    if save_plots:
        plot_paths = _save_shap_plots(
            shap_values=shap_values,
            X_explain=X_explain,
            feature_names=feature_names,
            explainer=explainer,
            model=model,
            output_dir=output_dir
        )
        results['plot_paths'] = plot_paths
    # Only use results for plot, they are not printed to terminal or saved otherwise
    return results

def _save_shap_plots(
    shap_values,
    X_explain: np.ndarray,
    feature_names: list,
    explainer,
    model,         
    output_dir: str
) -> dict:
    """Internal function to save SHAP visualization plots."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    X_explain_df = pd.DataFrame(X_explain, columns=feature_names)

    # Plot Feature Importance
    """
        Shows which properties matter most for classifying wines overall.
        Longer bars mean the feature has bigger impact on predictions across all three wine types.
        Colors show whether high or low values of each feature push predictions toward specific wine classes.
        SHAP value ->  "If I remove this feature, how much does my prediction change?" 
        averaged across all possible feature combinations.
    """
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain_df, plot_type="bar", show=False)
    plt.xlabel("Mean absolute SHAP value", fontsize=12)
    plt.title("Feature Importance Ranking", fontsize=14, pad=20)
    importance_path = output_path / "shap_feature_importance.png"
    plt.savefig(importance_path, bbox_inches='tight', dpi=150)
    plt.close()
    plot_paths['importance'] = str(importance_path)
    
    return plot_paths
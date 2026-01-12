"""
Training a KNN on vine data with WandB + Sweep Support
"""


import numpy as np
from make_dataset import data_loader
from model import create_model
import wandb
from wandb_logger import WandBLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from helper_functions import (
    one_hot_encode, 
    normalize_data, 
    compute_confusion_matrix,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_history,
    create_model_summary
)



def main():
    print("=" * 70)
    print("KNN algoritm sweep - Training on wine data")
    print("=" * 70)

    np.random.seed(42)

    # 1. Load Data
    df = data_loader()

    X = df.drop(columns=["class"])
    y = df["class"]

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (Because it is for KNN algo and therefore important)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Initialize WandB BEFORE referencing config
    print("\n[3/7] Initializing experiment tracking...")
    wandb_logger = WandBLogger(
        project_name="MLops_59",
        enabled=True
    )

    config = wandb.config  # <--- sweep parameters come from here


    print("\n[4/7] Creating KNN...")
    # Create and train model
    model = create_model(n_neighbors=config.K) #sweep paramater K for number of neighbors
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.4f}")

    wandb_logger.log_model_architecture(accuracy=acc, n_neighbors=config.K)


    num_classes = 3
    metrics = compute_metrics(y_test, preds, num_classes)

    wandb_logger.log_metrics({
        'test_accuracy': acc,
        'test_precision': metrics['avg_precision'],
        'test_recall': metrics['avg_recall'],
        'test_f1': metrics['avg_f1']
    })

    confusion = compute_confusion_matrix(y_test, preds, num_classes)

    # 7. Visualizations
    print("\n[7/7] Generating visualizations...")
    plot_training_history(history, save_path='training_history.png')
    wandb_logger.log_image('training_history', 'training_history.png')

    class_names = [1, 2, 3]
    plot_confusion_matrix(confusion, class_names=class_names, save_path='confusion_matrix.png')
    wandb_logger.log_image('confusion_matrix', 'confusion_matrix.png')
    wandb_logger.log_confusion_matrix(y_test, preds, class_names=class_names)

    wandb_logger.finish()
    print("\nAll done!")


if __name__ == "__main__":
    main()

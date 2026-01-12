#this is the training pipeline

from .make_dataset import data_loader
from .model import create_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    # Load data
    df = data_loader()

    X = df.drop(columns=["class"])
    y = df["class"]

    # Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train model
    model = create_model(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

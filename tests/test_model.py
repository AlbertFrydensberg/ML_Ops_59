import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_ops_59.data import data_loader
from ml_ops_59.model import create_model
from ml_ops_59.train import train


def test_create_model_is_knn():
    model = create_model(n_neighbors=3)
    # KNN has this attribute
    assert hasattr(model, "n_neighbors")
    assert model.n_neighbors == 3
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_train_returns_valid_accuracy():
    "We expect an accuracy between 0 and 1."
    acc = train(n_neighbors=5, test_size=0.2, seed=42)

    assert isinstance(acc, float) or isinstance(acc, np.floating)
    assert 0.0 <= acc <= 1.0


def test_train_reproducible_with_same_seed():
    acc1 = train(n_neighbors=5, test_size=0.2, seed=42)
    acc2 = train(n_neighbors=5, test_size=0.2, seed=42)

    # accuracy should match exactly ....
    assert np.isclose(acc1, acc2), f"Acc not reproducible: {acc1} vs {acc2}"




def test_model_predict_output_shape_matches_targets():
    """
    Checks that predict returns one label per sample - could be mistakes when reshaping.
    """
    df = data_loader()

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = create_model(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    assert len(preds) == len(y_test)

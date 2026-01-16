# src/ml_ops_59/model.py

from sklearn.neighbors import KNeighborsClassifier


def create_model(n_neighbors: int = 5, weights: str = "uniform", p: int = 2):
    """
    Create and return a KNN classifier

    Args:
        n_neighbors: number of neighbors (K)
        weights: 'uniform' or 'distance'
        p: Minkowski distance parameter (1 = Manhattan, 2 = Euclidean)
    """
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric="minkowski",
    )

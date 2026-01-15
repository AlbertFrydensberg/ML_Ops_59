
#this is the model file

from sklearn.neighbors import KNeighborsClassifier

def create_model(n_neighbors=5):
    """
    Create and return a KNN classifier
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)

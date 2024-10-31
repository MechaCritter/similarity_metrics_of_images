import logging

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib

from src.config import setup_logging

setup_logging()

class GlobalKMeans:
    """
    This class is used to train a k-means model to cluster image features.

    **Note**: Currently only implement for SIFT features.

    **Next**: Implement deep learning features.
    """
    def __init__(self, num_clusters: int = 4):
        """
        Class constructor
        """
        self._logger = logging.getLogger('Global_KMeans')

        self.num_clusters = num_clusters
        self.kmeans = None

    def train(self, descriptors: np.ndarray) -> None:
        """
        Trains the k-means model on the given descriptor.

        :param descriptors: An array of flattened image features (e.g. SIFT)
        """
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(descriptors.astype(np.float32))

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster for the given descriptors.

        :param descriptors: An array of flattened image features (e.g. SIFT)
        :return: Index of the cluster for each descriptor
        """
        return self.kmeans.predict(descriptors.astype(np.float32))
    
    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroids of the clusters.

        :return: The centroids of the clusters
        """
        return self.kmeans.cluster_centers_
    
    def get_labels(self) -> np.ndarray:
        """
        Returns the labels of the clusters.

        :return: The labels of the clusters
        """
        return self.kmeans.labels_
    
    def save_model(self, model_path: str) -> None:
        """
        Saves the k-means model to a .pkl file.

        :param model_path: The path to save the model
        """
        with open(model_path, 'wb') as file:
            joblib.dump(self.kmeans, file)

    def load_model(self, model_path: str) -> None:
        """
        Loads the k-means model from a file.

        :param model_path: path of the model to load (a .pkl file)

        """
        with open(model_path, 'rb') as file:
            self.kmeans = joblib.load(file)

class GlobalGMM:
    def __init__(self, num_clusters: int = 4):
        """
        Class constructor
        """
        self._logger = logging.getLogger('Global_GMM')

        self.num_clusters: int = num_clusters
        self.gmm: GaussianMixture = None

    def train(self, descriptors: np.ndarray):
        """
        Train Gaussian Mixture Model (GMM) on a large dataset of descriptors.

        :param descriptors: SIFT descriptors collected from a large dataset of images.
        """
        self.gmm = GaussianMixture(n_components=self.num_clusters, random_state=42, covariance_type='diag')
        self.gmm.fit(descriptors)

    def save_model(self, file_path: str):
        """
        Save the trained GMM model to a file.

        :param file_path: Path to save the trained model.
        """
        with open(file_path, 'wb') as file:
            joblib.dump(self.gmm, file)

    def load_model(self, file_path: str):
        """
        Load a pre-trained GMM model from a file.

        :param file_path: Path from which to load the trained model.

        :raises ValueError: If the loaded object is not a GaussianMixture object.
        """
        with open(file_path, 'rb') as file:
            if not isinstance((obj:=joblib.load(file)), GaussianMixture):
                raise ValueError(f"Expected GaussianMixture object, got {type(obj)}")
            self.gmm = obj

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Predict the cluster labels for the given descriptors.

        :param descriptors: Descriptors for which to find the cluster labels.
        :return: Array of cluster labels for each descriptor.
        """
        return self.gmm.predict(descriptors)

    def predict_proba(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each descriptor belonging to each cluster.

        :param descriptors: Descriptors for which to find cluster probabilities.
        :return: Probability of each descriptor belonging to each cluster.
        """
        return self.gmm.predict_proba(descriptors)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroids of the clusters.

        :return: The centroids of the clusters
        """
        return self.gmm.means_

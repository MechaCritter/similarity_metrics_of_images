import numpy as np
import joblib

from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:
    """
    This class is used to train a PCA model to reduce the dimensionality of image features, such as
    SIFT or root-SIFT.
    """
    def __init__(self, num_components: int = 64):
        """
        Class constructor
        """
        self.num_components = num_components
        self.pca = None

    def train(self, descriptors: np.ndarray) -> None:
        """
        Trains the PCA model on the given descriptor.
        :param descriptors: array of image descriptors
        """
        self.pca = PCA(n_components=self.num_components)
        self.pca.fit(descriptors)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Reduces the dimensionality of the descriptors.

        :param descriptors: array of image descriptors

        :return: reduced descriptors
        """
        return self.pca.transform(descriptors)

    def save_model(self, model_path: str) -> None:
        """
        Saves the PCA model to a .pkl file.

        :param model_path: path to save the model
        """
        joblib.dump(self.pca, model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads the PCA model from a file.

        :param model_path: path of the model to load (a .pkl file)
        """
        self.pca = joblib.load(model_path)


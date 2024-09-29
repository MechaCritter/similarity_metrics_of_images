from dataclasses import dataclass, field
import logging

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.feature import fisher_vector, learn_gmm
import numpy as np
import cv2
from typing import List, Tuple

from utils import get_centroids, ImageFeatureExtractor
from logger_config import setup_logging

setup_logging()

_logger_vv = logging.getLogger("VLAD_Vector")
_logger_fv = logging.getLogger("Fisher_Vector")

@dataclass
class BaseMetrics:
    """
    Base class for metrics, including SIFT and SURF features. Used for VLAD and Fischer Vector calculations.

    - **Note**: The keypoint centroids calculated in this class are the 2D coordinates of the keypoints (The keypoints themselves contain more information like angle, size, response, octave, and class_id).

    - **Note**: All the attributes are read-only. They are calculated internally and should not be modified.

    :param image: The image for which to calculate the metrics.
    :type image: np.ndarray
    :param num_clusters: The number of clusters to use for the KMeans clustering algorithm.
    :type num_clusters: int
    :param norm_order: The order of the norm to use for normalization. Default is 2 (l2 norm will be applied in this case).
    :type norm_order: int
    :param power_norm_weight: The weight to apply to the power normalization. Default is 0.5.
    :type power_norm_weight: float
    :param epsilon: A small value to add to the denominator to avoid division by zero.
    :type epsilon: float
    :param flatten: Whether to flatten the resulting vector (the vector becomes 1D). Default is False.
    :type flatten: bool
    :param verbose: Whether to print the keypoints data, descriptors, and other information. Default is False.
    :type verbose: bool

    **Attributes**:
    
    :ivar keypoints: List of cv2.KeyPoint objects. 
    :vartype keypoints: List[cv2.KeyPoint]
    :ivar keypoints_2d_coords: 2D coordinates of the keypoints on the image.
    :vartype keypoints_2d_coords: np.ndarray
    :ivar descriptors: Descriptors of the keypoints (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptors: np.ndarray
    :ivar descriptor_centroids: centroids of the descriptors (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptor_centroids: np.ndarray
    :ivar descriptor_labels: Labels of the descriptors (which cluster/centroid they belong to).
    :vartype descriptor_labels: np.ndarray
    :ivar keypoints_2d_coords_centroids: 2D coordinates of the keypoint centroids. At each centroid, a descriptor is placed.
    :vartype keypoints_2d_coords_centroids: np.ndarray
    :ivar keypoints_labels: Labels of the keypoints (which cluster/centroid they belong to).
    :vartype keypoints_labels: np.ndarray
    """
    image: np.ndarray = field(repr=False)
    num_clusters: int = 16
    norm_order: int = 2
    power_norm_weight: float = 0.5
    epsilon: float = 1e-9
    flatten: bool = False
    verbose: bool = False

    keypoints: List[cv2.KeyPoint] = field(init=False, repr=False) 
    keypoints_2d_coords: np.ndarray = field(init=False) 
    descriptors: np.ndarray = field(init=False)
    descriptor_centroids: np.ndarray = field(init=False)
    descriptor_labels: np.ndarray = field(init=False)
    keypoints_2d_coords_centroids: np.ndarray = field(init=False)
    keypoints_labels: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray):
            raise ValueError(f"Image must be a numpy array, not {type(self.image)}")
        if self.power_norm_weight < 0 or self.power_norm_weight > 1:
            raise ValueError("Power norm weight must be between 0 and 1.")
        self.get_sift()
        self.centroids_and_labels()
        if self.verbose:
            print(
                "====================================\n"
                "Vector Type: ", self.__class__.__name__, "\n"
                "Keypoints data:\n"
                "Number of keypoints: \n", len(self.keypoints), "\n"
                "Keypoint angles: \n", [kp.angle for kp in self.keypoints], "\n"
                "Keypoint sizes: \n", [kp.size for kp in self.keypoints], "\n"
                "Keypoint responses: \n", [kp.response for kp in self.keypoints], "\n"
                "Keypoint octaves: \n", [kp.octave for kp in self.keypoints], "\n"
                "Keypoint class IDs: \n", [kp.class_id for kp in self.keypoints], "\n"
                "====================================\n"
                "Keypoint centroids: \n", self.keypoints_2d_coords_centroids, "\n"
                "====================================\n"
                "Keypoint labels: \n", self.keypoints_labels, "\n"
                "====================================\n"
                "Keypoint 2D coordinates: \n", self.keypoints_2d_coords, "\n"
                "====================================\n"
                "Descriptor centroids: \n", self.descriptor_centroids, "\n"
                "====================================\n"
                "Descriptor vectors: \n", self.descriptors, "\n"
                "Length of one descriptor vector: \n", len(self.descriptors[0]), "\n"
                "====================================\n"
            )

    def get_sift(self):
        """
        Get the SIFT features for the image. This includes the keypoints (with all their attributes like angle, label, ...) and descriptors (128-dimensional vectors).
        """
        self.keypoints, self.descriptors = ImageFeatureExtractor.sift(self.image)

    def centroids_and_labels(self):
        self.descriptor_centroids, self.descriptor_labels = get_centroids(self.descriptors, self.num_clusters)
        self.get_keypoint_coords()
        self.keypoints_2d_coords_centroids, self.keypoints_labels = get_centroids(self.keypoints_2d_coords, self.num_clusters)

    def get_keypoint_coords(self):
        """
        Get the 2D coordinates of the keypoints.
        """
        self.keypoints_2d_coords = np.array([kp.pt for kp in self.keypoints])

@dataclass
class VLAD(BaseMetrics):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) for the given image.

    To retrieve to VLAD vector, simply call the `vector` attribute of the object.

    **Attributes**:

    :ivar vector: The VLAD vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    """
    vector: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.compute_vlad_vector()

    def compute_vlad_vector(self) -> None:
        """
        Compute the VLAD descriptor for the image. Each VLAD vector has a fixed size of
        128, as calculated using OpenCV's SIFT_create() function. After calculaing the VLAD
        vector, normalization is applied to the vector.
        """
        vlad = np.zeros((self.num_clusters, 128))

        for i in range(len(self.descriptors)):
            label = self.descriptor_labels[i]
            vlad[label] += self.descriptors[i] - self.descriptor_centroids[label]
        _logger_vv.debug("VLAD vector before normalization: %s", vlad)

        # Power normalization
        vlad = np.sign(vlad) * (np.abs(vlad) ** self.power_norm_weight)
        _logger_vv.debug("VLAD vector after power normalization: %s", vlad)

        # L2 normalization (if norm_order = 2)
        norm = np.linalg.norm(vlad, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
        _logger_vv.debug("Norm vector of VLAD vector: %s", norm)
        for idx, is_zero in enumerate(np.all(norm == 0, axis=1, keepdims=True)):
            if is_zero:
                _logger_vv.warning(f"VLAD Vector at index {idx} at coordinates {self.keypoints_2d_coords_centroids[idx]} is all zero.") 
        vlad = vlad / norm
        _logger_vv.debug("VLAD vector after L2 normalization: %s", vlad)

        if self.flatten:
            vlad = vlad.flatten()
            _logger_vv.debug("Flattened VLAD vector: %s", vlad)

        self.vector = vlad
        _logger_vv.info("Resulting VLAD vector: %s. Shape of vector: %s", self.vector, self.vector.shape)

@dataclass
class FischerVector(BaseMetrics):
    """
    Calculate the Fischer Vector for the given image. For D-dimensional input descriptors or vectors, and a K-mode GMM, 
    the Fisher vector dimensionality will be 2KD + K. Thus, its dimensionality is invariant to the number of descriptors/vectors.
    
    To retrieve the Fischer Vector, simply call the `vector` attribute of the object.
    
    **Attributes**:

    :ivar vector: The Fischer Vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    :ivar gmm: The Gaussian Mixture Model used to calculate the Fischer Vector.
    :vartype gmm: GaussianMixture
    """
    vector: np.ndarray = field(init=False)
    gmm: GaussianMixture = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.compute_fisher_vector()

    def compute_fisher_vector(self) -> None:
        """
        Compute the Fischer Vector for the image. The Fisher Vector is calculated using the
        Fisher Vector algorithm from the scikit-image library.
        """
        self.gmm = GaussianMixture(n_components=self.num_clusters, random_state=0, covariance_type='diag')
        self.gmm.fit(self.descriptors)
        
        # Extract the Fisher Vector
        self.vector = np.array([fisher_vector(self.descriptors, self.gmm, alpha=self.power_norm_weight)])

        _logger_fv.debug("Fisher vector before normalization: %s", self.vector)

        # Power normalization
        self.vector = np.sign(self.vector) * (np.abs(self.vector) ** self.power_norm_weight)
        _logger_fv.debug("Fisher vector after power normalization: %s", self.vector)

        # L2 normalization
        norm = np.linalg.norm(self.vector, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
        _logger_fv.debug("Norm vector of Fisher vector: %s", norm)
        for idx, is_zero in enumerate(np.all(norm == 0, axis=1, keepdims=True)):
            if is_zero:
                _logger_vv.warning(f"Fisher Vector at index {idx} at coordinates {self.keypoints_2d_coords_centroids[idx]} is all zero.") 
        self.vector = self.vector / norm

        _logger_fv.debug("Fisher vector after L2 normalization: %s", self.vector)

        if self.flatten:
            self.vector = self.vector.flatten()
            _logger_fv.debug("Flattened Fischer vector: %s", self.vector)

        _logger_fv.info("Resulting Fischer Vector: %s. Shape of vector: %s", self.vector, self.vector.shape)
    

if __name__ == "__main__":
    from src.datasets import FlowerDataSet
    path_to_test_data = 'data/raw/test'
    flower_data = FlowerDataSet('data/raw/train', plot=True)
    img = flower_data[20]
    #vlad = VLAD(img[0], verbose=True)
    fischer = FischerVector(img[0], verbose=True)

    

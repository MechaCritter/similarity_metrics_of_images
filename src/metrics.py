from dataclasses import dataclass, field
import logging

import torch
from skimage.feature import fisher_vector
import numpy as np
import cv2
from typing import List
from piq import ssim, multi_scale_ssim as ms_ssim

from models.Clustering import GlobalKMeans, GlobalGMM
from src.utils import sift, root_sift
from src.config import setup_logging

setup_logging()

_logger_vv = logging.getLogger("VLAD_Vector")
_logger_fv = logging.getLogger("Fisher_Vector")


@dataclass
class ClusteringBasedMetric:
    """
    Base class for clustering-based metrics (metrics that use k-means or GMM). Used for VLAD and Fischer Vector calculations.

    - **Note**: All the attributes are read-only. They are calculated internally and should not be modified.

    The class has following modifiable attributes:
    ----------------------------------------------
        - np.ndarray image: The image for which to calculate the metrics.
        - int norm_order: The order of the norm to use for normalization. Default is 2 (l2 norm will be applied in this case).
        - float power_norm_weight: The weight to apply to the power normalization. Default is 0.5.
        - float epsilon: A small value to add to the denominator to avoid division by zero.
        - bool flatten: Whether to flatten the resulting vector (the vector becomes 1D). Default is False.
        - bool verbose: Whether to print the keypoints data, descriptors, and other information. Default is False.
        - str feature: The feature to use for the image. Default is "sift". Accepted values: "sift", "root_sift".

    Please do not modify attributes stated below:
    --------------------------------------------

    :ivar keypoints: List of cv2.KeyPoint objects.
    :vartype keypoints: List[cv2.KeyPoint]
    :ivar descriptors: Descriptors of the keypoints (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptors: np.ndarray
    :ivar descriptor_centroids: centroids of the descriptors (VLAD or Fischer Vector, dim = (num_clusters, 128)).
    :vartype descriptor_centroids: np.ndarray
    :ivar descriptor_labels: Labels of the descriptors (which cluster/centroid they belong to).
    :vartype descriptor_labels: np.ndarray
    """
    image: np.ndarray = field(repr=False)
    norm_order: int = 2
    power_norm_weight: float = 0.5
    epsilon: float = 1e-9
    flatten: bool = True
    verbose: bool = False
    feature: str = "sift"

    keypoints: List[cv2.KeyPoint] = field(init=False, repr=False)
    descriptors: np.ndarray = field(init=False)
    descriptor_centroids: np.ndarray = field(init=False)
    descriptor_labels: np.ndarray = field(init=False)
    keypoints_2d_coords: np.ndarray = field(init=False)
    keypoints_labels: np.ndarray = field(init=False)

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray):
            raise ValueError(f"Image must be a numpy array, not {type(self.image)}")
        if self.power_norm_weight < 0 or self.power_norm_weight > 1:
            raise ValueError("Power norm weight must be between 0 and 1.")
        self.get_sift()
        self.get_keypoint_coords()

    def get_sift(self):
        """
        Get the SIFT features for the image. This includes the keypoints (with all their attributes like angle, label, ...) and descriptors (128-dimensional vectors).
        """
        if self.feature == "sift":
            self.keypoints, self.descriptors = sift(self.image)
        elif self.feature == "root_sift":
            self.keypoints, self.descriptors = root_sift(self.image)
        else:
            raise ValueError("Feature must be 'sift' or 'root_sift', not {self.feature}")

    def get_keypoint_coords(self):
        """
        Get the 2D coordinates of the keypoints.
        """
        self.keypoints_2d_coords = np.array([kp.pt for kp in self.keypoints])

@dataclass
class VLAD(ClusteringBasedMetric):
    """
    Calculate the Vector of Locally Aggregated Descriptors (VLAD) for the given image.

    To retrieve to VLAD vector, simply call the `vector` attribute of the object.

    :param k_means: The pre-trained k-means model to use for clustering the descriptors.
    :type k_means: GlobalKMeans

    :ivar _vector: The VLAD vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    """
    k_means: GlobalKMeans = None
    _vector: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.centroids_and_labels()
        self.compute_vlad_vector()

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
                "Descriptor centroids: \n", self.descriptor_centroids, "\n"
                "====================================\n"
                "Descriptor vectors: \n", self.descriptors, "\n"
                "Number of descriptor vectors: \n", len(self.descriptors), "\n"
                "Length of one descriptor vector: \n", len(self.descriptors[0]), "\n"
                "====================================\n"
            )

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    def centroids_and_labels(self):
        self.descriptor_labels = self.k_means.predict(self.descriptors)
        self.descriptor_centroids = self.k_means.get_centroids()

    def compute_vlad_vector(self) -> None:
        """
        Compute the VLAD descriptor for the image. Each VLAD vector has a fixed size of
        128, as calculated using OpenCV's SIFT_create() function. After calculaing the VLAD
        vector, normalization is applied to the vector.
        """
        vlad = np.zeros((len(self.k_means.get_centroids()), 128))

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

        vlad = vlad / norm
        _logger_vv.debug("VLAD vector after L2 normalization: %s", vlad)

        if self.flatten:
            vlad = vlad.flatten()
            _logger_vv.debug("Flattened VLAD vector: %s", vlad)

        self._vector = vlad
        _logger_vv.info("Resulting VLAD vector: %s. Shape of vector: %s", self._vector, self._vector.shape)

@dataclass
class FisherVector(ClusteringBasedMetric):
    """
    Calculate the Fischer Vector for the given image. For D-dimensional input descriptors or vectors, and a K-mode GMM, 
    the Fisher vector dimensionality will be 2KD + K. Thus, its dimensionality is invariant to the number of descriptors/vectors.
    
    To retrieve the Fischer Vector, simply call the `vector` attribute of the object.

    :param num_gaussians: The number of Gaussian components to use in the Gaussian Mixture Model (GMM). Default is 128.
    :type num_gaussians: int
    
    **Attributes**:

    :ivar _vector: The Fischer Vector for the image. Dimension would be (num_clusters, 128) if flatten is False, else (num_clusters * 128,).
    :vartype vector: np.ndarray
    :ivar gmm: The Gaussian Mixture Model used to calculate the Fischer Vector.
    :vartype gmm: GaussianMixture
    """
    gmm: GlobalGMM = None
    _vector: np.ndarray = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.centroids_and_labels()
        self.compute_fisher_vector()

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
                "Descriptor centroids: \n", self.descriptor_centroids, "\n"
                "====================================\n"
                "Descriptor vectors: \n", self.descriptors, "\n"
                "Length of one descriptor vector: \n", len(self.descriptors[0]), "\n"
                "====================================\n"
            )

    @property
    def vector(self) -> np.ndarray:
        return self._vector

    def centroids_and_labels(self):
        self.descriptor_labels = self.gmm.predict(self.descriptors)
        self.descriptor_centroids = self.gmm.get_centroids()

    def compute_fisher_vector(self) -> None:
        """
        Compute the Fischer Vector for the image. The Fisher Vector is calculated using the
        Fisher Vector algorithm from the scikit-image library.
        """
        # Extract the Fisher Vector
        self._vector = np.array([fisher_vector(self.descriptors, self.gmm.gmm, alpha=self.power_norm_weight)])

        _logger_fv.debug("Fisher vector before normalization: %s", self._vector)

        # Power normalization
        self._vector = np.sign(self._vector) * (np.abs(self._vector) ** self.power_norm_weight)
        _logger_fv.debug("Fisher vector after power normalization: %s", self._vector)

        # L2 normalization
        norm = np.linalg.norm(self._vector, axis=1, ord=self.norm_order, keepdims=True) + self.epsilon
        _logger_fv.debug("Norm vector of Fisher vector: %s", norm)
        self._vector = self._vector / norm

        _logger_fv.debug("Fisher vector after L2 normalization: %s", self._vector)

        if self.flatten:
            self._vector = self._vector.flatten()
            _logger_fv.debug("Flattened Fischer vector: %s", self._vector)

        _logger_fv.info("Resulting Fischer Vector: %s. Shape of vector: %s", self._vector, self._vector.shape)

@dataclass
class StructuralSimilarity:
    """
    Base class for SSIM and MS-SSIM metrics. Used for comparing two images.
    """
    image_1: np.ndarray = field(repr=False)
    image_2: np.ndarray = field(repr=False)
    data_range: float = 255

@dataclass
class SSIM(StructuralSimilarity):
    """
    Calculate the Structural Similarity Index (SSIM) between two images. Pass normal RGB images as input.
    To access the SSIM value, call the `value` attribute of the object.
    """
    _ssim: torch.Tensor= field(init=False)
    def __post_init__(self):
        image_1_tensor = torch.from_numpy(self.image_1).permute(2, 0, 1).unsqueeze(0).float()
        image_2_tensor = torch.from_numpy(self.image_2).permute(2, 0, 1).unsqueeze(0).float()
        self._ssim = ssim(image_1_tensor, image_2_tensor, data_range=self.data_range)
        print("SSIM:", self._ssim)

    @property
    def value(self):
        return self._ssim

@dataclass
class MS_SSIM(StructuralSimilarity):
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two images. Pass normal RGB images as input.
    To access the MS-SSIM value, call the `value` attribute of the object.
    """
    _ms_ssim: torch.Tensor= field(init=False)
    def __post_init__(self):
        image_1_tensor = torch.from_numpy(self.image_1).permute(2, 0, 1).unsqueeze(0).float()
        image_2_tensor = torch.from_numpy(self.image_2).permute(2, 0, 1).unsqueeze(0).float()
        self._ms_ssim = ms_ssim(image_1_tensor, image_2_tensor, data_range=self.data_range)
        print("MS-SSIM:", self._ms_ssim)

    @property
    def value(self):
        return self._ms_ssim


if __name__ == "__main__":
    from src.datasets import CustomDataSet
    flower_data = CustomDataSet(plot=True)
    image = flower_data[0]
    k_means = GlobalKMeans()
    gmm = GlobalGMM()
    k_means.load_model("models/pickle_model_files/k_means_model_flower_car_500imgs.pkl")
    gmm.load_model("models/pickle_model_files/gmm_model_flower_car_500imgs.pkl")
    for img in image:
        vlad = VLAD(image=img[0], k_means=k_means, flatten=True)
        print("Vlad vector:", vlad.vector)
        print("Length of VLAD vector:", len(vlad.vector))
        vlad_rootsift = VLAD(image=img[0], k_means=k_means, feature="root_sift", flatten=True)
        fisher_sift = FisherVector(image=img[0], gmm=gmm, flatten=True)
        print("Vlad RootSIFT vector:", vlad_rootsift.vector)
        print("Length of VLAD RootSIFT vector:", len(vlad_rootsift.vector))
        print("Similarity between VLAD and VLAD RootSIFT:", np.dot(vlad.vector, vlad_rootsift.vector.T))

    

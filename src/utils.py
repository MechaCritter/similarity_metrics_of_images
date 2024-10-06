import os
import logging
from typing import List, Tuple

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Append path to root. Delete at the end of the project.
import os
import sys
codespace_path = os.path.abspath('..')
sys.path.insert(0, codespace_path)
##############################################

from src.logger_config import setup_logging

setup_logging()

_logger_ife = logging.getLogger("Image_Feature_Extractor")
_logger_ip = logging.getLogger("Image_Processor")

# Decorators
def _check_is_numpy_image(func: callable):
    def wrapper(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image should be a numpy array")
        return func(image, *args, **kwargs)
    return wrapper

def get_centroids(data: np.ndarray, num_clusters:int):
    """
    Get the centroids of the clusters using KMeans. `data`should best be a numpy array.

    :param data: Data to cluster
    :type data: np.ndarray
    :num_clusters: Number of clusters
    :type num_clusters: int

    :return: Centroids of the clusters, the corresponding labels and the locations of the centroids
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)

    return kmeans.cluster_centers_, kmeans.labels_

def create_and_plot_synthetic_data(lower_limit: float, upper_limit: float, num_samples: int, plot_type: str = 'scatter'):
    """
    Generates synthetic data and plots it.

    :param lower_limit: Lower limit for the data
    :type lower_limit: float
    :param upper_limit: Upper limit for the data
    :type upper_limit: float
    :param num_samples: Number of samples to generate
    :type num_samples: int
    :param plot_type: Type of plot ('scatter' or 'linear')
    :type plot_type: str

    :return: x and y values
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    x = np.linspace(lower_limit, upper_limit, num_samples)
    y = np.random.uniform(lower_limit, upper_limit, num_samples)

    plt.figure(figsize=(10, 6))
    if plot_type == 'scatter':
        plt.scatter(x, y)
    elif plot_type == 'linear':
        plt.plot(x, y)
    else:
        raise ValueError("plot_type must be either 'scatter' or 'linear'")

    plt.title(f"Synthetic Data Plot ({plot_type})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    return x, y

@_check_is_numpy_image
def get_non_zero_pixel_indices(image: np.ndarray) -> tuple:
    """
    Get the indices of pixels that have at least one non-zero channel.
    Use this method to find coordinates of non-black pixels in an image.

    :param image: Input image
    :type image: np.ndarray
    :return: Indices of non-zero pixels in a tuple
    :rtype: tuple
    """
    return tuple(np.argwhere(np.any(image != 0, axis=-1)))

@_check_is_numpy_image
def plot_clusters_on_image(image: np.ndarray, 
                           data: np.ndarray, 
                           centroids: np.ndarray, 
                           labels: np.ndarray, 
                           keypoints: list[cv2.KeyPoint],
                           show_centroid_coords: bool = False,
                           show_centroid_labels: bool = False) -> None:
    """
    Plot two images with keypoints next to each other:
    - The left image shows which keypoint belongs to which cluster by using a color code.
    - The second plot display more rich information about the keypoints (e.g. size, orientation, etc.).

    :param image: Image to display as the background
    :type image: bp.ndarray
    :param data: 2D numpy array of data points (n_samples, n_features)
    :type data: np.ndarray
    :param centroids: 2D numpy array of centroids (n_clusters, n_features)
    :type centroids: np.ndarray
    :param labels: 1D numpy array of cluster labels for each data point
    :type labels: np.ndarray
    :param keypoints: List of cv2.KeyPoint objects
    :type keypoints: list[cv2.KeyPoint]
    :param show_centroid_coords: If True, display the coordinates of the centroids
    :type show_centroid_coords: bool
    :param show_centroid_labels: If True, display the labels of the centroids
    :type show_centroid_labels: bool
    """
    
    # Convert image to rgb format
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define unique cluster labels
    unique_labels = np.unique(labels)

    # ### Delete this later ###
    # img = cv2.drawKeypoints(img, keypoints, None)
    # #########################
    
    # Generate a color map with as many colors as unique clusters
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    # Create a figure with extra space on the right for the legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
    # Display the image as the background
    left, right, bottom, top = 0, img.shape[1], image.shape[0], 0 # Image is upside down on the y-axis when calling plt.imshow, hence bottom, top are flipped
    ax1.imshow(img, extent=[left, right, bottom, top])
    
    # Plot each cluster with a different color
    for cluster_idx in unique_labels:
        # Select data points belonging to the current cluster
        cluster_data = data[labels == cluster_idx]
        
        ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    s=20, c=[colors(cluster_idx)], 
                    label=f"Cluster {cluster_idx}", alpha=0.7)
    
    # Plot the centroids in a different style
    ax1.scatter(centroids[:, 0], centroids[:, 1], 
                s=120, c='black', marker='x', 
                label='Centroids')
    
    ax1.set_title("Clusters and Centroids on Image")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    
    # Keep axes in the range of the image size
    ax1.set_xlim(0, img.shape[1])
    ax1.set_ylim(img.shape[0], 0) # Flip the y-axis to match the image orientation

    # Place legend outside the plot
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # If show_centroid_coords is True, display the coordinates of the centroids
    if show_centroid_coords:
        for i, centroid in enumerate(centroids):
            ax1.text(centroid[0], centroid[1], f"Centroid {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})", fontsize=6, color='black')

    # If show_centroid_labels is True, display the labels of the centroids
    if show_centroid_labels:
        for i, centroid in enumerate(centroids):
            ax1.text(centroid[0], centroid[1], f"Cluster {i}", fontsize=6, color='black')

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax2.imshow(img_with_keypoints)
    ax2.set_title("Keypoints on Image")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.set_xlim(0, img.shape[1])
    ax2.set_ylim(img.shape[0], 0) # Flip the y-axis to match the image orientation

    plt.subplots_adjust(wspace=0.5)
    plt.grid(False)
    plt.show()

def plot_similarity_heatmap_between_2_vectors(vector1: np.ndarray, vector2: np.ndarray, **kwargs) -> None:
    """
    Plot a heatmap showing the similarity between two vectors. Use this method to compare similarity
    between VLAD/Fisher vectors.

    ***Note***: Make sure both vectors have shape `num_clusters x num_features`.

    :param vector1: First (VLAD/Fisher) vector
    :type vector1: np.ndarray
    :param vector2: Second (VLAD/Fisher) vector
    :type vector2: np.ndarray
    **kwargs: Additional keyword arguments (currently available: `title`, `xlabel`, `ylabel`)
    """
    if not vector1.shape[1] == vector2.shape[1]:
        raise ValueError(f"Both vectors must have the same number of features, but got {vector1.shape[1]} and {vector2.shape[1]} instead.")
    if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
        raise ValueError(f"Expected both vectors to be numpy arrays, but got {type(vector1)} and {type(vector2)} instead.")
    # Calculate cosine similarity between the two vectors
    similarity = cosine_similarity(vector1, vector2)
    plt.figure(figsize=(10, 6))
    sns.heatmap(similarity, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=[f"Cluster {i}" for i in range(vector1.shape[0])],
                yticklabels=[f"Cluster {i}" for i in range(vector2.shape[0])],
                             cbar_kws={"label": "Cosine Similarity"})
    plt.title("Cosine Similarity Heatmap between Two Vectors") if "title" not in kwargs else plt.title(kwargs["title"])
    plt.xlabel("Clusters of Image 1") if "xlabel" not in kwargs else plt.xlabel(kwargs["xlabel"])
    plt.ylabel("Clusters of Image 2") if "ylabel" not in kwargs else plt.ylabel(kwargs["ylabel"])
    plt.show() 

def is_subset(list1, list2):
    """
    Check if list1 is a subset of list2.

    :param list1: First list to check (potential subset)
    :type list1: list
    :param list2: Second list (or tuple) to check against (potential superset)
    :type list2: list or tuple

    :returns: True if list1 is a subset of list2, False otherwise
    :rtype: bool
    """
    if len(list1)> len(list2):
        raise ValueError("List1 must be have smaller or equal length than list2")
    return set(list1).issubset(list2)

def convert_to_integers(list_of_tuples: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """
    Convert all elements in a list of tuples to integers.

    :param list_of_tuples: List of tuples with float values
    :type list_of_tuples: List[Tuple[float, float]]
    :return: List of tuples with integer values
    :rtype: List[Tuple[int, int]]
    """
    return [(int(x), int(y)) for x, y in list_of_tuples]

# Classes
class ImageProcessor:
    """
    ***Note***: This class creates new images.
    Generic image processing class that can do following things:
    - Gaussian blur
    - Thresholding
    - Resizing

    Please alwasys pass the image as the first argument to the function, since
    the decorator assumes that the first argument is the image.

    ***Note***: Only use static methods in this class.
    """
    @staticmethod
    @_check_is_numpy_image
    def gaussian_blurr(image, kernel_size=3, sigma=1.0):
        _logger_ip.debug(f"This Gaussian kernel was used: \n"
                      f"{cv2.getGaussianKernel(kernel_size, sigma) @ cv2.getGaussianKernel(kernel_size, sigma).T}")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    @_check_is_numpy_image
    def thresholding(image, threshold_value=None, max_value=255, threshold_types: tuple = (cv2.THRESH_BINARY,)):
        """
        Currently only works for gray images.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresholded_image = cv2.threshold(gray_image, threshold_value, max_value, np.sum(threshold_types))
        _logger_ip.debug(f"Threshold value used: {threshold_value}")
        return thresholded_image

    @staticmethod
    @_check_is_numpy_image
    def resize(image, dimensions, interpolation=cv2.INTER_LINEAR):
        """
        Resizes the given image to the given dimensions. If a single integer is passed,
        both the width and height will be resized to that integer. If a tuple is passed,
        then it works like the normal cv2.resize function.
        """
        if isinstance(dimensions, int):
            dimensions = (dimensions, dimensions)
        return cv2.resize(image, dimensions, interpolation=interpolation)
    
    @staticmethod
    @_check_is_numpy_image
    def sharpen(image, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])):
        """
        Sharpens the given image using the given kernel.
        """
        return cv2.filter2D(image, -1, kernel)

class ImageFeatureExtractor:
    """
    ***Note***: This class simply extracts features from the image without creating new ones

    Like the ImageProcessor class, please always pass the image as the first argument to the function.

    ***Note***: Only use static methods in this class.
    """
    @staticmethod
    @_check_is_numpy_image
    def sift(image) -> Tuple[cv2.KeyPoint, np.ndarray]:
        """
        Extracts SIFT features from the given image.

        :param image: Input image
        :type image: np.ndarray

        :return: keypoints as a list of cv2.KeyPoint objects and descriptors as a numpy array
        :rtype: tuple
        """
        sift = cv2.SIFT.create()
        _logger_ife.debug(f"SIFT object created: {sift}")
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors

    @staticmethod
    @_check_is_numpy_image
    def surf(image):
        """
        Extracts SURF features from the given image.
        """
        surf = cv2.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(image, None)
        return keypoints, descriptors

    @staticmethod
    @_check_is_numpy_image
    def difference_of_gaussian(image: np.ndarray,
                               num_intervals: int,
                               num_octaves: int=1,
                               sigma: float=1.6,
                               plot: bool=False) -> list:
        """
        Calculates DoG for the given image.

        :param image: Input image
        :type image: np.ndarray
        :param num_intervals: Number of intervals (normally written as `s`) in each octave
        :type num_intervals: int
        :param num_octaves: Number of octaves
        :type num_octaves: int
        :return: List of octave images (the difference of gaussian images within each octave)
        :rtype: list
        """
        k = 2 ** (1.0 / num_intervals) # Scale factor
        octave_images = []
        octave_range = num_intervals + 3

        # Generate blurred images
        for octave in range(num_octaves):
            gaussian_images = []
            current_sigma = sigma
            _logger_ife.info(f"""
            Calculating DoG for octave {octave} with {num_intervals} intervals and sigma={sigma}:
            
            =====================================================================================\n
            """)

            for _ in range(octave_range):
                gaussian_images.append(ImageProcessor.gaussian_blurr(image, sigma=current_sigma))
                _logger_ife.debug(f"Sigma value used: {current_sigma}")
                current_sigma *= k

            # Calculate DoG and append to the octave images
            for i in range(1, len(gaussian_images)):
                dog = gaussian_images[i] - gaussian_images[i - 1]
                octave_images.append(dog)

            # Downsample the image by factor of 2 for the next octave
            _logger_ife.debug(f"Current image shape: {image.shape}")
            image = ImageProcessor.resize(image, (image.shape[1]//2, image.shape[0]//2))

        _logger_ife.debug("Total number of octave images: %s", len(octave_images))
        if plot:
            plt.figure(figsize=(25, 10))
            for i in range(num_octaves):
                for j in range(num_intervals + 2):
                    plt.subplot(num_octaves, num_intervals + 2, i * (num_intervals + 2) + j + 1)
                    plt.title(f"Octave: {i}, Interval: {j}")
                    plt.imshow(cv2.cvtColor(octave_images[i * (num_intervals + 2) + j], cv2.COLOR_BGR2RGB))

            plt.suptitle(f"""
                "Difference of Gaussian calculation with initial_sigma=1.6\n"
                "Number of intervals: {num_intervals}, Number of octaves: {num_octaves}",
                fontsize=20
                """)
            plt.show()
        return octave_images

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datasets import FlowerDataSet
    flower_data = FlowerDataSet('data/raw/train')
    image = flower_data[20][0]
    octave_images = ImageFeatureExtractor.difference_of_gaussian(image,
                                                                   num_intervals=5,
                                                                   num_octaves=2,
                                                                   plot=True)




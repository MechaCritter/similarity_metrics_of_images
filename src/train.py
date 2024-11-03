from typing import Union
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.utils import load_model
from src.config import IMAGE_SIZE
from src.metrics import VLAD, FisherVector
from utils import sift, resize, root_sift, standardize_data, save_model
from src.datasets import *

# --k_means-- #
def train_and_save_k_means_model(data_set: BaseDataset,
                                 num_clusters: int,
                                 feature: str='sift',
                                 test = False,
                                 model_path: str='models/pickle_model_files/k_means_model.pkl') -> None:
    """
    Trains the k-means model on the image features and saves it as a .pkl file.

    **Note**: Currently only implement for SIFT features.

    :param data_set: CustomDataSet object
    :param num_clusters: number of clusters
    :param feature: 'sift' or 'root_sift'
    :param test: if True, only 10 images are used for training
    :param model_path: path to save the model

    :raises ValueError: If the length of the descriptor vector is not 128
    :raises ValueError: If the feature is not 'sift' or 'root_sift'
    """
    sift_vectors_list = np.empty((0, 128))
    if feature == 'sift':
        feature_extractor = sift
    elif feature == 'root_sift':
        feature_extractor = root_sift
    else:
        raise ValueError(f"Feature has to be 'sift' or 'root_sift'. {feature} is not supported.")
    for i, (img, *rest) in enumerate(data_set):
        _, descriptors = feature_extractor(resize(img, IMAGE_SIZE))
        sift_vectors_list = np.append(sift_vectors_list, descriptors, axis=0)
        if test and i == 9:
            print("Done loading test images.")
            break

    k_means_model = KMeans(n_clusters=num_clusters, random_state=42)
    print(f"""
    Training KMeans model with {num_clusters} clusters on {i+1} images.
    - Number of descriptors: {len(sift_vectors_list)}
    - Images are resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}
    - Using feature: {feature}
        """)
    k_means_model.fit(sift_vectors_list.astype(np.float32))
    print("K-Means model trained successfully. Start saving the model...")
    save_model(k_means_model, model_path)
    print("K-Means model saved successfully.")

# --gmm-- #
def train_and_save_gmm_model(data_set: BaseDataset,
                             num_clusters: int,
                             feature: str='sift',
                             test = False,
                             model_path: str='models/pickle_model_files/gmm_model.pkl') -> None:
    """
    Trains the gmm model on the image features and saves it as a .pkl file.

    **Note**: Currently only implement for SIFT features.

    :param data_set: CustomDataSet object
    :param feature: 'sift' or 'root_sift'
    :param model_path: path to save the model
    :param num_clusters: number of clusters
    :param test: if True, only 10 images are used for training

    :raises ValueError: If the length of the descriptor vector is not 128
    :raises ValueError: If the feature is not 'sift' or 'root_sift'
    """
    sift_vectors_list = np.empty((0, 128))
    if feature == 'sift':
        feature_extractor = sift
    elif feature == 'root_sift':
        feature_extractor = root_sift
    else:
        raise ValueError(f"Feature has to be 'sift' or 'root_sift'. {feature} is not supported.")
    for i, (img, *rest) in enumerate(data_set):
        _, descriptors = feature_extractor(resize(img, IMAGE_SIZE))
        sift_vectors_list = np.append(sift_vectors_list, descriptors, axis=0)
        if test and i == 9:
            print("Done loading test images.")
            break

    gmm_model = GaussianMixture(n_components=num_clusters, random_state=42, covariance_type='diag')
    print(f"""
    Training GMM model with {num_clusters} clusters on {i+1} images.
    - Number of descriptors: {len(sift_vectors_list)}
    - Images are resized to {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}
    - Using feature: {feature}
        """)
    gmm_model.fit(sift_vectors_list.astype(np.float32))
    print("GMM model trained successfully. Start saving the model...")
    save_model(gmm_model, model_path)
    print("GMM model saved successfully.")

# --pca-- #
def train_and_save_pca_model(data_set: BaseDataset,
                             feature_type: str,
                             cluster_model: Union[KMeans, GaussianMixture],
                             num_components: int,
                             test = False,
                             model_path: str='models/pickle_model_files/pca_model.pkl') -> None:
        """
        TODO: specify number of input and output features in the name of the model file.
        Trains the PCA model on the given features matrix and saves it as a .pkl file.

        **Note**: train the k-means or gmm model, then use these models to get the features matrix.
        After that, train the PCA model on the features matrix.

        :param data_set: CustomDataSet object
        :param feature_type: 'vlad' or 'fisher'
        :param num_components: number of components to keep
        :param model_path: path to save the model
        :param test: if True, only 10 images are used for training

        :raises ValueError: If the wrong cluster model is passed for the feature type
        :raises ValueError: If the feature type is not 'vlad' or 'fisher'
        """
        image_data = data_set[0:] if not test else data_set[0:10]
        features_matrix = []
        if feature_type == 'vlad':
            if not isinstance(cluster_model, KMeans):
                raise ValueError("Cluster model has to be KMeans for VLAD features.")
            for img, _ in image_data:
                features_matrix.append(VLAD(image=img, k_means=cluster_model, flatten=True).vector)
        elif feature_type == 'fisher':
            if not isinstance(cluster_model, GaussianMixture):
                raise ValueError("Cluster model has to be GaussianMixture for Fisher features.")
            for img, _ in image_data:
                features_matrix.append(FisherVector(image=img, gmm=cluster_model, flatten=True).vector)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}. Use 'vlad' or 'fisher' instead.")

        pca = PCA(n_components=num_components)
        standarized_features = standardize_data(features_matrix:=np.array(features_matrix), axis=0)
        print(f"""
        Training PCA model with {num_components} components on {features_matrix.shape[0]} images.
        - Number of features: {features_matrix.shape[1]}
            """)
        pca.fit(standarized_features)
        print(f"PCA model trained successfully. Start saving the model...")
        save_model(pca, model_path)
        print("PCA model saved successfully.")


def main():
    """
    Trains and saves the KMeans and GMM models for the SIFT and RootSIFT features.
    - Features used: SIFT, RootSIFT
    - Number of clusters: 32, 64, 128, 256
    """
    data_set = ExcavatorDataset(return_type='image', purpose='train')
    num_clusters = [32, 64, 128, 256]
    for num_cluster in num_clusters:
        for feature in ['sift', 'root_sift']:
            train_and_save_gmm_model(data_set,
                                     num_cluster,
                                     feature=feature,
                                     test=True,
                                     model_path=f'models/pickle_model_files/gmm_model_k{num_cluster}_{feature}.pkl')
            train_and_save_k_means_model(data_set,
                                         num_cluster,
                                         feature=feature,
                                         test=True,
                                         model_path=f'models/pickle_model_files/k_means_model_k{num_cluster}_{feature}.pkl')

if __name__ == '__main__':
    start = time.time()
    main()
    print(f"Time taken: {time.time()-start:.2f} seconds.")








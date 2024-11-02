import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from src.metrics import VLAD, FisherVector
from src.datasets import BaseDataset


def calc_vlad_similarity(test_img: np.ndarray,
                         anchor_img: np.ndarray,
                         k_means_model: KMeans) -> float:
    """
    Compute VLAD vectors of the given images and return the cosine similarity between them.
    :param test_img: test image
    :param anchor_img: anchor image
    :param k_means_model: k-means model
    :return: cosine similarity
    """
    vlad_anchor_img = VLAD(image=anchor_img, k_means=k_means_model, flatten=True)
    vlad_test_img = VLAD(image=test_img,  k_means=k_means_model, flatten=True)

    vlad_anchor_img_vector = vlad_anchor_img.vector.reshape(1, -1)
    vlad_test_img_vector = vlad_test_img.vector.reshape(1, -1)

    return cosine_similarity(vlad_anchor_img_vector, vlad_test_img_vector)[0][0]

def calc_fisher_similarity(test_img: np.ndarray,
                            anchor_img: np.ndarray,
                            gmm_model: GaussianMixture) -> float:
     """
     Compute Fisher vectors of the given images and return the cosine similarity between them.
     :param test_img: test image
     :param anchor_img: anchor image
     :param gmm_model: GMM model
     :return: cosine similarity
     """
     fisher_anchor_img = FisherVector(image=anchor_img, gmm=gmm_model, flatten=True)
     fisher_test_img = FisherVector(image=test_img, gmm=gmm_model, flatten=True)

     fisher_anchor_img_vector = fisher_anchor_img.vector.reshape(1, -1)
     fisher_test_img_vector = fisher_test_img.vector.reshape(1, -1)

     return cosine_similarity(fisher_anchor_img_vector, fisher_test_img_vector)[0][0]

if __name__ == "__main__":
    dataset = BaseDataset(plot=True)
    data = dataset[18:20]
    img_1, label_1 = data[0]
    print(f"Image 1 label: {label_1}")
    img_2, label_2 = data[1]



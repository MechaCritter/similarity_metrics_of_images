import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.metrics import VLAD, FisherVector
from src.datasets import CustomDataSet
from models.Clustering import GlobalKMeans, GlobalGMM


def calc_vlad_similarity(test_img: np.ndarray,
                         anchor_img: np.ndarray,
                         k_means_model: GlobalKMeans) -> float:
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
                            gmm_model: GlobalGMM) -> float:
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
    dataset = CustomDataSet(plot=True)
    data = dataset[18:20]
    img_1, label_1 = data[0]
    print(f"Image 1 label: {label_1}")
    img_2, label_2 = data[1]
    print(f"Image 2 label: {label_2}")
    k_means_model = GlobalKMeans()
    gmm_model = GlobalGMM()
    k_means_model.load_model('models/clustering/k_means_model_flower_car_500imgs.pkl')
    gmm_model.load_model('models/clustering/gmm_model_flower_car_500imgs.pkl')
    vlad_score = calc_vlad_similarity(img_1, img_2, k_means_model)
    fisher_score = calc_fisher_similarity(img_1, img_2, gmm_model)
    print(f"VLAD similarity between images 1 and 2: {vlad_score}")
    print(f"Fisher similarity between images 1 and 2: {fisher_score}")


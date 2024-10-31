import numpy as np
from sklearn.decomposition import PCA

from models.Clustering import GlobalKMeans, GlobalGMM
from utils import sift, resize, root_sift, standardize_data
from datasets import CustomDataSet

# --k_means-- #
def train_k_means_model(data_set: CustomDataSet,
                        k_means_model: GlobalKMeans,
                        feature: str='sift',
                        model_path: str='models/clustering/k_means_model_flower_car_500imgs.pkl') -> None:
    """
    Trains the k-means model on the image features and saves it as a .pkl file.

    **Note**: Currently only implement for SIFT features.

    :param data_set: CustomDataSet object
    :param k_means_model: GlobalKMeans object
    :param feature: 'sift' or 'root_sift'
    :param model_path: path to save the model

    :raises ValueError: If the length of the descriptor vector is not 128
    :raises ValueError: If the feature is not 'sift' or 'root_sift'
    """
    flower_data = data_set[0:]
    sift_vectors_list = np.empty((0, 128))
    if feature == 'sift':
        feature_extractor = sift
    elif feature == 'root_sift':
        feature_extractor = root_sift
    else:
        raise ValueError(f"Feature has to be 'sift' or 'root_sift'. {feature} is not supported.")
    for img in flower_data:
        _, descriptors = feature_extractor(resize(img[0], 256))
        for descriptor in descriptors:
            if len(descriptor) != 128:
                raise ValueError("Length of descriptor vector has to be 128.")

            sift_vectors_list = np.append(sift_vectors_list, [descriptor], axis=0)

    if dim_reduction_factor:
        pca = PCA(n_components=12 // dim_reduction_factor)
        sift_vectors_list = standardize_data(sift_vectors_list, axis=0)
        sift_vectors_list = pca.fit_transform(sift_vectors_list)
    print(f"Shape of sift_vectors_list with dim_reduction_factor: {dim_reduction_factor} is: ", sift_vectors_list.shape)

    k_means_model.train(sift_vectors_list)
    k_means_model.save_model(model_path)

# --gmm-- #
def train_gmm_model(data_set: CustomDataSet,
                    gmm_model: GlobalGMM,
                    feature: str='sift',
                    model_path: str='models/clustering/gmm_model_flower_car_500imgs.pkl') -> None:
    """
    Trains the gmm model on the image features and saves it as a .pkl file.

    **Note**: Currently only implement for SIFT features.

    :param data_set: CustomDataSet object
    :param gmm_model: GlobalGMM object
    :param feature: 'sift' or 'root_sift'
    :param model_path: path to save the model

    :raises ValueError: If the length of the descriptor vector is not 128
    :raises ValueError: If the feature is not 'sift' or 'root_sift'
    """
    flower_data = data_set[0:500]
    sift_vectors_list = np.empty((0, 128))
    if feature == 'sift':
        feature_extractor = sift
    elif feature == 'root_sift':
        feature_extractor = root_sift
    else:
        raise ValueError(f"Feature has to be 'sift' or 'root_sift'. {feature} is not supported.")
    for img in flower_data:
        _, descriptors = feature_extractor(resize(img[0], 256))
        for descriptor in descriptors:
            if len(descriptor) != 128:
                raise ValueError("Length of descriptor vector has to be 128.")

            sift_vectors_list = np.append(sift_vectors_list, [descriptor], axis=0)

    if dim_reduction_factor:
        pca = PCA(n_components=12 // dim_reduction_factor)
        sift_vectors_list = standardize_data(sift_vectors_list, axis=0)
        sift_vectors_list = pca.fit_transform(sift_vectors_list)
    print(f"Shape of sift_vectors_list with dim_reduction_factor: {dim_reduction_factor} is: ", sift_vectors_list.shape)

    gmm_model.train(sift_vectors_list)
    gmm_model.save_model(model_path)

if __name__ == '__main__':
    flower_data_set = CustomDataSet(purpose='train')
    num_clusters = 16
    dim_reduction_factor = 2
    k_means_model = GlobalKMeans(num_clusters=num_clusters)
    train_k_means_model(flower_data_set,
                        k_means_model,
                        dim_reduction_factor=dim_reduction_factor,
                        model_path=f'models/pickle_model_files/k_means{num_clusters}_pca_{128//dim_reduction_factor}_excavator.pkl')

# if __name__ == '__main__':
#     flower_data_set = CustomDataSet(purpose='train', shuffle=True, equal_class_distribution=True)
#     gmm_model = GlobalGMM(num_clusters=4)
#     train_gmm_model(flower_data_set, gmm_model)




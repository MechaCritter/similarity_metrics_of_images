import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.losses import DiceLoss

from src.metrics import VLAD, FisherVector
from src.utils import multi_class_dice_score, multiclass_iou
from src.datasets import ExcavatorDataset, Excavators
from models.Segmentation import UNet, DeepLabV3


def process_image(image: torch.Tensor,
                  ground_truth_mask: torch.Tensor,
                  k_means: KMeans,
                  gmm: GaussianMixture,
                  u_net: UNet,
                  deeplabv3: DeepLabV3,
                  num_classes: int) -> dict:
    """
    Process the image to extract information for evaluation.

    :param image: target image
    :param ground_truth_mask: ground truth mask of the image
    :param k_means: KMeans model
    :param gmm: Gaussian Mixture Model
    :param u_net: UNet model
    :param deeplabv3: DeepLabV3 model
    :param num_classes: number of classes in the dataset
    """
    # Compute vectors
    vlad = VLAD(image=image, k_means=k_means, flatten=True).vector
    fisher = FisherVector(image=image, gmm=gmm, flatten=True).vector

    # Model preds
    u_net_confidence, u_net_pred_mask  = u_net.predict_single_image(image)
    deeplabv3_confidence, deeplabv3_pred_mask = deeplabv3.predict_single_image(image)

    # Compute metrics
    u_net_iou = multiclass_iou(u_net_pred_mask, ground_truth_mask, num_classes=num_classes)
    deeplabv3_iou = multiclass_iou(deeplabv3_pred_mask, ground_truth_mask, num_classes=num_classes)
    u_net_dice_score = multi_class_dice_score(u_net_pred_mask, ground_truth_mask, num_classes=num_classes)
    deeplabv3_dice_score = multi_class_dice_score(deeplabv3_pred_mask, ground_truth_mask, num_classes=num_classes)

    return {
        'vlad_vector': vlad,
        'fisher_vector': fisher,
        'u_net_confidence': u_net_confidence,
        'u_net_iou': u_net_iou,
        'u_net_dice': u_net_dice_score,
        'deeplabv3_confidence': deeplabv3_confidence,
        'deeplabv3_iou': deeplabv3_iou,
        'deeplabv3_dice': deeplabv3_dice_score
    }










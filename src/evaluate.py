from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Type
import json

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torchvision import transforms

from src.metrics import *
from src.utils import multi_class_dice_score, multiclass_iou, gaussian_blur, compress_image, get_enum_member
from src.datasets import ExcavatorDataset, Excavators
from models.Segmentation import UNet, DeepLabV3

TRANSFORMER = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
])

IMAGE_DATA = defaultdict(lambda: {'path': None,
                                  'label': None,
                                  'type': None, # 'train' or 'test'
                                  'vlad_vector': None,
                                  'fisher_vector': None,
                                  'u_net_confidence': None,
                                  'u_net_iou': None,
                                  'u_net_dice_score': None,
                                  'deeplabv3_confidence': None,
                                  'deeplabv3_iou': None,
                                  'deeplabv3_dice_score': None})

GAUSSIAN_SIGMAS = [2*i for i in range(1, 10)]
COMPRESSION_QUALITIES = [i for i in range(10, 101, 10)]

@dataclass
class ImageComparisonMetrics:
    target_image: np.ndarray
    target_image_path: str
    anchor_image: np.ndarray
    anchor_image_path: str
    vlad_cosine_similarity: Optional[float] = None
    fisher_cosine_similarity: Optional[float] = None
    ssim_score: Optional[float] = None
    ms_ssim_score: Optional[float] = None
    gaussian_blur_sigma: Optional[float] = None
    image_quality: Optional[float] = None

# TODO: add a function to compute model outputs, then remove the model predictions from the process_image function (then, call this function in the process_image function)

def process_image(image: np.ndarray,
                  ground_truth_mask: np.ndarray,
                  k_means: KMeans,
                  gmm: GaussianMixture,
                  u_net: UNet,
                  deeplabv3: DeepLabV3,
                  num_classes: int) -> dict:
    """
    TODO: this method should take a dataset instead of image and mask arrays.
    # TODO: VLAD and Fisher only work with numpy arrays, whereas the models work with torch tensors. Resolve this issue!
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
        'u_net_dice_score': u_net_dice_score,
        'deeplabv3_confidence': deeplabv3_confidence,
        'deeplabv3_iou': deeplabv3_iou,
        'deeplabv3_dice_score': deeplabv3_dice_score
    }


def compute_ssim_scores(dataset: ExcavatorDataset,
                        class_enum: Type[Excavators],
                        gaussian_sigmas: list,
                        compression_qualities: list,
                        u_net_model: UNet,
                        deeplabv3_model: DeepLabV3,
                        output_json: str='res/ssim_scores.json') -> None:
    """
    Compute SSIM and MS-SSIM scores between all image pairs in the dataset,
    applying different levels of Gaussian blur and compression. Furthermore, model predictions
    are computed for each image pair.

    **Note**: pass 'all' to return_type in the dataset object to return image array, mask, label, and path.

    :param dataset: Dataset object returning ImageData instances.
    :param class_enum: Enum class for the dataset classes.
    :param gaussian_sigmas: List of sigma values for Gaussian blur.
    :param compression_qualities: List of quality values for compression.
    :param u_net_model: UNet class.
    :param deeplabv3_model: DeepLabV3 class.
    :param output_json: Path to output JSON file.
    """
    ssim_results = []
    num_images = len(dataset)
    num_cls = len(class_enum)
    TEMPORARY = 4 # Delete this line after testing
    for i in range(TEMPORARY): # TODO: Change to num_images after testing
        img_data_i = dataset[i]
        image_i_array = img_data_i.image_array
        image_i_mask = img_data_i.mask_array
        image_i_label = img_data_i.label
        image_i_path = img_data_i.image_path

        for j in range(i+1, TEMPORARY):  # Avoid duplicate pairs
            img_data_j = dataset[j]
            image_j_array = img_data_j.image_array
            image_j_mask = img_data_j.mask_array
            image_j_label = img_data_j.label
            image_j_path = img_data_j.image_path

            for sigma in gaussian_sigmas:
                # Apply Gaussian blur
                print(f"Applying Gaussian blur with sigma={sigma} to images {i} and {j}")
                blurred_i = gaussian_blur(image_i_array, sigma=sigma)
                blurred_j = gaussian_blur(image_j_array, sigma=sigma)

                for quality in compression_qualities:
                    # Apply compression
                    print(f"Applying compression with quality={quality} to images {i} and {j}")
                    compressed_i = compress_image(blurred_i, quality=quality)
                    compressed_j = compress_image(blurred_j, quality=quality)
                    print(f"Shape of compressed image {i}: {compressed_i.shape}")
                    print(f"Shape of compressed image {j}: {compressed_j.shape}")

                    # Compute SSIM
                    ssim_score = SSIM(compressed_i, compressed_j, data_range=255)
                    ms_ssim_score = MS_SSIM(compressed_i, compressed_j, data_range=255)

                    # Get enum classes
                    if not (class_i_enum := get_enum_member(image_i_label, class_enum)):
                        raise ValueError(f"Class {image_i_label} not found in the enum class.")

                    if not (class_j_enum := get_enum_member(image_j_label, class_enum)):
                        raise ValueError(f"Class {image_j_label} not found in the enum class.")

                    print(f"Class of interest of image {i}: {class_i_enum}")
                    print(f"Class of interest of image {j}: {class_j_enum}")

                    # Model predictions
                    u_net_i_proba, u_net_i_pred_mask = u_net_model.predict_single_image(compressed_i, gt_mask=image_i_mask, cls_index=class_i_enum)
                    u_net_j_proba, u_net_j_pred_mask = u_net_model.predict_single_image(compressed_j, gt_mask=image_j_mask, cls_index=class_j_enum)
                    deeplabv3_i_proba, deeplabv3_i_pred_mask = deeplabv3_model.predict_single_image(compressed_i, gt_mask=image_i_mask, cls_index=class_i_enum)
                    deeplabv3_j_proba, deeplabv3_j_pred_mask = deeplabv3_model.predict_single_image(compressed_j, gt_mask=image_j_mask, cls_index=class_j_enum)

                    print(f"""Shape of image {i}'s mask: {image_i_mask.shape}")
                        Image {i}'s mask is on device: {image_i_mask.device}")
                        Shape of image {j}'s mask: {image_j_mask.shape}")
                        Image {j}'s mask is on device: {image_j_mask.device}")
                        Shape of u-net prediction mask {i}: {u_net_i_pred_mask.shape}")
                        u-net's prediction mask {i} is on device: {u_net_i_pred_mask.device}")
                        Shape of u-net prediction mask {j}: {u_net_j_pred_mask.shape}")
                        u-net's prediction mask {j} is on device: {u_net_j_pred_mask.device}")
                        Shape of deeplabv3 prediction mask {i}: {deeplabv3_i_pred_mask.shape}")
                        DeepLabV3's prediction mask {i} is on device: {deeplabv3_i_pred_mask.device}")
                        Shape of deeplabv3 prediction mask {j}: {deeplabv3_j_pred_mask.shape}")
                        DeepLabV3's prediction mask {j} is on device: {deeplabv3_j_pred_mask.device}""")

                    # Calculate dice score and IoU between the predicted masks and the ground truth masks
                    u_net_i_dice_score = multi_class_dice_score(u_net_i_pred_mask, image_i_mask, num_classes=num_cls)
                    u_net_i_iou = multiclass_iou(u_net_i_pred_mask, image_i_mask, num_classes=num_cls)
                    u_net_j_dice_score = multi_class_dice_score(u_net_j_pred_mask, image_j_mask, num_classes=num_cls)
                    u_net_j_iou = multiclass_iou(u_net_j_pred_mask, image_j_mask, num_classes=num_cls)
                    deeplabv3_i_dice_score = multi_class_dice_score(deeplabv3_i_pred_mask, image_i_mask, num_classes=num_cls)
                    deeplabv3_i_iou = multiclass_iou(deeplabv3_i_pred_mask, image_i_mask, num_classes=num_cls)
                    deeplabv3_j_dice_score = multi_class_dice_score(deeplabv3_j_pred_mask, image_j_mask, num_classes=num_cls)
                    deeplabv3_j_iou = multiclass_iou(deeplabv3_j_pred_mask, image_j_mask, num_classes=num_cls)

                    # Prepare result data
                    result_data = {
                        'image_i_path': image_i_path,
                        'image_i_label': image_i_label,
                        'u_net_i_proba': u_net_i_proba,
                        'u_net_i_dice_score': float(u_net_i_dice_score),
                        'u_net_i_iou': float(u_net_i_iou),
                        'deeplabv3_i_proba': deeplabv3_i_proba,
                        'deeplabv3_i_dice_score': float(deeplabv3_i_dice_score),
                        'deeplabv3_i_iou': float(deeplabv3_i_iou),
                        'image_j_path': image_j_path,
                        'image_j_label': image_j_label,
                        'u_net_j_proba': u_net_j_proba,
                        'u_net_j_dice_score': float(u_net_j_dice_score),
                        'u_net_j_iou': float(u_net_j_iou),
                        'deeplabv3_j_proba': deeplabv3_j_proba,
                        'deeplabv3_j_dice_score': float(deeplabv3_j_dice_score),
                        'deeplabv3_j_iou': float(deeplabv3_j_iou),
                        'sigma': sigma,
                        'quality': quality,
                        'ssim_score': float(ssim_score.value),
                        'ms_ssim_score': float(ms_ssim_score.value)
                    }

                    # Append to results
                    ssim_results.append(result_data)

    # Save results to JSON file
    with open(output_json, 'w') as f:
        json.dump(ssim_results, f, indent=4)

if __name__ == '__main__':
    dataset = ExcavatorDataset(purpose='train', return_type='all', transform=TRANSFORMER)
    dlv3 = DeepLabV3(model_path='models/torch_model_files/DeepLabV3.pt')
    unet = UNet(model_path='models/torch_model_files/UNet.pt')
    compute_ssim_scores(dataset=dataset,
                        class_enum=Excavators,
                        gaussian_sigmas=GAUSSIAN_SIGMAS,
                        compression_qualities=COMPRESSION_QUALITIES,
                        u_net_model=unet,
                        deeplabv3_model=dlv3,
                        output_json='res/ssim_scores.json')













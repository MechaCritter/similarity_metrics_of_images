import logging
from enum import Enum

import torch
import torch.nn.functional as F
from torch.fft import ifftshift
from torchvision import transforms
from segmentation_models_pytorch import Unet, DeepLabV3
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU


from src.datasets import BaseDataset, ExcavatorDataset, Excavators


class UNet:
    def __init__(self,
                 data_set: BaseDataset = None,
                 model_path: str = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 metrics: list = None,
                 activation: str = 'softmax2d',
                 encoder_name: str = 'resnet34',
                encoder_weights: str = 'imagenet',
                 classes: int = 12,
                 lr: float = 1e-3
                 ):
        """
        Class constructor.

        :param data_set:
        :param model_path:
        :param criterion: default is DiceLoss
        :param optimizer: default is Adam
        :param metrics: default is IoU
        :param activation:
        :param encoder_name:
        :param encoder_weights:
        :param classes:
        """
        self._logger = logging.getLogger('UNet')
        self.encoder_name = encoder_name
        self.classes = classes
        self.activation = activation
        self.lr = lr

        if model_path:
            self.model = torch.jit.load(model_path)
            self._logger.info(f"Model loaded from {model_path} with parameters: {self.model}")
        else:
            self.model = Unet(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              classes=classes,
                              activation=activation)
            self._logger.info(f"""New model created with the following info:
            - Encoder name: {self.encoder_name}
            - Activation: {self.activation}
            - Classes: {self.classes}""")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Device used for UNet: {self.device}")
        self.model.to(self.device)

        self.criterion = criterion if criterion else DiceLoss()
        self.metrics = metrics if metrics else [IoU()]
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=self.lr)


        # self.model = Unet(encoder_name=encoder_name,
        #                   encoder_weights=encoder_weights,
        #                   classes=classes,
        #                   activation=activation)

        #self.model.load_state_dict(torch.load(model_path, map_location=self.device))



        self.data_set = data_set

        self._train_epoch = TrainEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True
        )
        self._valid_epoch = ValidEpoch(
            model=self.model,
            loss=self.criterion,
            metrics=self.metrics,
            device=self.device,
            verbose=True
        )

    def predict(self,
                image: torch.Tensor,
                gt_mask: torch.Tensor,
                cls_index: int | Enum) -> tuple[float, torch.Tensor]:
        """
        Predict the prediction score and the mask of the given image.

        How it works
        ============
        First, get the coordinate of all pixels that correspond to the mask of the class of interest. Then,
        in the predicted proba matrix, sum up all the probabilities of the class of interest at the corresponding
        coordinates. Then, divide it by the number of pixels in total to avoid bias when the number of mask
        pixels in two images are different.

        :param image: input image as a tensor
        :param gt_mask: ground truth mask as a tensor
        :param cls_index: index of class of interest. Can also be an Enum object

        :return: average confidence of the class of interest and the predicted mask
        """
        if isinstance(cls_index, Enum):
            self._logger.info("Enum object for class %s detected. Converting to integer %s", cls_index, cls_index:=cls_index.value)
        self.model.eval()
        image = image.to(self.device).unsqueeze(0)
        gt_mask = gt_mask.to(self.device).unsqueeze(0)
        with torch.no_grad():
            # Output: Probabilities of each class: torch.tensor([1, num_classes, H, W])
            output = self.model(image)
            self._logger.debug("background proba: %s", background_proba:=output[0, 0, :, :].cpu().numpy())
            self._logger.debug("Min proba: %s", min_bk:=background_proba.min())
            # # Softmax across class dimension: torch.tensor([1, num_classes, H, W])
            # proba_softmax = F.softmax(output, dim=1)

            # Get proba matrix of class of interest: torch.tensor([H, W])
            class_probs = output[0, cls_index, :, :]
            self._logger.debug("class probs: %s", class_prob:=class_probs.cpu().numpy())

            # Get coordinates of mask pixels that correspond to the class of interest: torch.tensor([H, W])
            mask_idx = (gt_mask.squeeze(0) == cls_index)
            self._logger.debug("Mask index shape: %s", mask_idx.shape)

            # Get the sum of probabilities of the class of interest
            total_confidence = class_probs[mask_idx].sum().item()
            self._logger.debug("Total confidence: %s", total_confidence)

            # Get number of pixels of the class of interest
            num_pixels = mask_idx.sum().item()
            self._logger.debug("Number of pixels containing class %s: %s", cls_index, num_pixels)

            if num_pixels == 0:
                avr_confidence = 0
            else:
                avr_confidence = total_confidence / num_pixels

            # Predicted mask: torch.tensor([1, H, W])
            _, predicted_mask = torch.max(output, 1)
            self._logger.debug("Average confidence of the class of interest: %s", avr_confidence)
            self._logger.debug("predicted mask: %s", pred:=predicted_mask.cpu().numpy())
            self._logger.debug("11 is in the mask: %s", 11 in pred)
            return avr_confidence, predicted_mask

class DeepLabV3:
    def __init__(self, model_path: str = 'models/torch_model_files/DeepLabV3.pt'):
        self.model = torch.jit.load(model_path)

if __name__ =="__main__":
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader

    unet = UNet(model_path='models/torch_model_files/UNet.pt')

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    dataset = ExcavatorDataset(transform=transformer, purpose = 'test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # conf_excavator = []
    # for i in range(100,140):
    #     img, mask, lbl = dataset[i][0]
    #     print("Shape of image:", img.shape)
    #     mask = dataset.rgb_to_mask(mask)
    #     print("Shape of mask:", mask.shape)
    #     class_of_interest = Excavators.TRUCK
    #     confidence, pred_mask = unet.predict(img, mask, class_of_interest)
    #     pred_mask = pred_mask[0, :, :].cpu().numpy()
    #     conf_excavator.append(confidence)
    #     print("Shape of predicted mask:", pred_mask.shape)
    # print("Confidence of excavator:", conf_excavator)
    eval_logs= unet._valid_epoch.run(dataloader)
    print("Evaluation logs:", eval_logs)

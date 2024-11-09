import logging
import os
from enum import Enum

from torch.utils.data import DataLoader

import torch
from segmentation_models_pytorch import Unet, DeepLabV3
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU

from src.datasets import BaseDataset, ExcavatorDataset, Excavators
from src.utils import mask_to_rgb
from src.losses import MultiClassDiceLoss

class UNetModel:
    def __init__(self,
                 model_path: str = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 metrics: list = None,
                 activation: str = None,
                 encoder_name: str = 'resnet34',
                encoder_weights: str = 'imagenet',
                 classes: int = 12,
                 lr: float = 1e-3,
                 momentum: float = 0.9
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
            self.model = torch.load(model_path)
            self._logger.info(f"Model loaded from {model_path} with parameters: {self.model}")
        else:
            self.model = Unet(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              classes=classes,
                              activation=activation)
            self._logger.info(f"""New U-Net model created with the following info:
                            - Encoder name: {self.encoder_name}
                            - Activation: {self.activation}
                            - Classes: {self.classes}""")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Device used for UNet: {self.device}")
        self.model.to(self.device)

        self.criterion = criterion if criterion else MultiClassDiceLoss(mode='multiclass') # TODO: ignore background`?
        self.metrics = metrics if metrics else [IoU()]
        self.optimizer = optimizer if optimizer else torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum)


        # self.model = Unet(encoder_name=encoder_name,
        #                   encoder_weights=encoder_weights,
        #                   classes=classes,
        #                   activation=activation)

        #self.model.load_state_dict(torch.load(model_path, map_location=self.device))

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

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              model_save_path: str = 'models/torch_model_files/DeepLabV3.pt') -> None:
        """
        # TODO: ignore background class
        Train the model with the given data loaders.

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param num_epochs: number of epochs to train
        :param model_save_path: path to save the model
        """
        self._logger.info("Training DeepLabV3 model")
        if os.path.exists(model_save_path):
            raise FileExistsError(f"Model file already exists at {model_save_path}. Please delete it or specify a new path.")
        self._logger.info("""Training parameters:
        - Number of epochs: %s
        - Model save path: %s, 
        - Device: %s,
        - Criterion: %s,
        - Optimizer: %s,
        - Metrics: %s,
        - Activation: %s,
        - Encoder weights: %s,
        - Classes: %s,
        - Learning rate: %s
        """, num_epochs, model_save_path, self.device, self.criterion, self.optimizer, self.metrics, self.activation, self.encoder_name, self.classes, self.lr)
        max_score = 0
        for epoch in range(num_epochs):
            print('\nEpoch: {}'.format(epoch))
            train_logs = self._train_epoch.run(train_loader)
            valid_logs = self._valid_epoch.run(val_loader)

            # Save best model
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model.state_dict(), model_save_path)
                self._logger.info('Model saved at: %s', model_save_path)

    def predict_single_image(self,
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

        :param image: input image as a tensor. Shape. [1, C, H, W]
        :param gt_mask: ground truth mask as a tensor. Shape: [1, H, W]
        :param cls_index: index of class of interest. Can also be an Enum object

        :return: average confidence of the class of interest and the predicted mask

        :raises ValueError: if cls_index is not an integer or an Enum object with integer value
        """
        if not isinstance(cls_index, int):
            if not isinstance(cls_index, Enum):
                raise ValueError(f"cls_index must be an integer or an Enum object with integer value, got {type(cls_index)} instead.")
            else:
                self._logger.info("Enum object for class %s detected. Converting to integer %s", cls_index, cls_index:=cls_index.value)

        self.model.eval()
        image = image.to(self.device).unsqueeze(0)
        gt_mask = gt_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Output: Probabilities of each class: torch.tensor([1, num_classes, H, W])
            self._logger.debug("Image type: %s", image.dtype)
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
            return avr_confidence, predicted_mask.cpu().squeeze(0)

class DeepLabV3Model:
    def __init__(self,
                 model_path: str = None,
                 criterion: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 metrics: list = None,
                 activation: str = None,
                 encoder_name: str = 'resnet34',
                 encoder_weights: str = 'imagenet',
                 classes: int = 12,
                 lr: float = 1e-3,
                 momentum: float = 0.9
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
        self._logger = logging.getLogger('DeepLabV3')
        self.encoder_name = encoder_name
        self.classes = classes
        self.activation = activation
        self.lr = lr

        if model_path:
            self.model = torch.load(model_path)
            self._logger.info(f"Model loaded from {model_path} with parameters: {self.model}")
        else:
            self.model = DeepLabV3(encoder_name=encoder_name,
                                        encoder_weights=encoder_weights,
                                        classes=classes,
                                        activation=activation)
            self._logger.info(f"""New DeepLabV3 model created with the following info:
            - Encoder name: {self.encoder_name}
            - Activation: {self.activation}
            - Classes: {self.classes}""")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._logger.info(f"Device used for UNet: {self.device}")
        self.model.to(self.device)

        self.criterion = criterion if criterion else MultiClassDiceLoss(mode='multiclass')
        # This code was needed to avoid an error in the library 'segmentation_models_pytorch'
        self.criterion.__name__ = 'DiceLoss'

        self.metrics = metrics if metrics else [IoU()]
        self.optimizer = optimizer if optimizer else torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum)


        # self.model = Unet(encoder_name=encoder_name,
        #                   encoder_weights=encoder_weights,
        #                   classes=classes,
        #                   activation=activation)

        #self.model.load_state_dict(torch.load(model_path, map_location=self.device))

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

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              model_save_path: str = 'models/torch_model_files/DeepLabV3.pt') -> None:
        """
        # TODO: ignore background class
        Train the model with the given data loaders.

        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :param num_epochs: number of epochs to train
        :param model_save_path: path to save the model
        """
        self._logger.info("Training DeepLabV3 model")
        if os.path.exists(model_save_path):
            raise FileExistsError(f"Model file already exists at {model_save_path}. Please delete it or specify a new path.")
        self._logger.info("""Training parameters:
        - Number of epochs: %s
        - Model save path: %s, 
        - Device: %s,
        - Criterion: %s,
        - Optimizer: %s,
        - Metrics: %s,
        - Activation: %s,
        - Encoder weights: %s,
        - Classes: %s,
        - Learning rate: %s
        """, num_epochs, model_save_path, self.device, self.criterion, self.optimizer, self.metrics, self.activation, self.encoder_name, self.classes, self.lr)
        max_score = 0
        for epoch in range(num_epochs):
            print('\nEpoch: {}'.format(epoch))
            train_logs = self._train_epoch.run(train_loader)
            valid_logs = self._valid_epoch.run(val_loader)

            # Save best model
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model.state_dict(), model_save_path)
                self._logger.info('Model saved at: %s', model_save_path)

    def predict_single_image(self,
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

        :param image: input image as a tensor. Shape. [1, C, H, W]
        :param gt_mask: ground truth mask as a tensor. Shape: [1, H, W]
        :param cls_index: index of class of interest. Can also be an Enum object

        :return: average confidence of the class of interest and the predicted mask

        :raises ValueError: if cls_index is not an integer or an Enum object with integer value
        """
        if not isinstance(cls_index, int):
            if not isinstance(cls_index, Enum):
                raise ValueError(f"cls_index must be an integer or an Enum object with integer value, got {type(cls_index)} instead.")
            else:
                self._logger.info("Enum object for class %s detected. Converting to integer %s", cls_index, cls_index:=cls_index.value)

        self.model.eval()
        image = image.to(self.device).unsqueeze(0)
        gt_mask = gt_mask.to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Output: Probabilities of each class: torch.tensor([1, num_classes, H, W])
            self._logger.debug("Image type: %s", image.dtype)
            output = self.model(image)
            # # Softmax across class dimension: torch.tensor([1, num_classes, H, W])
            # proba_softmax = F.softmax(output, dim=1)

            # Get proba matrix of class of interest: torch.tensor([H, W])
            class_probs = output[0, cls_index, :, :]

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
            self._logger.debug("Unique values in predicted mask: %s", torch.unique(predicted_mask))
            return avr_confidence, predicted_mask.cpu().squeeze(0)

if __name__ =="__main__":
    from torchvision import transforms
    import cv2

    dlv3 = DeepLabV3Model(model_path='models/torch_model_files/DeepLabV3.pt')
    unet = UNetModel()
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    trainloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='train',
                                              return_type='image+mask',
                                              one_hot_encode_mask=True
                                              ), batch_size=20, shuffle=False) # TODO: setting batch size above 1 won't work?
    validloader = DataLoader(ExcavatorDataset(transform=transformer,
                                              purpose='validation',
                                              return_type='image+mask',
                                                one_hot_encode_mask=True
                                              ), batch_size=1, shuffle=False)
    dlv3.train(trainloader, validloader, 10, model_save_path='models/torch_model_files/DeepLabV3_Huy_lr1e-3_momentum9e-1.pt')

    # #


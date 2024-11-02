import os
from enum import Enum
from typing import Optional
from collections import Counter

import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from dotenv import load_dotenv

from src.config import *

setup_logging()
load_dotenv()


class BaseDataset(Dataset):
    """
    Allows for quick access dataset images, simply provide the link to folder containing train and test images.
    """
    def __init__(self,
                 train_img_data_dir: str = TRAIN_IMG_DATA_PATH_EXCAVATOR,
                 validation_img_data_dir: Optional[str] = None,
                 test_img_data_dir: str = TEST_IMG_DATA_PATH_EXCAVATOR,
                 train_mask_data_dir: Optional[str] = None,
                 validation_mask_data_dir: Optional[str] = None,
                 test_mask_data_dir: Optional[str] = None,
                 transform: transforms = None,
                 plot: bool = False,
                 verbose: bool = False,
                 purpose: str = 'train',
                 class_colors: dict = None):
        """
        Class constructor. Remember to pass data to whether train or test data directory.
        If only one path is provided, it is passed to the train dataset.

        :param train_img_data_dir: Path to the train data directory
        :param validation_img_data_dir: Path to the validation data directory (optional)
        :param test_img_data_dir: Path to the test data directory
        :param train_mask_data_dir: Path to the train mask data directory (for segmentation tasks)
        :param validation_mask_data_dir: Path to the validation mask data directory (for segmentation tasks)
        :param test_mask_data_dir: Path to the test mask data directory (for segmentation tasks)
        :param transform: Transformation to apply to the images
        :param plot: Whether to plot the images
        :param verbose: Whether to print out extra information for debugging
        :param purpose: Whether the data is for training, validation or testing
        :param class_colors: Dictionary containing the class colors for the dataset for segmentation tasks

        :raises FileNotFoundError: If 'train_data_dir' or 'test_data_dir' does not exist
        """
        self._logger = logging.getLogger('Data_Set')
        self._logger.name = self.__class__.__name__
        self._max_img_to_plot = 10
        self._verbose = verbose
        self.plot: bool = plot
        self._purpose = purpose
        self._class_colors = class_colors

        self._logger.debug(f"Initializing BaseDataset with purpose: {self._purpose}")
        self._logger.debug(f"Train image data directory: {train_img_data_dir}")
        self._logger.debug(f"Validation image data directory: {validation_img_data_dir}")
        self._logger.debug(f"Test image data directory: {test_img_data_dir}")

        if not os.path.exists(train_img_data_dir):
            raise FileNotFoundError(f"Directory {train_img_data_dir} does not exist.")

        if not os.path.exists(test_img_data_dir):
            raise FileNotFoundError(f"Directory {test_img_data_dir} does not exist.")

        self.train_img_data_dir = train_img_data_dir
        self.validation_img_data_dir = validation_img_data_dir
        self.test_img_data_dir = test_img_data_dir
        self.train_mask_data_dir = train_mask_data_dir
        self.validation_mask_data_dir = validation_mask_data_dir
        self.test_mask_data_dir = test_mask_data_dir

        self.images = []
        self.mask_annots = []

        self.transform: transforms = transform

        # Method calls
        self.load_images()
        if train_mask_data_dir:
            self.load_masks()

    @property
    def class_colors(self) -> dict:
        return self._class_colors

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.

        :raises ValueError: If the purpose is not 'train', 'validation' or 'test'
        """
        match self._purpose:
            case 'train':
                self.images = self._load_from_dir(self.train_img_data_dir)
            case 'validation':
                self.images = self._load_from_dir(self.validation_img_data_dir)
            case 'test':
                self.images = self._load_from_dir(self.test_img_data_dir)
            case _:
                raise ValueError(f"Purpose has to be 'train', 'validation' or 'test'.")
        if self._verbose:
            self._logger.info("Loaded %s images with purpose '%s'", len(self.images), self._purpose)

    def load_masks(self):
        """
        Load the masks from the directories. Called upon initialization of the class.

        :raises ValueError: If the purpose is not 'train', 'validation' or 'test'
        """
        match self._purpose:
            case 'train':
                self.mask_annots = self._load_from_dir(self.train_mask_data_dir, key='mask_path')
            case 'validation':
                self.mask_annots = self._load_from_dir(self.validation_mask_data_dir, key='mask_path')
            case 'test':
                self.mask_annots = self._load_from_dir(self.test_mask_data_dir, key='mask_path')
            case _:
                raise ValueError(f"Purpose has to be 'train', 'validation' or 'test'.")
        if self._verbose:
            self._logger.info("Loaded %s masks with purpose '%s'", len(self.mask_annots), self._purpose)

    def _load_from_dir(self, data_dir: str, key = 'image_path') -> None:
        """
        Internal method used to pass the data directory and return a list that contain dictionaries of the image path and label.

        :param data_dir: Path to the directory containing the images. Pass the path to the `data/train`and `data/test` directories. The names
        of the subdirectories are used as labels.
        :type data_dir: str

        :return: A list of path to the images and their labels
        :rtype: list
        """
        images = []
        for label in os.listdir(data_dir):
            for image in os.listdir(os.path.join(data_dir, label)):
                image_path = os.path.join(data_dir, label, image)
                images.append({key: image_path, 'label': label})

        return images

    def _plot_image(self, images_and_labels: list[tuple[np.ndarray, int]]) -> None:
        """
        Plot the image with its file path and label. Only call from inside.
        If image shape of (3, width, height) is passed, it is converted to (width, height, 3) before plotting.
        """
        for i in range(len(images_and_labels)):
            img, *rest, lbl = images_and_labels[i]
            _, ax = plt.subplots()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
            ax.set_title(f"Label: {lbl}")
            ax.axis('off')

        plt.show()

    def __getitem__(self, indices: int | slice) -> list[tuple[np.ndarray, int]]:
        """
        Get the data at the specified index range. If `plot` is set to True, `plot_image` is called.

        :param indices: Index or slice of the image to retrieve
        :type index: int | slice

        :return: A list containing tuples, each with an image, (mask) and label
        :rtype: list[tuple(np.ndarray, int)]

        :raises IndexError: If the index is out of range
        """
        if isinstance(indices, slice):
            data = [self.__get_single_item(index) for index in range(*indices.indices(len(self.images)))]
        else:
            data = [self.__get_single_item(indices)]

        if len(data) == 0:
            raise IndexError(f"Index out of range. Data for {self._purpose} purpose only contains {len(self.images)} images.")

        if self.plot:
            if len(data) > self._max_img_to_plot:
                self._logger.warning("Too many images to plot. Plotting only the first %s images",
                                     self._max_img_to_plot)
                self._plot_image(data[:self._max_img_to_plot])
            else:
                self._plot_image(data)

        return data

    def __len__(self) -> int:
        return len(self.images)

    def __get_single_item(self, index: int) -> tuple[np.ndarray, int] | tuple[np.ndarray, np.ndarray, int]:
        """
        Gets a single image and label (and mask, if path is specified) at the specified index. If
        `plot` is set to True, `plot_image` is called.

        :param index: Index of the image to retrieve
        :type index: int

        :return: Image and label (+ mask)
        :rtype: tuple[np.ndarray, int] | tuple[np.ndarray, np.ndarray, int]
        """
        image_path = self.images[index]['image_path']

        label = self.images[index]['label']
        self._logger.debug("Got label %s for image %s", label, image_path)

        self._logger.info("Retrieving image %s with label %s", image_path, label)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Load mask if provided
        if self.mask_annots:
            self._logger.info("Masks are detected for dataset %s", self.__class__.__name__)
            mask_path = self.mask_annots[index]['mask_path']
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                raise FileNotFoundError(f"""
                Mask not found for image {image_path}
                Mask path is supposed to be: {mask_path}
                Make sure you name the mask images the same as the image name.
                """)

            if self.transform:
                mask = self.transform(mask)
            return image, mask, label

        return image, label

class Excavators(Enum):
    BACKGROUND = 0
    BULLDOZER = 1
    CAR = 2
    CATERPILLAR = 3
    CRANE = 4
    CRUSHER = 5
    DRILLER = 6
    EXCAVATOR = 7
    HUMAN = 8
    ROLLER = 9
    TRACTOR = 10
    TRUCK = 11


class ExcavatorDataset(BaseDataset):
    """
    Excavator dataset class.
    """
    def __init__(self,
                 train_img_data_dir: str = TRAIN_IMG_DATA_PATH_EXCAVATOR,
                 train_mask_data_dir: str = TRAIN_MASK_DATA_PATH_EXCAVATOR,
                 test_img_data_dir: str = TEST_IMG_DATA_PATH_EXCAVATOR,
                 test_mask_data_dir: str = TEST_MASK_DATA_PATH_EXCAVATOR,
                 transform: transforms = None,
                 plot: bool = False,
                 verbose: bool = False,
                 purpose: str = 'train',
                 class_colors: dict = None):
        super().__init__(train_img_data_dir=train_img_data_dir,
                         train_mask_data_dir=train_mask_data_dir,
                         test_img_data_dir=test_img_data_dir,
                         test_mask_data_dir=test_mask_data_dir,
                         transform=transform,
                         plot=plot,
                         verbose=verbose,
                         purpose=purpose,
                         class_colors=class_colors)
        self._class_colors = {
            Excavators.BACKGROUND: torch.from_numpy(np.array([0, 0, 0])),
            Excavators.BULLDOZER: torch.from_numpy(np.array([0, 183, 235])),
            Excavators.CAR: np.array([255, 255, 0]),
            Excavators.CATERPILLAR: np.array([0, 16, 235]),
            Excavators.CRANE: np.array([199, 252, 0]),
            Excavators.CRUSHER: np.array([255, 0, 140]),
            Excavators.DRILLER: np.array([14, 122, 254]),
            Excavators.EXCAVATOR: np.array([255, 171, 171]),
            Excavators.HUMAN: np.array([254, 0, 86]),
            Excavators.ROLLER: np.array([255, 0, 255]),
            Excavators.TRACTOR: np.array([128, 128, 0]),
            Excavators.TRUCK: np.array([134, 34, 255]),
        }
        # Normalize and convert to tensors
        self._normalized_class_colors = {
            key: torch.tensor(value / 255.0, dtype=torch.float32)
            for key, value in self._class_colors.items()
        }

    def rgb_to_mask(self, rgb_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts RGB mask image to class index mask image.
        **Note**: broadcast the mask to shape (3xHxW) before passing it to this method.

        :param rgb_mask: RGB mask image tensor with shape (3, H, W)

        :return: Class index mask image
        """
        if not rgb_mask.shape[0] == 3:
            raise ValueError(f"RGB mask image has to have shape (3, H, W). Got shape: {rgb_mask.shape} Use `torch.permute` to change the shape.")

        mask = torch.zeros((rgb_mask.shape[-2], rgb_mask.shape[-1]), dtype=torch.float32)
        for cls, color in self._normalized_class_colors.items():
            mask[torch.all(rgb_mask.to(torch.float32) == color.view(3,1,1), axis=0)] = cls.value
        return mask


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    excavator_data = ExcavatorDataset(transform=transformer)
    img, mask, lbl = excavator_data[0][0]
    print("Shape of image:", img.shape)
    print("Shape of mask:", mask.shape)
    cls_mask = excavator_data.rgb_to_mask(mask)
    print("Unique values in mask:", torch.unique(cls_mask))

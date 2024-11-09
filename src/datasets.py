import os
from enum import Enum
from typing import Optional
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

from src.config import *
from src.utils import rgb_to_mask, mask_to_rgb, plot_image

setup_logging()
load_dotenv()

__all__ = ['ExcavatorDataset', 'Excavators', 'BaseDataset']

@dataclass
class ImageData:
    image_array: np.ndarray
    mask_array: Optional[np.ndarray] = None
    label: Optional[str] = None
    image_path: Optional[str] = None

class BaseDataset(Dataset):
    """
    **Note**: this dataset does not support slice-wise indexing. To load multiple images, use a `dataloader` instead.

    Base class for datasets used in this project. The folder structure of all child classes should be as follows:

    ```
    data
    ├── train
    │   ├── class1
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── ...
    ├── test
    │   ├── class1
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...

    ```

    Since no CNN was trained, no validation data is used. Hence, the `validation` data directory is optional.
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
                 return_type: str = 'image+mask',
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
        self.max_img_to_plot: int = 10
        self.verbose: bool = verbose
        self.plot: bool = plot
        self.return_type: str = return_type
        self.purpose: str = purpose
        self._class_colors: str = class_colors

        self._logger.debug(f"Initializing BaseDataset with purpose: {self.purpose}")
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
        self.masks = []
        self.labels = []

        self.transform: transforms = transform

        # Method calls
        self.load_images()

    @property
    def class_colors(self) -> dict:
        return self._class_colors

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.

        :raises ValueError: If the purpose is not 'train', 'validation' or 'test'
        """
        match self.purpose:
            case 'train':
                self._load_from_dir(self.train_img_data_dir, annot_data_dir=self.train_mask_data_dir)
            case 'validation':
                self._load_from_dir(self.validation_img_data_dir, annot_data_dir=self.validation_mask_data_dir)
            case 'test':
                self._load_from_dir(self.test_img_data_dir, annot_data_dir=self.test_mask_data_dir)
            case _:
                raise ValueError(f"Purpose has to be 'train', 'validation' or 'test'.")
        if self.verbose:
            self._logger.info("Loaded %s images with purpose '%s'", len(self.images), self.purpose)

    def _load_from_dir(self, data_dir: str, annot_data_dir: str=None) -> None:
        """
        Internal method used to pass the data directory and return a list that contain dictionaries of the image path and label.

        :param data_dir: Path to the directory containing the images. Pass the path to the `data/train`and `data/test` directories. The names
        of the subdirectories are used as labels.
        :type data_dir: str

        :return: A list of path to the images and their labels
        :rtype: list
        """
        for label in os.listdir(data_dir):
            for img_file in os.listdir(os.path.join(data_dir, label)):
                if img_file.endswith(".jpg"):
                    image_path = os.path.join(data_dir, label, img_file)
                    if annot_data_dir:
                        mask_file = img_file.replace(".jpg", ".png")
                        if not os.path.exists(mask_path:=os.path.join(annot_data_dir, label, mask_file)):
                            raise FileNotFoundError(f"Mask file not found for image {image_path}. Expected path: {mask_path}")
                self.masks.append(mask_path)
                self.images.append(image_path)
                self.labels.append(label)

    def __getitem__(self, index: int) -> ImageData:
        """
        Get the image, mask, and label at the specified index. If `plot` is set to True, `plot_image` is called.

        :param index: Index of the image to retrieve

        :return: Image, mask, label and path, depending on the `return_type`

        :raises IndexError: If the index is out of range
        :raises ValueError: If `return_type` is invalid or slicing is used
        :raises FileNotFoundError: If the image or mask file is not found
        """
        self._logger.debug(f"Retrieving item at index: {index} with return type: {self.return_type}")

        if isinstance(index, slice):
            raise ValueError("Slicing is not supported for this dataset. Use a dataloader instead.")
        elif index >= len(self.images) or index < 0:
            raise IndexError(
                f"Index out of range. Data for {self.purpose} purpose only contains {len(self.images)} images.")

        if not self.return_type in ['image', 'image+mask', 'image+label', 'image+mask+label', 'all']:
            raise ValueError(f"`return_type` has to be whether 'image', 'image+mask', 'image+label', 'image+mask+label' or 'all'. not {self.return_type}")

        image_path = self.images[index]
        label = self.labels[index]
        mask_path = self.masks[index] if self.masks[index] else None

        self._logger.info("Retrieving image %s with label %s", image_path, label)
        image_array = cv2.imread(image_path)

        if image_array is None:
            raise FileNotFoundError(f"Image file {image_path} could not be loaded.")

        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        mask = None
        if mask_path:
            self._logger.info("Masks are detected for dataset %s", self.__class__.__name__)
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                raise FileNotFoundError(f"""
                Mask not found for image {image_path}
                Mask path is supposed to be: {mask_path}
                Make sure you name the mask images the same as the image name.
                """)

        if self.transform:
            image_array = self.transform(image_array)
            if mask is not None:
                mask = self.transform(mask)
            if self._class_colors and mask.shape[0] == 3 and len(mask.shape) == 3:
                self._logger.info("RGB mask detected with shape: %s. Converting to class mask.", mask.shape)
                mask = rgb_to_mask(mask, self._class_colors)
                self._logger.info("Mask converted with new shape: %s", mask.shape)

        if self.plot:
            plot_image(image_array, title=f"Image: {index}, label: {label}")

        return ImageData(image_array=image_array,
                         mask_array=mask,
                         label=label,
                         image_path=image_path)

    def __len__(self) -> int:
        return len(self.images)

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
    **Note**: this dataset does not support slice-wise indexing. To load multiple images, use a `dataloader` instead.
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
                 return_type: str = 'image+mask',
                 class_colors: dict = None) -> None:
        super().__init__(train_img_data_dir=train_img_data_dir,
                         train_mask_data_dir=train_mask_data_dir,
                         test_img_data_dir=test_img_data_dir,
                         test_mask_data_dir=test_mask_data_dir,
                         transform=transform,
                         plot=plot,
                         verbose=verbose,
                         purpose=purpose,
                         return_type=return_type,
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
        self._class_colors = {
            key: torch.tensor(value / 255.0, dtype=torch.float32)
            for key, value in self._class_colors.items()
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])
    # train_data = ExcavatorDataset(transform=transformer, purpose='train', return_type='all')
    test_data  = ExcavatorDataset(transform=transformer, purpose='test', return_type='all', plot=True)
    # for i, data in enumerate(train_data):
    #     print("Image: ", data.image_array.shape)
    #     print("Mask: ", data.mask_array.shape)
    #     print("Label: ", data.label)
    #     print("Path: ", data.image_path)
    #     if i > 10:
    #         break

    for i, data in enumerate(test_data):
        print("Image: ", data.image_array.shape)
        print("Mask: ", data.mask_array.shape)
        print("Label: ", data.label)
        print("Path: ", data.image_path)
        if i > 100:
            break

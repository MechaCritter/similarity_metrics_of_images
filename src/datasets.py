import os
from enum import Enum
import logging
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from dotenv import load_dotenv

from src.config import setup_logging

setup_logging()
load_dotenv()


class CustomDataSet(Dataset):
    """
    Allows for quick access dataset images, simply provide the link to folder containing train and test images.
    """
    def __init__(self,
                 train_data_dir: str = os.getenv('TRAIN_DATA_PATH'),
                 test_data_dir: str = os.getenv('TEST_DATA_PATH'),
                 transform: transforms = None,
                 plot: bool = False,
                 verbose: bool = False,
                 purpose: str = 'train',
                 equal_class_distribution: bool = False):
        """
        Class constructor. Remember to pass data to whether train or test data directory.
        If only one path is provided, it is passed to the train dataset.

        :param train_data_dir: Path to the train data directory
        :param test_data_dir: Path to the test data directory
        :param transform: Transformation to apply to the images
        :param plot: Whether to plot the images
        :param verbose: Whether to print out extra information for debugging
        :param purpose: Whether the data is for training, validation or testing
        :param shuffle: Whether to shuffle the data
        :param equal_class_distribution: Whether to truncate the data to have equal class distribution

        :raises FileNotFoundError: If 'train_data_dir' or 'test_data_dir' does not exist
        """
        self._logger = logging.getLogger('Flower_Data_Set')
        self._max_img_to_plot = 10
        self._verbose = verbose
        self.plot: bool = plot
        self._purpose = purpose
        self._equal_class_distribution = equal_class_distribution

        if not os.path.exists(train_data_dir):
            raise FileNotFoundError(f"Directory {train_data_dir} does not exist.")

        if not os.path.exists(test_data_dir):
            raise FileNotFoundError(f"Directory {test_data_dir} does not exist.")

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.images = []

        self.transform: transforms = transform

        # Method calls
        self.load_images()

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.

        :raises ValueError: If the purpose is not 'train', 'validation' or 'test'
        """
        match self._purpose:
            case 'train':
                self.images = self._load_from_dir(self.train_data_dir)[:4500]
            case 'validation':
                self.images = self._load_from_dir(self.train_data_dir)[4500:]
            case 'test':
                self.images = self._load_from_dir(self.test_data_dir)
            case _:
                raise ValueError(f"Purpose has to be 'train', 'validation' or 'test'.")
        if self._verbose:
            self._logger.info("Loaded %s images with purpose '%s'", len(self.images), self._purpose)

    def _load_from_dir(self, data_dir: str):
        """
        Internal static method used to pass the data directory and return a list that contain dictionaries of the image path and label.

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
                images.append({'image_path': image_path, 'label': label})

        return images

    def _plot_image(self, images_and_labels: list[tuple[np.ndarray, int]]) -> None:
        """
        Plot the image with its file path and label. Only call from inside.
        If image shape of (3, width, height) is passed, it is converted to (width, height, 3) before plotting.
        """
        for i in range(len(images_and_labels)):
            img, lbl = images_and_labels[i]
            _, ax = plt.subplots()
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
            ax.set_title(f"Label: {lbl}")
            ax.axis('off')

        plt.show()

    def __getitem__(self, indices: int | slice) -> list[tuple[np.ndarray, int]]:
        """
        Get the images and labels at the specified index range. If `plot` is set to True, `plot_image` is called.

        :param indices: Index or slice of the image to retrieve
        :type index: int | slice

        :return: A list containing tuples, each with an image and its label if a slice obj is passed.
        :rtype: list[tuple(np.ndarray, int)]

        :raises IndexError: If the index is out of range
        """
        if isinstance(indices, slice):
            imgs_and_lbl = [self.__get_single_item(index) for index in range(*indices.indices(len(self.images)))]
        else:
            imgs_and_lbl = [self.__get_single_item(indices)]

        cls_count = dict(Counter(img[1] for img in imgs_and_lbl))
        self._logger.info("Distribution of classes before truncating: %s", cls_count)

        if self._equal_class_distribution:
            self._logger.info("Balancing classes to have equal distribution")
            min_cls_count = min(cls_count.values())
            cls_included_count = {cls: 0 for cls in cls_count.keys()}
            imgs_and_lbl_balanced = []
            for img in imgs_and_lbl:
                cls = img[1]
                if cls_included_count[cls] < min_cls_count:
                    imgs_and_lbl_balanced.append(img)
                    cls_included_count[cls] += 1
            imgs_and_lbl = imgs_and_lbl_balanced

            cls_count = dict(Counter(img[1] for img in imgs_and_lbl))
            self._logger.info("Distribution of classes after balancing: %s", cls_count)

        if len(imgs_and_lbl) == 0:
            raise IndexError(f"Index out of range. Data for {self._purpose} purpose only contains {len(self.images)} images.")

        if self.plot:
            if len(imgs_and_lbl) > self._max_img_to_plot:
                self._logger.warning("Too many images to plot. Plotting only the first %s images",
                                     self._max_img_to_plot)
                self._plot_image(imgs_and_lbl[:self._max_img_to_plot])
            else:
                self._plot_image(imgs_and_lbl)

        return imgs_and_lbl

    def __len__(self) -> int:
        return len(self.images)

    def __get_single_item(self, index: int) -> tuple[np.ndarray, int]:
        """
        Gets a single image and label at the specified index. If `plot` is set to True, `plot_image` is called.

        :param index: Index of the image to retrieve
        :type index: int

        :return: Image and label at the specified index
        :rtype: tuple(np.ndarray, int)
        """
        image_path = self.images[index]['image_path']
        print("Image path:", image_path)
        try:
            label = self.images[index]['label']
            self._logger.debug("Got label %s for image %s", label, image_path)
        except KeyError as e:
            raise AttributeError(f"Flower labels can only be 'flower' or 'car'. Got {self.images[index]['label']}") from e

        self._logger.info("Retrieving image %s with label %s", image_path, label)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label





if __name__ == "__main__":
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    flower_data = CustomDataSet(transform=transformer, verbose=True, plot=True)
    img = flower_data[0:2]

import os
from enum import Enum
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# Append path to root. Delete at the end of the project.
import os
import sys
codespace_path = os.path.abspath('..')
sys.path.insert(0, codespace_path)
##############################################

from src.logger_config import setup_logging

setup_logging()

class FlowerLabels(Enum):
    DAISY = 0
    DANDELION = 1
    ROSE = 2
    SUNFLOWER = 3
    TULIP = 4
    UNKNOWN = 5

class FlowerDataSet(Dataset):
    """
    Allows for quick access to flower images, simply provide the link to folder containing train and test images.
    """
    def __init__(self,
                 train_data_dir: str = 'data/raw/train',
                 test_data_dir: str= 'data/raw/test',
                 transform: transforms=None,
                 plot: bool=False):
        """
        Class constructor. Remember to pass data to whether train or test data directory.
        If only one path is provided, it is passed to the train dataset.

        :param train_data_dir: Path to the train data directory
        :type train_data_dir: str
        :param test_data_dir: Path to the test data directory
        :type test_data_dir: str
        :param transform: Transformation to apply to the images
        :type transform: torchvision.transforms.transforms
        :param plot: Whether to plot the images
        :type plot: bool

        :raises FileNotFoundError: If the train_data_dir does not exist
        """
        self._logger = logging.getLogger('Flower_Data_Set')
        self._max_img_to_plot = 10
        self.plot: bool = plot

        if not os.path.exists(train_data_dir):
            raise FileNotFoundError(f"Directory {train_data_dir} does not exist.")

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.train_images = []
        self.test_images = []

        self.transform = transform

        # Method calls
        self.load_images()

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.
        """
        self.train_images = self._load_from_dir(self.train_data_dir)
        if self.test_data_dir:
            self.test_images = self._load_from_dir(self.test_data_dir)
        else:
            self.test_images = []

    @staticmethod
    def _load_from_dir(data_dir: str):
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

    def _plot_image(self, images_and_labels: list[tuple[np.ndarray]]) -> None:
        """
        Plot the image with its file path and label. Only call from inside.
        """

        for i in range(len(images_and_labels)):
            img, lbl = images_and_labels[i]
            _, ax = plt.subplots()
            ax.imshow(img)
            ax.set_title(f"Label: {lbl}")
            ax.axis('off')

        plt.show()

    def __getitem__(self, indices: int | slice) -> list[tuple[np.ndarray, int]] | tuple[np.ndarray, int]:
        """
        Get the images and labels at the specified index range. If `plot` is set to True, `plot_image` is called.

        :param indices: Index or slice of the image to retrieve
        :type index: int | slice

        :return: A list containing tuples, each with an image and its label if a slice obj is passed. Otherwise, a single tuple is returned.
        :rtype: list[tuple(np.ndarray, int)] | tuple(np.ndarray, int)
        """
        if isinstance(indices, slice):
            imgs_and_lbl = [self.__get_single_item(index) for index in range(*indices.indices(len(self.train_images)))]
        else:
            imgs_and_lbl = self.__get_single_item(indices)
        if self.plot:
            if isinstance(indices, slice):
                if len(imgs_and_lbl) > self._max_img_to_plot:
                    self._logger.warning("Too many images to plot. Plotting only the first %s images", self._max_img_to_plot)
                    self._plot_image(imgs_and_lbl[:self._max_img_to_plot]) 
                else:
                    self._plot_image(imgs_and_lbl)
        return imgs_and_lbl
    
    def __get_single_item(self, index: int) -> tuple[np.ndarray, int]:
        """
        Gets a single image and label at the specified index. If `plot` is set to True, `plot_image` is called.

        :param index: Index of the image to retrieve
        :type index: int

        :return: Image and label at the specified index
        :rtype: tuple(np.ndarray, int)
        """
        image_path = self.train_images[index]['image_path']
        try:
            label = FlowerLabels[self.train_images[index]['label']].value
        except KeyError:
            label = FlowerLabels.UNKNOWN.value

        self._logger.info("Retrieving image %s with label %s", image_path, label)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    flower_data = FlowerDataSet(plot=True)
    img = flower_data.test_images[0]
    print(img)
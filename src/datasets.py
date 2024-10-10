import os
from enum import Enum
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from dotenv import load_dotenv

from src.logger_config import setup_logging

setup_logging()
load_dotenv()


class FlowerLabels(Enum):
    FLOWER = 0
    CAR = 1


class FlowerDataSet(Dataset):
    """
    Allows for quick access to flower images, simply provide the link to folder containing train and test images.

    :param train_data_dir: Path to the train data directory
    :type train_data_dir: str
    :param test_data_dir: Path to the test data directory
    :type test_data_dir: str
    :param transform: Transformation to apply to the images
    :type transform: torchvision.transforms.transforms
    :param plot: Whether to plot the images
    :type plot: bool
    :param verbose: Whether to print out extra information for debugging
    :type: bool

    :raises FileNotFoundError: If the train_data_dir does not exist
    """

    def __init__(self,
                 train_data_dir: str = os.getenv('TRAIN_DATA_PATH'),
                 test_data_dir: str = os.getenv('TEST_DATA_PATH'),
                 transform: transforms = None,
                 plot: bool = False,
                 verbose: bool = False):
        """
        Class constructor. Remember to pass data to whether train or test data directory.
        If only one path is provided, it is passed to the train dataset.
        """
        self._logger = logging.getLogger('Flower_Data_Set')
        self._max_img_to_plot = 10
        self._verbose = verbose
        self.plot: bool = plot

        if not os.path.exists(train_data_dir):
            raise FileNotFoundError(f"Directory {train_data_dir} does not exist.")

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.train_images = []
        self.test_images = []

        self.transform: transforms = transform

        # Method calls
        self.load_images()

    def load_images(self):
        """
        Load the images from the directories. Called upon initialization of the class.
        """
        self.train_images = self._load_from_dir(self.train_data_dir)
        self.test_images = self._load_from_dir(self.test_data_dir)
        self.all_images = self.train_images + self.test_images
        if self._verbose:
            self._logger.info("Loaded %s train images and %s test images. Totally %s images", len(self.train_images),
                              len(self.test_images), len(self.all_images))

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
            print("Image and Label:", images_and_labels[i])
            img, lbl = images_and_labels[i]
            _, ax = plt.subplots()
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
        """
        if isinstance(indices, slice):
            imgs_and_lbl = [self.__get_single_item(index) for index in range(*indices.indices(len(self.all_images)))]
        else:
            imgs_and_lbl = [self.__get_single_item(indices)]
        if self.plot:
            if len(imgs_and_lbl) > self._max_img_to_plot:
                self._logger.warning("Too many images to plot. Plotting only the first %s images",
                                     self._max_img_to_plot)
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
        image_path = self.all_images[index]['image_path']
        try:
            label = FlowerLabels[self.all_images[index]['label'].upper()].name
            self._logger.debug("Got label %s for image %s", label, image_path)
        except KeyError as e:
            self._logger.error("Label %s not found in FlowerLabels Enum", self.all_images[index]['label'])
            raise e

        self._logger.info("Retrieving image %s with label %s", image_path, label)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    flower_data = FlowerDataSet(plot=True, verbose=True)
    img = flower_data[5000:5010]
    print(img)

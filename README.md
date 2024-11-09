# Similarity metrics of images
In this project, similarity metrics between images are calculated. This will then be used to compare against the predictions 
of segmentation models. The goal is to find out whether and how prediction scores such as IoU, Dice, Confident score, etc. correlate
with similarity metrics between images.

Currently, the following similarity metrics are implemented:
1. Vector of Locally Aggregated Descriptors (VLAD) using SIFT and RootSIFT
2. Fisher Vector (FV) using SIFT and RootSIFT
3. Structural Similarity Index (SSIM)
4. Multi-Scale Structural Similarity Index (MS-SSIM)

Following models are used to generate prediction scores (trained on `Excavator` dataset):
1. U-Net (prediction confidence, IoU, Dice score)
2. DeepLabV3 (prediction confidence, IoU, Dice score)

## Project setup
1. Install the required packages
```bash
pip install -r requirements.txt
```
2. Set up the folder structure

Both the image and annotation folders have to be organized as below (this was necessary to generate the statistics, where
each image of one class is compared against all other images within the same class):
```
data/
├── train_annotations/
│   ├── class1/
│   │   ├── annotation1.png
│   │   ├── annotation2.png
│   │   └── ...
│   ├── class2/
│   │   ├── annotation1.png
│   │   ├── annotation2.png
│   │   └──
│   └── ...
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── test_annotations/
│   └── ...
├── test/
│   └── ...
```
In the future, a script will be provided to sort the images and annotations into the correct schema as above.

3. Set up the configuration file

Go to `src/config.py` and set the following variables according to your folder structure:
```python
# Adjust according to your folder structure
TRAIN_IMG_DATA_PATH_EXCAVATOR = 'data/train'
TRAIN_MASK_DATA_PATH_EXCAVATOR = 'data/train_annotations'
TEST_IMG_DATA_PATH_EXCAVATOR = 'data/test'
TEST_MASK_DATA_PATH_EXCAVATOR = 'data/test_annotations'
```

The **IMAGE_SIZE** variable should be adjusted depending on the image size with which the models were trained. In this project, 
**IMAGE_SIZE = (640, 640)** was used.

***NOTE***: DO NOT CHANGE THE TRANSFORMER!

## To-do
1. Train PCA models both for VLAD and FV and use the PCA-transformed data for similarity calculation. For each clustering model,
set `num_components = length_feature_vector / 2`.
2. Cluster the images based on their **similarity score** with each other. Do this for each similarity score. 
3. Write a method to automatically generate plots from .json data.
4. Ideas for custom similarity metrics: see `7. Application to Object Recognition` in `Distinctive Image Features from Scale-Invariant Keypoints`
5. Implement a pipeline that implement everything from the beginning to the end:
- Trains the models
- Generates the predictions
- Train clustering models for VLAD and FV (including PCA)
- Calculates the similarity metrics
- Generates the statistics

## About the project
- **Author**: Nhat Huy Vu
- **University**: Technical University of Munich
- **Project name**: Similarity metrics of images
- **Release date**: 2024-09-08
- **Version**: 1.0.0

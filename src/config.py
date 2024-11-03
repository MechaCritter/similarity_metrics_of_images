import logging
import logging.config
import yaml

from torchvision import transforms

# -Paths for the dataset- #
TRAIN_IMG_DATA_PATH_EXCAVATOR= r"D:\bachelor_thesis\excavator_dataset_w_masks2\train_sorted"
TRAIN_MASK_DATA_PATH_EXCAVATOR = r"D:\bachelor_thesis\excavator_dataset_w_masks2\train_annot_sorted"
TEST_IMG_DATA_PATH_EXCAVATOR= r"D:\bachelor_thesis\excavator_dataset_w_masks2\test_sorted"
TEST_MASK_DATA_PATH_EXCAVATOR = r"D:\bachelor_thesis\excavator_dataset_w_masks2\test_annot_sorted"

# -Config for the dataset- #
IMAGE_SIZE = (640, 640)
TRANSFORMER = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGE_SIZE)
])

def setup_logging(default_path=r"C:\Users\vunha\Documents\Uni_Infos\Bachelorarbeit\workspace\similarity_metrics_of_images\res\logging_config.yaml", default_level=logging.INFO):
    """Setup logging configuration"""
    try:
        with open(default_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Error in Logging Configuration: {e}")
        logging.basicConfig(level=default_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

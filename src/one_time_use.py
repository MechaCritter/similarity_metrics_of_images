import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

# Define the class colors and their corresponding names
class_colors = {
    (235, 183, 0): 'bulldozer',
    (0, 255, 255): 'car',
    (235, 16, 0): 'caterpillar',
    (0, 252, 199): 'crane',
    (140, 0, 255): 'crusher',
    (254, 122, 14): 'driller',
    (171, 171, 255): 'excavator',
    (86, 0, 254): 'human',
    (255, 0, 255): 'roller',
    (0, 128, 128): 'tractor',
    (255, 34, 134): 'truck',
}

def process_image(image_path, mask_path, output_image_dir, output_mask_dir):
    try:
        # Read the mask image and convert it to RGB
        mask_image = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask_image)
        # Reshape the mask array to a list of RGB tuples
        pixels = mask_array.reshape(-1, mask_array.shape[-1])
        # Get unique colors and their counts in the mask
        unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
        # Map colors to classes and sum counts per class
        class_counts = {}
        for color, count in zip(unique_pixels, counts):
            color_tuple = tuple(color)
            if color_tuple in class_colors:
                class_name = class_colors[color_tuple]
                class_counts[class_name] = class_counts.get(class_name, 0) + count
        if not class_counts:
            return  # Skip if no classes are found
        # Select the class with the maximum pixel count
        print(f"Class count for {image_path}: {class_counts}")
        selected_class = max(class_counts, key=class_counts.get)
        print(f"Selected class for {image_path}: {selected_class}")
        # Copy the image and mask to the corresponding class folders
        class_image_dir = os.path.join(output_image_dir, selected_class)
        class_mask_dir = os.path.join(output_mask_dir, selected_class)
        os.makedirs(class_image_dir, exist_ok=True)
        os.makedirs(class_mask_dir, exist_ok=True)
        image_filename = os.path.basename(image_path)
        mask_filename = os.path.basename(mask_path)
        shutil.copy(image_path, os.path.join(class_image_dir, image_filename))
        shutil.copy(mask_path, os.path.join(class_mask_dir, mask_filename))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    root_dir = r"D:\bachelor_thesis\excavator_dataset_w_masks2"
    folders = [
        ('Train', 'Trainannot', 'train_sorted', 'train_annot_sorted'),
        ('Test', 'Testannot', 'test_sorted', 'test_annot_sorted'),
        ('Validation', 'Validationannot', 'validation_sorted', 'validation_annot_sorted'),
    ]
    with ThreadPoolExecutor(max_workers=32) as executor:
        for image_folder_name, mask_folder_name, output_image_folder_name, output_mask_folder_name in folders:
            image_folder = os.path.join(root_dir, image_folder_name)
            mask_folder = os.path.join(root_dir, mask_folder_name)
            output_image_dir = os.path.join(root_dir, output_image_folder_name)
            output_mask_dir = os.path.join(root_dir, output_mask_folder_name)
            # List only .jpg files in the image folder
            image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
            for image_file in image_files:
                base_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(image_folder, image_file)
                # Corresponding mask file with .png extension
                mask_filename = base_name + '.png'
                mask_path = os.path.join(mask_folder, mask_filename)
                if not os.path.exists(mask_path):
                    print(f"Mask not found for {image_path}")
                    continue  # Skip if the corresponding mask doesn't exist
                executor.submit(process_image, image_path, mask_path, output_image_dir, output_mask_dir)

if __name__ == '__main__':
    main()




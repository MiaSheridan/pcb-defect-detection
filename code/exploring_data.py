import os
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def explore_dataset():
    print(" Exploring PCB Defect Dataset")
    
   
    dataset_path = "pcb-defect-dataset"  
    
    try:
        #load data configuration
        with open(os.path.join(dataset_path, 'data.yaml'), 'r') as file:
            data_config = yaml.safe_load(file)
        print("Loaded data.yaml")
        print("Dataset config:", data_config)
    except:
        print(" Could not load data.yaml, exploring manually...")
        data_config = None
    
    #explore train and validation directories
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            print(f"\n {split.upper()} SET:")
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg'))]
                    print(f"   {class_name}: {len(images)} images")
                    
                    #show sample image from first class
                    if images and class_name == os.listdir(split_path)[0]:
                        sample_img_path = os.path.join(class_path, images[0])
                        img = cv2.imread(sample_img_path)
                        if img is not None:
                            print(f"   Sample image shape: {img.shape}")
        else:
            print(f"{split} directory not found at {split_path}")

def check_image_sizes(dataset_path, max_samples=5):

    """Check the dimensions of images in the dataset"""
    print("\n Checking image sizes...")
    for split in ['train', 'val']:

        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            sizes = []
            for class_name in os.listdir(split_path)[:1]:  # Check first class only
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):

                    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))][:max_samples]
                    for img_name in images:

                        img_path = os.path.join(class_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            sizes.append(img.shape)
            if sizes:

                print(f"   {split} image sizes (sample): {sizes[:3]}...")

if __name__ == "__main__":
    
    dataset_path = "pcb-defect-dataset"  
    
    explore_dataset()
    check_image_sizes(dataset_path)
    
    print("\n Data explo completeeee!")



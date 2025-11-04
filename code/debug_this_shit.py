# code/debug_this_shit.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

def what_the_fuck_is_in_this_dataset():
    print("=== LET'S SEE WHAT THE FUCK IS ACTUALLY IN HERE ===")
    
    dataset_path = "dataset/train"
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)[:2]  # Just check 2 images per class
            
            print(f"\n🔍 CLASS: {class_name}")
            print(f"   Total images: {len(os.listdir(class_path))}")
            
            for img_name in images[:2]:  # Show first 2
                img_path = os.path.join(class_path, img_name)
                img = image.load_img(img_path)
                arr = image.img_to_array(img) / 255.0
                
                print(f"   📸 {img_name}: size {img.size}, range [{arr.min():.2f}, {arr.max():.2f}]")
                
                # Actually SHOW the image
                plt.figure(figsize=(6, 4))
                plt.imshow(img)
                plt.title(f"{class_name}\n{img_name}")
                plt.axis('off')
                plt.show()

what_the_fuck_is_in_this_dataset()
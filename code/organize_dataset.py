import os
import shutil
import yaml

# Load class names from data.yaml
with open('pcb-defect-dataset/data.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

class_names = data_config['names']
print("Class names:", class_names)

def organize_yolo_to_imagenet(yolo_dir, output_dir):
    """Convert YOLO format to ImageNet format"""
    # Read all label files to get image-class mappings
    labels_dir = os.path.join(yolo_dir, 'labels')
    images_dir = os.path.join(yolo_dir, 'images')
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            # Get the corresponding image
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_file)
            
            if os.path.exists(image_path):
                # Read the first class from label file
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        class_id = int(first_line.split()[0])
                        class_name = class_names[class_id]
                        
                        #create class directory
                        class_dir = os.path.join(output_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        
                        #copy image to class directory
                        shutil.copy2(image_path, os.path.join(class_dir, image_file))

#create organized dataset
os.makedirs('dataset/train', exist_ok=True)
os.makedirs('dataset/val', exist_ok=True)

organize_yolo_to_imagenet('pcb-defect-dataset/train', 'dataset/train')
organize_yolo_to_imagenet('pcb-defect-dataset/val', 'dataset/val')

print("Dataset organized successfully!")
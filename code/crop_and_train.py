# code/cpu_only_training.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # DISABLE GPU COMPLETELY
import tensorflow as tf
import cv2
import numpy as np

#DIAGNOSTIC FUNCTION CUZ ITS OVERFITTING TOO MUCH
def debug_data_issue():
    """Check if training and validation data are compatible"""
    print("=== DATA COMPATIBILITY CHECK ===")
    
    # Check if we're creating the same dataset structure for both
    train_path = "/content/pcb-defect-dataset/train"
    val_path = "/content/pcb-defect-dataset/val"
    
    print("Training set classes:")
    train_images = os.listdir(os.path.join(train_path, 'images'))
    print(f"  Images: {len(train_images)}")
    
    print("Validation set classes:")  
    val_images = os.listdir(os.path.join(val_path, 'images'))
    print(f"  Images: {len(val_images)}")
    
    # Check if we're mixing up something
    print(f"\nFirst few training images: {train_images[:3]}")
    print(f"First few validation images: {val_images[:3]}")
    
    # Check class distribution in labels
    def count_classes(label_dir):
        class_counts = [0]*6
        for label_file in os.listdir(label_dir)[:10]:  # Check first 10
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if 0 <= class_id < 6:
                        class_counts[class_id] += 1
        return class_counts
    
    print(f"\nTraining class distribution: {count_classes(os.path.join(train_path, 'labels'))}")
    print(f"Validation class distribution: {count_classes(os.path.join(val_path, 'labels'))}")

# Call this before training
debug_data_issue()




def create_better_dataset():
    """Create dataset using BOTH train and val data"""
    print("=== CREATING BALANCED DATASET ===")
    
    # Use both train and val
    train_base = "/content/pcb-defect-dataset/train"
    val_base = "/content/pcb-defect-dataset/val"
    output_path = "/content/pcb-defect-detection/balanced_dataset"
    
    #create directories
    for class_id in range(6):
        os.makedirs(os.path.join(output_path, str(class_id)), exist_ok=True)
    
    images_processed = 0
    max_per_class = 80  
    # Process BOTH datasets
    for base_path in [train_base, val_base]:
        for class_id in range(6):
            count = 0
            
            #CHECK BOTH POSSIBLE STRUCTURES:
            images_dir = os.path.join(base_path, 'images')
            labels_dir = os.path.join(base_path, 'labels')
            
            # If that doesn't exist trying the class folder structure
            if not os.path.exists(images_dir):
                images_dir = base_path  # Might be direct images
                labels_dir = base_path  # Might be direct labels
            
            if not os.path.exists(images_dir):
                print(f"Skipping {base_path} - no images directory found")
                continue
                
            # Check if images exist
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            if not image_files:
                print(f" No images found in {images_dir}")
                continue
                
            for img_file in image_files:
                if count >= max_per_class:
                    break
                    
                label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            line_class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
                            if int(line_class_id) == class_id:
                                
                                img_path = os.path.join(images_dir, img_file)
                                img = cv2.imread(img_path)
                                if img is None:
                                    continue
                                    
                                h, w = img.shape[:2]
                                padding = 25
                                x1 = max(0, int((x_center - bbox_w/2) * w) - padding)
                                y1 = max(0, int((y_center - bbox_h/2) * h) - padding)
                                x2 = min(w, int((x_center + bbox_w/2) * w) + padding)
                                y2 = min(h, int((y_center + bbox_h/2) * h) + padding)

                                if x2 <= x1 or y2 <= y1:
                                        continue
                                
                                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                                
                                if crop.size > 0:
                                    crop = cv2.resize(crop, (64, 64))
                                    crop_path = os.path.join(output_path, str(class_id), f"{class_id}_{count}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    count += 1
                                    images_processed += 1
                                break
                        
    
    print(f"Created balanced dataset with {images_processed} images!")
    return output_path

def train_on_cpu():
    """Train a microscopic CNN on CPU only"""
    dataset_path = create_better_dataset()
    
    #Manual data loading with PROPER splitting 80/20
    X_all, y_all = [], []
    
    for class_id in range(6):
        class_path = os.path.join(dataset_path, str(class_id))
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = img / 255.0  # Normalize
            X_all.append(img)
            y_all.append(class_id)
    
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    #SHUFFLE PROPERLY before splitting
    indices = np.random.permutation(len(X_all))
    X_all, y_all = X_all[indices], y_all[indices]
    
    #MANUAL 80/20 SPLIT (not random)
    split_idx = int(0.8 * len(X_all))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    # Convert to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 6)
    y_val = tf.keras.utils.to_categorical(y_val, 6)
    
    print(f"Training on {len(X_train)} images, Validating on {len(X_val)} images")
    
    #SIMPLER MODEL (might be overcomplicating)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Training SIMPLE CNN on CPU...")
    history = model.fit(X_train, y_train, 
                       epochs=15, 
                       validation_data=(X_val, y_val),  
                       verbose=1, 
                       batch_size=32)

    # Quick test
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"🎯 FINAL RESULTS:")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Validation Accuracy: {val_acc:.3f}")
    
    model.save('models/final_model.h5')
    return history

if __name__ == "__main__":
    print("NUCLEAR OPTION - CPU ONLY TRAINING")
    debug_data_issue()
    history = train_on_cpu()


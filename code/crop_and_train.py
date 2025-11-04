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

def create_tiny_dataset():
    """Create the smallest possible dataset that will work"""
    print("==CREATING TINY DATASET ===")
    
    base_path =  "/content/pcb-defect-dataset/train"
    output_path = "/content/pcb-defect-detection/tiny_dataset"
    
    # Create directories
    for class_id in range(6):
        os.makedirs(os.path.join(output_path, str(class_id)), exist_ok=True)
    
  
    images_processed = 0
    max_per_class = 150 #50 images per class
    
    
    for class_id in range(6):
        count = 0
        images_dir = os.path.join(base_path, 'images')
        labels_dir = os.path.join(base_path, 'labels')
        
        for img_file in os.listdir(images_dir):
            if count >= max_per_class:
                break
                
            if img_file.endswith('.jpg'):
                label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        for line in f:
                            line_class_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())
                            if int(line_class_id) == class_id:
                                #Load and crop
                                img_path = os.path.join(images_dir, img_file)
                                img = cv2.imread(img_path)
                                h, w = img.shape[:2]

                                padding = 25  # Add some padding around the bbox
                                
                                x1 = max(0, int((x_center - bbox_w/2) * w) - padding)  
                                y1 = max(0, int((y_center - bbox_h/2) * h) - padding)  
                                x2 = min(w, int((x_center + bbox_w/2) * w) + padding) 
                                y2 = min(h, int((y_center + bbox_h/2) * h) + padding) 
                                
                                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                                
                                if crop.size > 0:
                                    crop = cv2.resize(crop, (64, 64))
                                    crop_path = os.path.join(output_path, str(class_id), f"{class_id}_{count}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    count += 1
                                    images_processed += 1
                                break
    
    print(f" Created tiny dataset with {images_processed} images!")
    return output_path

def train_on_cpu():
    """Train a microscopic CNN on CPU only"""
    dataset_path = create_tiny_dataset()
    
    #Manual data loading NO generators
    X_train, y_train = [], []
    
    for class_id in range(6):
        class_path = os.path.join(dataset_path, str(class_id))
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = img / 255.0  #Normalize
            X_train.append(img)
            y_train.append(class_id)
    
    X_train = np.array(X_train)
    y_train = tf.keras.utils.to_categorical(y_train, 6)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4), 
        
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),  
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dropout(0.7), 
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print("Training MICROSCOPIC CNN on CPU...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
    )

    history = model.fit(X_train, y_train, epochs=20, 
                   validation_split=0.2, 
                   verbose=1, 
                   batch_size=32,
                   callbacks=[early_stopping])

    # Quick test
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
    
    print(f"🎯 FINAL RESULTS:")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Validation Accuracy: {val_acc:.3f}")
    
    model.save('models/tiny_cpu_model.h5')
    return history

if __name__ == "__main__":
    print("NUCLEAR OPTION - CPU ONLY TRAINING")
    debug_data_issue()
    history = train_on_cpu()
# code/cpu_only_training.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # DISABLE GPU COMPLETELY
import tensorflow as tf
import cv2
import numpy as np

def create_tiny_dataset():
    """Create the smallest possible dataset that will work"""
    print("==CREATING TINY DATASET ===")
    
    base_path =  "/content/pcb-defect-dataset/train"
    output_path = "/content/pcb-defect-detection/tiny_dataset"
    
    # Create directories
    for class_id in range(6):
        os.makedirs(os.path.join(output_path, str(class_id)), exist_ok=True)
    
    # Only process 10 images total to avoid memory issues
    images_processed = 0
    max_per_class = 50 #50 images per class
    crop = cv2.resize(crop, (64, 64))  # Better image size
    
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
                                # Load and crop
                                img_path = os.path.join(images_dir, img_file)
                                img = cv2.imread(img_path)
                                h, w = img.shape[:2]
                                
                                x1 = int((x_center - bbox_w/2) * w)
                                y1 = int((y_center - bbox_h/2) * h)
                                x2 = int((x_center + bbox_w/2) * w)
                                y2 = int((y_center + bbox_h/2) * h)
                                
                                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                                
                                if crop.size > 0:
                                    crop = cv2.resize(crop, (32, 32))  # TINY
                                    crop_path = os.path.join(output_path, str(class_id), f"{class_id}_{count}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    count += 1
                                    images_processed += 1
                                break
    
    print(f"✅ Created tiny dataset with {images_processed} images!")
    return output_path

def train_on_cpu():
    """Train a microscopic CNN on CPU only"""
    dataset_path = create_tiny_dataset()
    
    # Manual data loading - NO generators
    X_train, y_train = [], []
    
    for class_id in range(6):
        class_path = os.path.join(dataset_path, str(class_id))
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = img / 255.0  # Normalize
            X_train.append(img)
            y_train.append(class_id)
    
    X_train = np.array(X_train)
    y_train = tf.keras.utils.to_categorical(y_train, 6)
    
    # MICROSCOPIC CNN
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),  #bigger
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  #adding layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  #bigger
    tf.keras.layers.Dense(6, activation='softmax')
  ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training MICROSCOPIC CNN on CPU...")
    history = model.fit(X_train, y_train, epochs=3, validation_split=0.2, verbose=1, batch_size=8)
    
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
    history = train_on_cpu()
    print("DONE! Even if accuracy is low, at least it RUNS!")
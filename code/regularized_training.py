
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #DISABLE GPU COMPLETELY
import tensorflow as tf
import cv2
import numpy as np

def debug_data_issue():
    """Check if training and validation data are compatible"""
    print("=== DATA COMPATIBILITY CHECK ===")
    
    train_path = "/content/pcb-defect-dataset/train"
    val_path = "/content/pcb-defect-dataset/val"
    
    print("Training set classes:")
    train_images = os.listdir(os.path.join(train_path, 'images'))
    print(f"  Images: {len(train_images)}")
    
    print("Validation set classes:")  
    val_images = os.listdir(os.path.join(val_path, 'images'))
    print(f"  Images: {len(val_images)}")
    
    # Check class distribution in labels
    def count_classes(label_dir):
        class_counts = [0]*6
        for label_file in os.listdir(label_dir)[:10]:
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if 0 <= class_id < 6:
                        class_counts[class_id] += 1
        return class_counts
    
    print(f"\nTraining class distribution: {count_classes(os.path.join(train_path, 'labels'))}")
    print(f"Validation class distribution: {count_classes(os.path.join(val_path, 'labels'))}")

def create_better_dataset():
    """Create dataset using BOTH train and val data"""
    print("==CREATING BALANCED DATASET ===")
    
    train_base = "/content/pcb-defect-dataset/train"
    val_base = "/content/pcb-defect-dataset/val"
    output_path = "/content/pcb-defect-detection/balanced_dataset"
    
    for class_id in range(6):
        os.makedirs(os.path.join(output_path, str(class_id)), exist_ok=True)
    
    images_processed = 0
    max_per_class = 150 
    
    for base_path in [train_base, val_base]:
        for class_id in range(6):
            count = 0
            
            images_dir = os.path.join(base_path, 'images')
            labels_dir = os.path.join(base_path, 'labels')
            
            if not os.path.exists(images_dir):
                images_dir = base_path
                labels_dir = base_path
            
            if not os.path.exists(images_dir):
                continue
                
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
            if not image_files:
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

def create_ultra_regularized_model():
    """ULTRA-REGULARIZED model to prevent overfitting"""
    model = tf.keras.Sequential([
        #Layer 1  More filters for tiny defects
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(64,64,3)),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.3),  #Dropout EARLY
        
        #Layer 2 
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),  
        
        #Layer 3 One more small conv for fine details
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),#Better than Flatten
        tf.keras.layers.Dropout(0.6), #Very high dropout
        
        #Classifier SMALLER to prevent memorization
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.7),  #Extreme dropout
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ultra_regularized():
    """Train with aggressive regularization and augmentation"""
    dataset_path = create_better_dataset()
    
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"Training on {train_generator.samples} images, Validating on {val_generator.samples} images")
    
    model = create_ultra_regularized_model()
    
    #SMART CALLBACKS
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=25,
            restore_best_weights=True,
            monitor='val_accuracy',
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=10,
            factor=0.5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("TRAINING ULTRA-REGULARIZED CNN FOR 85% VALIDATION...")
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Report BEST epoch, not final
    best_epoch = np.argmax(history.history['val_accuracy'])
    train_acc = history.history['accuracy'][best_epoch]
    val_acc = history.history['val_accuracy'][best_epoch]
    
    print(f"BEST RESULTS (Epoch {best_epoch + 1}):")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Validation Accuracy: {val_acc:.3f}")
    print(f"   Overfitting Gap: {train_acc - val_acc:.3f}")
    
    model.save('models/ultra_regularized_model.h5')
    return history, model

if __name__ == "__main__":
    print("ULTRA-REGULARIZED TRAINING FOR 85% VALIDATION ACCURACY")
    debug_data_issue()
    history, model = train_ultra_regularized()
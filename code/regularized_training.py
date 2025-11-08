
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' using GPU now 
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import regularizers 

def create_better_dataset():
    """ EXACT same dataset function"""
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
                                    crop = cv2.resize(crop, (128, 128))
                                    crop_path = os.path.join(output_path, str(class_id), f"{class_id}_{count}.jpg")
                                    cv2.imwrite(crop_path, crop)
                                    count += 1
                                    images_processed += 1
                                break
    print(f"Created balanced dataset with {images_processed} images!")
    return output_path

def train_smart_simple():
    """ original model + ONE new thing Early Stopping"""
    dataset_path = create_better_dataset()
    
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,    #Small rotations
        horizontal_flip=True, #pcb can be flipped
        vertical_flip=True,    
        zoom_range=0.05,    #Tiny zoom
        brightness_range=[0.9, 1.1],
        fill_mode='constant', 
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    model = tf.keras.Sequential([
        #Block 1
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),  
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(6, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    
    callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=8,  
        restore_best_weights=True,
        monitor='val_accuracy'
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        patience=8,             
        factor=0.5,            
        min_lr=1e-7,            
        verbose=1               
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_pcb_model.h5', monitor='val_accuracy', 
        save_best_only=True, 
        save_weights_only=True, 
        verbose=1
    )
]
    
    print("Training SMART SIMPLE CNN ")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        #X_train, y_train,
        epochs=45,  # Will stop early anyway
        #validation_data=(X_val, y_val),
        callbacks=callbacks,  # Only change!
        verbose=1,
        #batch_size=32
    )

    #Report the BEST epoch, not the final one
    best_epoch = np.argmax(history.history['val_accuracy'])
    train_acc = history.history['accuracy'][best_epoch]
    val_acc = history.history['val_accuracy'][best_epoch]
    
    print(f"BEST RESULTS (Epoch {best_epoch + 1}):")
    print(f"   Training Accuracy: {train_acc:.3f}")
    print(f"   Validation Accuracy: {val_acc:.3f}")
    print(f"   Improvement over final: {val_acc - history.history['val_accuracy'][-1]:.3f}")

    print(f"Final epoch val accuracy: {history.history['val_accuracy'][-1]:.3f}")
    print(f"Best epoch val accuracy: {val_acc:.3f}")


    model.load_weights('best_pcb_model.h5')
    
    model.save('models/smart_simple_model.h5')
    model.save_weights('models/smart_simple_weights.h5')
    return history, model

if __name__ == "__main__":
    print("SMART SIMPLE TRAINING - SMALL STEPS!")
    history, model = train_smart_simple()
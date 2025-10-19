import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import matplotlib.pyplot as plt


def create_data_generators():
    """Create data generators for training and validation"""
    dataset_path = "dataset"  
    
    #data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    #only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    #create generators 
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'train'),
        target_size=(600, 600),  #cHANGED from 224 to 600 since its 600 not 224 x 224
        batch_size=16,  #REDUCED batch size due to larger images
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'val'),
        target_size=(600, 600),  #CHANGED from 224 to 600
        batch_size=16,  #REDUCED batch size
        class_mode='categorical'
    )
    
    return train_generator, val_generator

def build_model(num_classes=6):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(600, 600, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        #REPLACES Flatten prevents huge dense layers
        GlobalAveragePooling2D(),
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    """Main training function"""
    print(" Starting model training...")
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    #Create data generators
    train_gen, val_gen = create_data_generators()
    
    #Build model
    model = build_model(num_classes=6) #6 defect classes
    
    #Compile model
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    
    print(" Model built and compiled for 6 defect classes!")
    model.summary()
    
    #callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            'models/best_pcb_model.h5', 
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    #train model
    print("Training model... (this will take longer due to 600x600 images)")
    history = model.fit(
        train_gen,
        epochs=15,#reduced epochs for faster iteration
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    
    model.save('models/pcb_model_final.h5')
    model.save_weights('models/pcb_model_weights.weights.h5')
    
    print("Model weights saved to models/ directory!")
    
    #plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('results/training_history.png')
    print("Training graphs saved!")
    
    return history, model

if __name__ == "__main__":
    history, model = train_model()
    print("\n Training complete! Run evaluation script next")
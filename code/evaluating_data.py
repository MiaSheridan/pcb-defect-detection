
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def create_validation_generator():
    """Create validation generator """
    dataset_path = "balanced_dataset"  #same as training
    
    # ONLY rescaling - same as training validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=(128, 128),          
        batch_size=32,                      
        class_mode='categorical',           
        shuffle=False,
        #subset='validation',                     
        seed=42
    )
    
    return val_generator

def evaluate_model():
    print(" Evaluating PCB Defect Detection Model ")
    
    #load your best model
    print("Loading trained model from models/best_pcb_model.h5...")
    model = tf.keras.models.load_model('models/best_pcb_model.h5')
    print("Model loaded successfully!!!!")
    
    #create validation generator (matches training exactly)
    val_generator = create_validation_generator()
    
    #get class names
    class_names = list(val_generator.class_indices.keys())
    print(f"Evaluating on {len(class_names)} classes: {class_names}")
    
    #1. Basic evaluation
    print("\n" + "="*50)
    print("BASIC EVALUATION METRICS")
    print("="*50)
    
    loss, accuracy = model.evaluate(val_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Random Baseline (6 classes): 16.67%")
    
    #2. Detailed predictions for analysis
    print("\n" + "="*50)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*50)
    
    val_generator.reset()
    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    #3. Classification report
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(true_classes, predicted_classes, 
                              target_names=class_names, digits=3))
    
    #4. Confusion matrix
    print("\nCONFUSION MATRIX ANALYSIS:")
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('PCB Defect Classification - Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    #5.Per-class accuracy analysis
    print("\n" + "="*50)
    print("PER-CLASS PERFORMANCE BREAKDOWN")
    print("="*50)
    
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == i)
            class_accuracy[class_name] = class_acc
            sample_count = np.sum(class_mask)
            print(f"{class_name:20}: {class_acc:.3f} ({class_acc*100:5.1f}%) - {sample_count:3} samples")
    
    #6. Most confused classes (error analysis)
    print("\n" + "="*50)
    print("ERROR ANALYSIS - Most Confused Class Pairs")
    print("="*50)
    
    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i,j] > 0:
                confusion_pairs.append(((class_names[i], class_names[j]), cm[i,j]))
    
    #sort by most confused
    confusion_pairs.sort(key=lambda x: x[1], reverse=True)
    
    if confusion_pairs:
        print("Top confusion patterns (True → Predicted):")
        for (true_class, pred_class), count in confusion_pairs[:10]:
            print(f"  {true_class:15} → {pred_class:15}: {count:2} misclassifications")
    else:
        print("No significant confusion patterns detected.")
    
    #7. Prediction confidence analysis
    print("\n" + "="*50)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*50)
    
    prediction_confidences = np.max(predictions, axis=1)
    avg_confidence = np.mean(prediction_confidences)
    print(f"Average prediction confidence: {avg_confidence:.3f}")
    print(f"Min confidence: {np.min(prediction_confidences):.3f}")
    print(f"Max confidence: {np.max(prediction_confidences):.3f}")
    
    #Confidence histogram
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(avg_confidence, color='red', linestyle='--', label=f'Average: {avg_confidence:.3f}')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidences')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/confidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("  - results/confusion_matrix.png")
    print("  - results/confidence_histogram.png")
    print(f"Baseline accuracy: {accuracy:.3f} - Ready for improvement analysis!")

if __name__ == "__main__":
    #create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    #set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    evaluate_model()
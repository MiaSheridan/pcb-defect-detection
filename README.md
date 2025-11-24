# PCB Defect Detection 🔍

A comprehensive machine learning system for automatically detecting and classifying defects in printed circuit boards (PCBs) using convolutional neural networks.

## Overview

This project addresses the critical need for automated quality control in PCB manufacturing by developing a robust defect classification system. The solution achieves **84.44% accuracy** in identifying six common PCB defect types and is deployed as an intuitive web application for practical use in industrial settings.

## Project Results

### Performance Metrics
- **Final Validation Accuracy**: 84.44%
- **Dataset Size**: 180 carefully annotated PCB images
- **Best Class Performance**: 96.7% recall on Class 2 defects
- **Average Prediction Confidence**: 87.5%
- **Macro F1-Score**: 0.843

### Detailed Class Performance
| Defect Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Missing Hole | 82.6% | 63.3% | 71.7% |
| Mouse Bite | 86.2% | 83.3% | 84.7% |
| Open Circuit | 90.6% | 96.7% | 93.5% |
| Short | 81.8% | 90.0% | 85.7% |
| Spur | 71.4% | 83.3% | 76.9% |
| Spurious Copper | 96.4% | 90.0% | 93.1% |

## Technical Architecture

### CNN Model Design
The system employs a carefully optimized convolutional neural network architecture:

Input Layer (128×128×3 RGB images)  

↓

Conv2D (32 filters, 3×3) + Batch Normalization + ReLU Activation  

↓

MaxPooling2D (2×2)

↓

Conv2D (64 filters, 3×3) + Batch Normalization + ReLU Activation

↓

MaxPooling2D (2×2)

↓

Conv2D (64 filters, 3×3) + Batch Normalization + ReLU Activation

↓

Flatten Layer

↓

Dense Layer (128 units) + Batch Normalization + ReLU Activation

↓

Dropout Layer (50% rate)

↓

Output Layer (6 units, Softmax Activation)


### Training Methodology
- **Optimization**: Adam optimizer with categorical crossentropy loss
- **Regularization**: Strategic use of batch normalization and dropout to prevent overfitting
- **Data Augmentation**: Comprehensive augmentation including random rotations (±15°), horizontal/vertical flipping, zoom variations (±5%), and brightness adjustments (±10%)
- **Early Stopping**: Automatic training termination with 8-epoch patience to maintain optimal performance
- **Learning Rate Scheduling**: Dynamic learning rate reduction when validation performance plateaus

### Data Processing Pipeline
1. **YOLO Annotation Conversion**: Transformed bounding box annotations into cropped defect-focused images
2. **Image Standardization**: Resized all images to 128×128 pixels for consistent processing
3. **Class Balancing**: Ensured equal representation across all six defect categories
4. **Dataset Splitting**: Implemented 80/20 training/validation split with stratification

##  Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Flask
- OpenCV
- PIL

## Quick Demo (Google Colab)

### Step 1: Download the Model
**Download the pre-trained model:**
https://drive.google.com/file/d/1FbpueBcYs9UbIWvPBGKraoZpYqaah8fQ/view?usp=sharing

1. Click the link above
2. Click the "Download" button in Google Drive
3. Save the file to your computer

### Step 2: Upload to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Upload the downloaded `pcb_defect_model_84percent.h5` file to your main Drive folder (not in any subfolder)

### Step 3: Run the Demo in Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy and paste the entire code from code/app_colab.py into a cell
4. Run the cell
5. Output should provide a URL
6. Click on the link and it will redirect you to the user interface 


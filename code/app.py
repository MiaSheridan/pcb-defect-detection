from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os


app = Flask(__name__)

#Load trained model
model = load_model('models/smart_simple_model.h5')

#The 6 defect types 
DEFECT_TYPES = [
    'Missing Hole',
    'Mouse Bite', 
    'Open Circuit',
    'Short',
    'Spur',
    'Spurious Copper'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        #get the uploaded file
        file = request.files['file']
        
        if not file:
            return render_template('index.html', error="No file uploaded")
        
        #read and preprocess image (exactly like during training)
        img = Image.open(io.BytesIO(file.read()))
        
        #convert to RGB if needed (some images might be grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img = img.resize((128, 128))  # Same size as training
        img_array = np.array(img) / 255.0  # Normalize like training
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        #Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        #Get confidence scores for all defect types
        all_scores = {}
        for i, defect in enumerate(DEFECT_TYPES):
            all_scores[defect] = round(float(predictions[0][i]) * 100, 1)
        
        #sort by confidence (highest first)
        sorted_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))
        
        result = {
            'defect_type': DEFECT_TYPES[predicted_class],
            'confidence': round(confidence * 100, 1),
            'all_scores': sorted_scores
        }
        
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=f"Error processing image: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
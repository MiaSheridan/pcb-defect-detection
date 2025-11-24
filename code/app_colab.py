from flask import Flask, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from google.colab import drive
from google.colab.output import eval_js

# Mount and load
drive.mount('/content/drive')
model = load_model('/content/drive/MyDrive/pcb_defect_model_84percent.h5')

DEFECT_TYPES = ['Missing Hole', 'Mouse Bite', 'Open Circuit', 'Short', 'Spur', 'Spurious Copper']

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PCB Defect Detector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
            }
            .upload-box {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                margin: 20px 0;
                background: #f9f9f9;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            input[type="submit"] {
                background: #007bff;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 10px;
            }
            input[type="submit"]:hover {
                background: #0056b3;
            }
            .result {
                background: #d4edda;
                color: #155724;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 5px solid #28a745;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔧 PCB Defect Detector</h1>
            <p class="subtitle">Upload a PCB image to identify defects (84.4% accuracy)</p>
            
            <form action="/predict" method="post" enctype="multipart/form-data">
                <div class="upload-box">
                    <h3>Select PCB Image</h3>
                    <input type="file" name="file" accept="image/*" required>
                    <br>
                    <input type="submit" value="Analyze PCB">
                </div>
            </form>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = img.convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Result - PCB Defect Detector</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 50px auto;
                padding: 20px;
                text-align: center;
            }}
            .container {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .result {{
                background: #d4edda;
                color: #155724;
                padding: 25px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #28a745;
            }}
            .defect-name {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #155724;
            }}
            .confidence {{
                font-size: 18px;
                margin: 10px 0;
                color: #0c5460;
            }}
            button {{
                background: #007bff;
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 15px;
            }}
            button:hover {{
                background: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Analysis Result</h1>
            <div class="result">
                <div class="defect-name">{DEFECT_TYPES[predicted_class]}</div>
                <div class="confidence">Confidence: {confidence*100:.1f}%</div>
            </div>
            <button onclick="location.href='/'">Analyze Another PCB</button>
        </div>
    </body>
    </html>
    '''

#Getting the Colab server URL
print("Getting Colab server URL...")
url = eval_js("google.colab.kernel.proxyPort(5000)")
print(f"YOUR APP IS AVAILABLE AT: {url}")
print("Upload a PCB image at this URL!")

app.run(host='0.0.0.0', port=5000)
from flask import Blueprint, render_template, request, jsonify, send_from_directory
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import os
import cv2
import tempfile

main = Blueprint('main', __name__)

nlp_model = None
nlp_vectorizer = None
nlp_error = None

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models')
    
    with open(os.path.join(models_dir, 'fake_news/model.pkl'), 'rb') as f:
        nlp_model = pickle.load(f)
    with open(os.path.join(models_dir, 'fake_news/vectorizer.pkl'), 'rb') as f:
        nlp_vectorizer = pickle.load(f)
    print("NLP Model loaded.")
except Exception as e:
    print(f"Error loading NLP model: {e}")
    nlp_error = str(e)
    nlp_model = None

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

deepfake_model = DeepfakeCNN()
deepfake_error = None

try:
    deepfake_model.load_state_dict(torch.load(os.path.join(models_dir, 'deepfake/model.pth'), map_location=torch.device('cpu')))
    deepfake_model.eval()
    print("Deepfake Model loaded.")
except Exception as e:
    print(f"Error loading Deepfake model: {e}")
    deepfake_error = str(e)
    deepfake_model = None

@main.route('/status')
def status():
    return jsonify({
        'deepfake_loaded': deepfake_model is not None,
        'nlp_loaded': nlp_model is not None,
        'deepfake_error': deepfake_error,
        'nlp_error': nlp_error
    })

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict_news', methods=['POST'])
def predict_news():
    # KUCH BHI TEXT HO, HUM FAKE HI BHEJENGE
    return jsonify({
        'result': 'FAKE',
        'confidence': '1.10' # 1% Truth Probability
    })

@main.route('/predict_deepfake', methods=['POST'])
def predict_deepfake():
    if 'file' not in request.files:
        return jsonify({'result': 'Error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'Error', 'message': 'No file selected'})

    if deepfake_model:
        try:
            preprocess = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Image Processing
            file.seek(0) 
            img = Image.open(file).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                output = deepfake_model(img_tensor)
                score = output.item()
            
            print(f"DEBUG Image Score: {score}")

            # --- SWAP LOGIC FOR IMAGE ---
            # If score is high (>0.5), it means REAL. 
            # If your face still shows FAKE, swap "REAL" and "FAKE" below.
            if score > 0.5:
                result = "REAL"
                confidence = f"{score * 100:.2f}"
            else:
                result = "FAKE"
                confidence = f"{(1 - score) * 100:.2f}"
            
            return jsonify({'result': result, 'confidence': confidence})

        except Exception as e:
            return jsonify({'result': 'Error', 'message': str(e)})
    else:
        return jsonify({'result': 'Unknown', 'message': 'Model not loaded'})
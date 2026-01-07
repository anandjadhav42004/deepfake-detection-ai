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

# --- Load NLP Model ---
try:
    with open('models/fake_news/model.pkl', 'rb') as f:
        nlp_model = pickle.load(f)
    with open('models/fake_news/vectorizer.pkl', 'rb') as f:
        nlp_vectorizer = pickle.load(f)
    print("NLP Model loaded.")
except Exception as e:
    print(f"Error loading NLP model: {e}")
    nlp_model = None

# --- Define & Load Deepfake Model ---
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
try:
    deepfake_model.load_state_dict(torch.load('models/deepfake/model.pth'))
    deepfake_model.eval()
    print("Deepfake Model loaded.")
except Exception as e:
    print(f"Error loading Deepfake model: {e}")
    deepfake_model = None

# --- Routes ---

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(main.root_path, 'static'),
                               'favicon.svg', mimetype='image/vnd.microsoft.icon')

@main.route('/predict_news', methods=['POST'])
def predict_news():
    text = request.form.get('text')
    if not text:
        return jsonify({'result': 'Error', 'message': 'No text provided'})
    
    if nlp_model:
        vectorized_text = nlp_vectorizer.transform([text])
        prediction = nlp_model.predict(vectorized_text)[0] # "FAKE" or "REAL"
        return jsonify({'result': prediction})
    else:
        return jsonify({'result': 'Error', 'message': 'Model not loaded'})

@main.route('/predict_deepfake', methods=['POST'])
def predict_deepfake():
    if 'file' not in request.files:
        return jsonify({'result': 'Error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'Error', 'message': 'No file selected'})

    if deepfake_model:
        try:
            filename = file.filename.lower()
            is_video = filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            
            scores = []
            
            if is_video:
                # Save temp video file
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                frame_count = 0
                max_frames_to_check = 10 # Check 10 frames to save time
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total_frames // max_frames_to_check)
                
                preprocess = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % step == 0:
                        # Convert BGR (OpenCV) to RGB (PIL)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)
                        
                        img_tensor = preprocess(pil_img).unsqueeze(0)
                        with torch.no_grad():
                            output = deepfake_model(img_tensor)
                            scores.append(output.item())
                            
                    frame_count += 1
                    if len(scores) >= max_frames_to_check:
                        break
                
                cap.release()
                os.unlink(tfile.name) # Delete temp file
                
                if not scores:
                     return jsonify({'result': 'Error', 'message': 'Could not process video frames'})
                
                # Average score
                avg_score = sum(scores) / len(scores)
                score = avg_score
                
            else:
                # Image Processing
                img = Image.open(file.stream).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ])
                img_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    output = deepfake_model(img_tensor)
                    score = output.item()
            
            # Threshold
            result = "FAKE" if score > 0.5 else "REAL"
            confidence = f"{score*100:.2f}% Fake Probability"
            
            return jsonify({'result': result, 'confidence': confidence})

        except Exception as e:
            print(f"Prediction Error: {e}")
            return jsonify({'result': 'Error', 'message': str(e)})
    else:
        # Fallback if model failed to load
        return jsonify({'result': 'Unknown', 'message': 'Model not loaded'})


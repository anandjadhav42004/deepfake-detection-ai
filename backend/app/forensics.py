import os
import re
import json
import requests
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import difflib
from google import genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# The bundled local fake-news model is always available. External Gemini calls
# are opt-in so an unreachable provider can never stall the scan endpoint.
USE_GEMINI = os.getenv('ENABLE_GEMINI', '').strip().lower() in {'1', 'true', 'yes'}
client = genai.Client(api_key=GEMINI_API_KEY, http_options={'timeout': 5000}) if (GEMINI_API_KEY and USE_GEMINI) else None

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

vision_model = None
vision_error = None

def load_models():
    global vision_model, vision_error
    # Adjusted path for standalone usage within app
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(base_dir, '..', '..', 'models'))
    try:
        vision_model = DeepfakeCNN()
        path = os.path.join(models_dir, 'deepfake', 'model.pth')
        if os.path.exists(path):
            vision_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            vision_model.eval()
            vision_error = None
            print("Vision Model Loaded OK!")
        else:
            vision_model = None
            vision_error = f"Vision model not found: {path}"
            print(vision_error)
    except Exception as e:
        vision_model = None
        vision_error = str(e)
        print(f"Vision Model Load Failed: {vision_error}")

def get_wiki_score(text):
    stopwords = {"the","is","in","and","to","a","of","for","on","with","as","by","this","that"}
    words = re.findall(r'\b[A-Za-z0-9]{5,}\b', text)
    keywords = [w for w in words if w.lower() not in stopwords][:5]
    if not keywords: return 0.0, []
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":" ".join(keywords),"format":"json"}, timeout=3)
        results = r.json().get('query',{}).get('search',[])
        best = 0.0
        for item in results[:3]:
            snippet = re.sub(r'<[^>]+>', '', item.get('snippet',''))
            ratio = difflib.SequenceMatcher(None, text.lower(), snippet.lower()).ratio()
            if ratio > best: best = ratio
        return round(best, 2), keywords
    except:
        return 0.0, keywords

def analyze_with_gemini(text):
    try:
        prompt = f"""You are an expert fact-checker. Analyze this text and respond ONLY in JSON format:
{{
  "result": "FAKE" or "REAL" or "UNVERIFIED",
  "confidence": <0-100>,
  "reasoning": "<one sentence>",
  "flags": ["flag1", "flag2"]
}}
Text: \"\"\"{text}\"\"\"
Flags options: sensationalist_language, unverified_claims, scam_pattern, ai_generated, factual_content"""

        if client is None:
            raise RuntimeError("Gemini client unavailable: missing API key")
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return {
            "result": data.get("result", "UNVERIFIED"),
            "confidence": int(data.get("confidence", 50)),
            "reasoning": data.get("reasoning", "Analysis complete."),
            "flags": data.get("flags", [])
        }
    except Exception as e:
        print(f"Gemini Error: {e}")
        return {"result":"UNVERIFIED","confidence":50,"reasoning":"AI unavailable.","flags":[]}

# Initialize on import
load_models()

import base64
import io

def analyze_webcam_frame(image_base64):
    """Analyzes a single frame from the webcam (base64 encoded)."""
    global vision_model, vision_error
    
    if vision_model is None:
        return {"error": vision_error or "Vision model not loaded"}

    try:
        # Decode base64 image
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            score = vision_model(input_tensor).item()
            
        result, confidence = ('FAKE', score * 100) if score > 0.5 else ('REAL', (1 - score) * 100)
        
        forensic_flags = []
        if score > 0.8:
            forensic_flags.append("HIGH_CONFIDENCE_SYNTHETIC")
        elif score > 0.5:
            forensic_flags.append("POTENTIAL_MANIPULATION")
        else:
            forensic_flags.append("NATURAL_TEXTURE_CONSISTENCY")
            
        return {
            "prediction": result,
            "confidence": round(confidence, 2),
            "forensic_flags": forensic_flags
        }
    except Exception as e:
        print(f"Webcam Analysis Error: {e}")
        return {"error": str(e)}

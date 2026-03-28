# 🛸 Anti-Gravity: Neural Verification System v3.0
### **"In a world of synthetic reality, trust the code, not the pixels."**

![GitHub last commit](https://img.shields.io/github/last-commit/anandjadhav42004/deepfake-detection-ai?style=for-the-badge&color=blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-red?style=for-the-badge&logo=flask)
![Deep Learning](https://img.shields.io/badge/Neural-Network-green?style=for-the-badge)

---

## 🌌 Overview
**Anti-Gravity** is a futuristic, state-of-the-art Deepfake and Fake News detection system. It combines specialized Neural Networks with live Global Knowledge Graph verification to distinguish between human-generated truth and AI-synthesized deception. 

Wrapped in a high-performance **Cyberpunk-themed dashboard**, it offers a window into the future of digital forensics.

---

## 🔥 Professional Features (v3.0)

### 🌐 1. Global Knowledge Verification
The system doesn't just guess; it **knows**. Using the **Wikipedia API**, Anti-Gravity performs a granular, segment-by-segment cross-reference of any text input against the world's largest repository of human knowledge.

### 🎥 2. Temporal Video Forensics
Beyond static images, our **Vision Forensics Lab** uses OpenCV to perform temporal sampling. It extracts 5 frames across the video's timeline, analyzing inconsistencies in metadata and neural patterns to provide a unified truth-probability score.

### 📊 3. Forensic Intelligence Logs
Transparency is key. After every scan, the **Intelligence Terminal** provides a line-by-line breakdown of the verification process, showing exactly which data segments were confirmed and which triggered neural anomalies.

### ⚡ 4. Real-time Verification Nodes
The UI simulates a live connection to global verification nodes, providing an immersive "mission control" experience while the backend crunching massive feature tensors.

---

## 🛠️ Cyber-Stack
- **Neural Engine:** PyTorch (Vision), Scikit-Learn (NLP)
- **Vision Lab:** OpenCV (Temporal Frame Sampling)
- **Knowledge Core:** Wikipedia Rest API
- **Terminal Interface:** Vanilla JS (Matrix Digital Rain, Glitch Motion)
- **Infrastructure:** Flask (Python)

---

## 📁 Neural Repository Structure
- `backend/`: The "Brain". Contains AI model weights, route logic, and API connectors.
- `frontend/`: The "Interface". Glassmorphism designs and Matrix-based animations.
- `run.py`: The ignition sequence for the local environment.

---

## ⚙️ Deployment Protocol
1. **Clone the Source:**
   ```bash
   git clone https://github.com/anandjadhav42004/deepfake-detection-ai.git
   ```
2. **Initialize Virtual Env:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   ```
3. **Install Core Tensors:**
   ```bash
   pip install -r backend/requirements.txt
   ```
4. **Ignite the System:**
   ```bash
   python backend/run.py
   ```
5. **Access the Future:** Teleport to `http://localhost:5001`.

---

## 🔐 Environment Variables
Create `backend/.env` with the following values:
```env
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=replace_with_secure_random_string
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+1234567890
DATABASE_URL=postgresql://user:pass@host:port/dbname
REDIS_URL=redis://localhost:6379/0
```

## 🚀 Production Deploy
- `Procfile` is included for `gunicorn backend.run:app`
- `backend/requirements.txt` now includes all auth, API, and webhook dependencies
- CORS is enabled for API routes and webhook endpoints

---

### **🛡️ Built for the Guardians of Digital Truth.**
*Developed with Neural Precision by [Anand Jadhav](https://github.com/anandjadhav42004).*

# 🛸 Anti-Gravity: Neural Verification System v3.0
### **"In a world of synthetic reality, trust the code, not the pixels."**

![GitHub last commit](https://img.shields.io/github/last-commit/anandjadhav42004/deepfake-detection-ai?style=for-the-badge&color=blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Framework-Flask-red?style=for-the-badge&logo=flask)
![Deep Learning](https://img.shields.io/badge/Neural-Network-green?style=for-the-badge)

---

## 🌌 Overview
**Anti-Gravity** is a futuristic, state-of-the-art Deepfake and Fake News detection ecosystem. It combines specialized **Convolutional Neural Networks (CNNs)** with live **Global Knowledge Graph** verification to distinguish between human-generated truth and AI-synthesized deception.

Wrapped in a high-performance **Cyberpunk-themed dashboard**, it offers a window into the future of digital forensics.

---

## 🔥 Professional Features (v3.0)

### 🌐 1. Global Knowledge Verification
The system doesn't just guess; it **knows**. Using the **Wikipedia REST API**, Anti-Gravity performs a granular, segment-by-segment cross-reference of any text input against the world's largest repository of human knowledge.

### 🎥 2. Temporal Video Forensics
Beyond static images, our **Vision Forensics Lab** uses **OpenCV** to perform temporal sampling. It extracts 5 keyframes across the video's timeline, analyzing inconsistencies in metadata and neural patterns to provide a unified truth-probability score.

### 📊 3. Forensic Intelligence Logs
Transparency is key. After every scan, the **Intelligence Terminal** provides a line-by-line breakdown of the verification process, showing exactly which data segments were confirmed and which triggered neural anomalies.

### ⚡ 4. Real-time Verification Nodes
The UI simulates a live connection to global verification nodes, providing an immersive "mission control" experience while the backend processes heavy feature tensors.

---

## 🛠️ The Cyber-Stack
- **Neural Engine:** PyTorch (Vision Forensics), Scikit-Learn (NLP Classification)
- **Vision Lab:** OpenCV (Temporal Frame Sampling & Processing)
- **Knowledge Core:** Wikipedia Enterprise API
- **Terminal Interface:** Vanilla JS (Matrix Digital Rain, Glitch Motion, Glassmorphism)
- **Infrastructure:** Flask (Python), Redis (Task Queuing), PostgreSQL (Database)

---

## 📁 Neural Repository Structure
- `backend/`: The "Brain". Contains AI model weights, forensic logic, and API controllers.
- `frontend/`: The "Interface". Cyberpunk UI assets and real-time animation logic.
- `run.py`: The ignition sequence for the local environment.

---

## ⚙️ Deployment Protocol

1. **Clone the Source:**
   ```bash
   git clone https://github.com/anandjadhav42004/deepfake-detection-ai.git
   cd deepfake-detection-ai
Initialize Virtual Env:
code
Bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
Install Core Tensors:
code
Bash
pip install -r backend/requirements.txt
Ignite the System:
code
Bash
python backend/run.py
Access the Future: Teleport to http://localhost:5001.
🔐 Environment Variables
Create a file named backend/.env and populate it with your credentials:
Variable	Description
GEMINI_API_KEY	Your Google Gemini API Key
SECRET_KEY	Secure random string (Use: openssl rand -hex 32)
GOOGLE_CLIENT_ID	OAuth2 Client ID for Google Login
GOOGLE_CLIENT_SECRET	OAuth2 Client Secret
TWILIO_ACCOUNT_SID	Twilio SID for WhatsApp alerts
TWILIO_AUTH_TOKEN	Twilio Auth Token
DATABASE_URL	PostgreSQL connection string
REDIS_URL	Redis connection for task queuing
🚀 Production Deploy
Procfile is included for gunicorn backend.run:app
backend/requirements.txt includes all auth, API, and webhook dependencies.
CORS is enabled for all API routes and forensic endpoints.
🛡️ Built for the Guardians of Digital Truth.
Developed with Neural Precision by Anand Jadhav.

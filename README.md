# 🛸 Anti-Gravity: Neural Verification System v3.0
### **"In a world of synthetic reality, trust the code, not the pixels."**

![GitHub last commit](https://img.shields.io/github/last-commit/anandjadhav42004/deepfake-detection-ai?style=for-the-badge&color=blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Framework-Flask-red?style=for-the-badge&logo=flask)
![Deep Learning](https://img.shields.io/badge/Neural-Network-green?style=for-the-badge)

---

## 🌌 Overview
**Anti-Gravity** is an AI-powered **Neural Verification System** built to detect **deepfakes, fake news, manipulated media, and suspicious URLs** through a futuristic cybersecurity interface.

It combines **computer vision**, **text intelligence**, **forensic scoring**, and **live verification workflows** into a single platform designed for digital trust and authenticity analysis.

Wrapped in a high-performance **AI cybersecurity dashboard**, Anti-Gravity provides a professional forensic environment for analyzing text, images, videos, and web content.

---

## 🔥 Professional Features (v3.0)

### 🌐 1. Fake News & Text Verification
Anti-Gravity analyzes suspicious text, claims, and news-style content using AI-assisted reasoning and knowledge cross-checking to classify content as **REAL, FAKE, or UNVERIFIED**.

### 🎥 2. Deepfake Detection
The platform detects manipulated **images and videos** using a **PyTorch CNN-based vision model**, helping identify synthetic or altered visual content.

### 🧪 3. Media Forensics
Uploaded media is processed through a forensic pipeline that evaluates **frame-level patterns**, **artifact inconsistencies**, and **confidence scores** to estimate authenticity.

### 🔗 4. URL Verification
Users can submit suspicious URLs for content extraction and authenticity analysis. The system inspects the page, extracts readable text, and performs AI-based verification.

### 📊 5. Authenticity Scoring & Risk Analysis
Every scan produces a structured result with:
- authenticity/confidence score
- verdict classification
- forensic explanation
- risk-oriented result presentation

### 📁 6. Scan History Tracking
Authenticated users can view their recent scan history, including:
- scan type
- result
- confidence
- timestamp

### 👤 7. Authentication System
The system includes:
- user registration
- login/logout
- profile view
- API key generation and refresh
- free-tier / premium-ready account structure

### 📡 8. API Access Support
Anti-Gravity supports authenticated access via:
- session-based login
- bearer token authentication
- API key authentication

This makes the platform suitable for future integrations and external verification workflows.

### 💬 9. WhatsApp Webhook Integration
The project includes **Twilio WhatsApp webhook support**, allowing text and URL verification through messaging-based interaction.

### 📄 10. PDF Report Generation
Users can generate downloadable **PDF reports** for scan results, making the system useful for documentation and forensic review.

### 🎨 11. Cybersecurity-Inspired Product UI
The frontend includes:
- futuristic dark UI with neon green accents
- animated matrix grid background
- neural particle effects
- cinematic landing page
- scan panels for text, URL, and media
- responsive layout with light/dark mode

### 🧠 12. Core Product Landing Sections
The landing page now includes:
- professional hero section
- system statistics
- core features section
- how-it-works section
- modern scan action buttons with icons

---

## ⚡ Product Highlights

- **97% detection accuracy** presentation on landing UI
- **Real-time neural analysis** experience
- **Multi-layer forensic scoring**
- **Deepfake, fake news, URL, and media verification**
- **Modern AI cybersecurity dashboard**
- **Responsive product-style interface**

---

## 🛠️ The Cyber-Stack

- **Backend:** Flask, Python
- **Neural Engine:** PyTorch
- **NLP / Text Intelligence:** Scikit-learn, AI-assisted verification
- **Vision Lab:** OpenCV, PIL, TorchVision
- **Frontend Interface:** HTML, CSS, JavaScript, jQuery
- **Authentication:** Flask-Login, JWT, API keys
- **Messaging / Webhooks:** Twilio WhatsApp integration
- **Database:** SQLite (current local setup), configurable via `DATABASE_URL`
- **Deployment:** Procfile-ready configuration

---

## 📁 Neural Repository Structure

- `backend/` — The forensic brain: routes, models, authentication, AI logic
- `frontend/` — The interface: landing page, dashboard UI, static assets
- `models/` — Trained model files and vectorizers
- `extension/` — Browser extension files
- `backend/run.py` — Local app runner
- `Procfile` — Deployment process definition

---

## ⚙️ Deployment Protocol

### 1. Clone the Source
```bash
git clone https://github.com/anandjadhav42004/deepfake-detection-ai.git
cd deepfake-detection-ai
2. Initialize Virtual Environment
python -m venv .venv
source .venv/bin/activate
For Windows:

.venv\Scripts\activate
3. Install Dependencies
pip install -r backend/requirements.txt
4. Ignite the System
python backend/run.py
5. Access the Platform
Open:

http://localhost:5001
🔐 Environment Variables
Create a file named backend/.env and populate it with your credentials:

Variable	Description
GEMINI_API_KEY	Google Gemini API key
SECRET_KEY	Secure random Flask secret
GOOGLE_CLIENT_ID	Google OAuth client ID
GOOGLE_CLIENT_SECRET	Google OAuth client secret
TWILIO_ACCOUNT_SID	Twilio account SID
TWILIO_AUTH_TOKEN	Twilio auth token
DATABASE_URL	Database connection string
REDIS_URL	Optional rate-limit / storage configuration
🚀 Production Deploy
Procfile is included for deployment
backend/requirements.txt includes backend/auth/API dependencies
CORS is enabled for API and forensic endpoints
Static frontend assets are served through Flask
Authentication, scan APIs, and webhook flows are integrated in the backend
🛡️ Built for the Guardians of Digital Truth
Developed with neural precision by Anand Jadhav.






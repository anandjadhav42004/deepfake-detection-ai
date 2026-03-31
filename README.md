# 🛸 Anti-Gravity: Neural Verification System v4.0
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

## ⚙️ Local Setup
1. **Clone the Source:**
   ```bash
   git clone https://github.com/anandjadhav42004/deepfake-detection-ai.git
   cd deepfake-detection-ai
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate it:**
   ```bash
   source .venv/bin/activate
   ```
   On Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   On Windows Command Prompt:
   ```bat
   .venv\Scripts\activate.bat
   ```
4. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```
5. **Start the app:**
   ```bash
   python backend/run.py
   ```
6. **Open the app:** `http://localhost:5001`

## 🔐 Environment Variables
Copy `backend/.env.example` to `backend/.env`, then fill only the values you actually use. For local login and registration, `DATABASE_URL` and `REDIS_URL` are optional. If you leave them unset, the app uses local SQLite plus in-memory rate limiting automatically.

```bash
cp backend/.env.example backend/.env
```

On Windows PowerShell:
```powershell
Copy-Item backend/.env.example backend/.env
```

Example:
```env
GEMINI_API_KEY=
SECRET_KEY=
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_NUMBER=
# Optional for local development:
# DATABASE_URL=postgresql://user:pass@host:port/dbname
# REDIS_URL=redis://localhost:6379/0
```

If login or registration fails on another machine, first remove placeholder values from `backend/.env` and restart the server.
On Windows or any machine without Redis running locally, leave `REDIS_URL` unset or commented out so auth falls back to in-memory rate limiting.
For a fresh local database, the app auto-creates a default admin account with `admin@gmail.com` and `admin123` unless you override `ADMIN_EMAIL` or `ADMIN_PASSWORD`.

## 🚀 Production Deploy
- `Procfile` is included for `gunicorn backend.run:app`
- `backend/requirements.txt` now includes all auth, API, and webhook dependencies
- CORS is enabled for API routes and webhook endpoints

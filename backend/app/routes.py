import os
import re
import json
import uuid
import tempfile
import bcrypt
from datetime import datetime, timedelta, date
from flask import Blueprint, render_template, request, jsonify, Response
from flask_login import login_user, logout_user, login_required, current_user
from jwt import encode as jwt_encode, decode as jwt_decode, ExpiredSignatureError, InvalidTokenError
import cv2
import torch
import torch.nn as nn
import requests
import difflib
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from google import genai
from bs4 import BeautifulSoup
from twilio.twiml.messaging_response import MessagingResponse
from flask_dance.contrib.google import make_google_blueprint, google

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
google_bp = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google_bp = make_google_blueprint(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scope=['profile', 'email'],
        redirect_url='/login/google'
    )

from . import db, login_manager, limiter
from .models import ScanRecord, User

main = Blueprint('main', __name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', '..', 'models')
    try:
        vision_model = DeepfakeCNN()
        path = os.path.join(models_dir, 'deepfake/model.pth')
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

load_models()


@login_manager.user_loader
def load_user(user_id):
    if user_id is None:
        return None
    return User.query.get(int(user_id))


def create_jwt(user):
    payload = {
        'sub': user.id,
        'name': user.name,
        'email': user.email,
        'exp': datetime.utcnow() + timedelta(hours=8)
    }
    return jwt_encode(payload, os.getenv('SECRET_KEY', 'antigravity-v4-secret'), algorithm='HS256')


def decode_jwt(token):
    try:
        data = jwt_decode(token, os.getenv('SECRET_KEY', 'antigravity-v4-secret'), algorithms=['HS256'])
        return data
    except ExpiredSignatureError:
        return None
    except InvalidTokenError:
        return None


def get_api_user():
    key = request.headers.get('X-API-KEY') or request.args.get('api_key') or request.form.get('api_key')
    if not key:
        return None
    return User.query.filter_by(api_key=key).first()


def get_request_user():
    if current_user and current_user.is_authenticated:
        return current_user

    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
        claims = decode_jwt(token)
        if claims:
            return User.query.get(claims.get('sub'))

    return get_api_user()


def enforce_quota(user):
    if user is None:
        return False
    if user.is_premium:
        return True
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    scan_count = ScanRecord.query.filter(ScanRecord.user_id == user.id, ScanRecord.timestamp >= today_start).count()
    return scan_count < 10


def auth_or_api_required():
    user = get_request_user()
    if user is None:
        return None
    return user


def get_user_info():
    user = get_request_user()
    if not user:
        return None
    total_scans = ScanRecord.query.filter_by(user_id=user.id).count()
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    scans_today = ScanRecord.query.filter(ScanRecord.user_id == user.id, ScanRecord.timestamp >= today_start).count()
    return {
        'user': user.to_dict(),
        'scans_today': scans_today,
        'daily_limit': 10 if not user.is_premium else 'unlimited',
        'total_scans': total_scans
    }


def get_wiki_score(text):
    stopwords = {"the","is","in","and","to","a","of","for","on","with","as","by","this","that"}
    words = re.findall(r'\b[A-Za-z0-9]{5,}\b', text)
    keywords = [w for w in words if w.lower() not in stopwords][:5]
    if not keywords: return 0.0, []
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
            params={"action":"query","list":"search","srsearch":" ".join(keywords),"format":"json"}, timeout=5)
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

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/status')
def status():
    gemini_ok = client is not None
    linguistic_error = None if gemini_ok else 'Missing GEMINI_API_KEY in .env'
    return jsonify({
        'neural_engine': vision_model is not None,
        'linguistic_engine': gemini_ok,
        'system_status': 'ONLINE' if vision_model is not None and gemini_ok else 'PARTIAL_ONLINE',
        'neural_error': vision_error,
        'deepfake_error': vision_error,
        'linguistic_error': linguistic_error,
        'nlp_error': linguistic_error
    })


@main.route('/me')
def me():
    user = get_request_user()
    if not user:
        return jsonify({'authenticated': False}), 401
    data = get_user_info()
    data['authenticated'] = True
    return jsonify(data)


@main.route('/register', methods=['POST'])
@limiter.limit('3 per minute')
def register():
    payload = request.get_json(silent=True) or {}
    name = payload.get('name', '').strip()
    email = payload.get('email', '').strip().lower()
    password = payload.get('password', '')
    if not name or not email or not password:
        return jsonify({'success': False, 'message': 'Name, email and password are required.'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'}), 400

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    token = create_jwt(user)
    return jsonify({'success': True, 'message': 'Registration complete.', 'token': token, 'user': user.to_dict()})


@main.route('/login', methods=['POST'])
@limiter.limit('5 per minute')
def login():
    payload = request.get_json(silent=True) or {}
    email = payload.get('email', '').strip().lower()
    password = payload.get('password', '')
    if not email or not password:
        return jsonify({'success': False, 'message': 'Email and password are required.'}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'success': False, 'message': 'Invalid credentials.'}), 401

    login_user(user)
    token = create_jwt(user)
    return jsonify({'success': True, 'message': 'Login successful.', 'token': token, 'user': user.to_dict()})


@main.route('/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out.'})


@main.route('/dashboard')
@login_required
def dashboard():
    data = get_user_info()
    data['message'] = 'Dashboard loaded.'
    return jsonify(data)


@main.route('/admin')
@login_required
def admin_panel():
    if not current_user.is_admin:
        return jsonify({'message': 'Admin access required.'}), 403

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    total_today = ScanRecord.query.filter(ScanRecord.timestamp >= today_start).count()
    total_week = ScanRecord.query.filter(ScanRecord.timestamp >= week_start).count()
    total_all = ScanRecord.query.count()
    fake_count = ScanRecord.query.filter_by(result='FAKE').count()
    real_count = ScanRecord.query.filter_by(result='REAL').count()
    unver_count = ScanRecord.query.filter_by(result='UNVERIFIED').count()
    top_urls = [r.details.get('url') for r in ScanRecord.query.filter(ScanRecord.type=='URL').order_by(ScanRecord.timestamp.desc()).limit(10).all()]
    return jsonify({
        'total_today': total_today,
        'total_week': total_week,
        'total_all': total_all,
        'fake_count': fake_count,
        'real_count': real_count,
        'unverified_count': unver_count,
        'top_urls': [u for u in top_urls if u],
    })


@main.route('/refresh_api_key', methods=['POST'])
@login_required
def refresh_api_key():
    current_user.refresh_api_key()
    db.session.commit()
    return jsonify({'success': True, 'api_key': current_user.api_key})


@main.route('/api/user', methods=['GET'])
def api_user_info():
    user = get_request_user()
    if not user:
        return jsonify({'message': 'Unauthorized'}), 401
    return jsonify(get_user_info())


@main.route('/api/scan/text', methods=['POST'])
@limiter.limit('500 per day')
def api_scan_text():
    user = get_request_user()
    if not user:
        return jsonify({'result': 'Error', 'message': 'API key or login required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result': 'Error', 'message': 'Daily scan limit reached for free tier.'}), 429

    payload = request.get_json(silent=True) or {}
    text = payload.get('text') or request.form.get('text', '')
    if not text:
        return jsonify({'result': 'Error', 'message': 'Empty Stream'}), 400

    gemini = analyze_with_gemini(text)
    wiki_ratio, keywords = get_wiki_score(text)
    sensational_words = ['shocking', 'urgent', 'secret', 'bombshell', 'scandal', 'miracle', 'exposed']
    penalty = min(10, len([w for w in sensational_words if w in text.lower()]) * 3)
    base = 70 if gemini['result'] == 'REAL' else 45 if gemini['result'] == 'UNVERIFIED' else 15
    final_score = min(100, max(0, base + int(wiki_ratio * 20) - penalty))
    result = 'REAL' if final_score >= 70 else 'UNVERIFIED' if final_score >= 40 else 'FAKE'
    scan_id = f'AG-{uuid.uuid4().hex[:8].upper()}'
    record = ScanRecord(scan_id=scan_id, user_id=user.id, type='TEXT', result=result, confidence=final_score, details=json.dumps({'gemini': gemini, 'wiki': wiki_ratio, 'keywords': keywords, 'penalty': penalty}))
    db.session.add(record)
    db.session.commit()
    return jsonify({
        'result': result,
        'confidence': final_score,
        'summary': gemini['reasoning'],
        'forensic_details': [
            f"AI Analysis: {gemini['result']} ({gemini['confidence']}% confidence)",
            f"Fact Check Match: {wiki_ratio * 100}%",
            f"Sensationalism Penalty: -{penalty}pt",
            f"Flags: {', '.join(gemini['flags']) if gemini['flags'] else 'None'}"
        ],
        'scan_id': scan_id
    })


@main.route('/api/scan/url', methods=['POST'])
@limiter.limit('500 per day')
def api_scan_url():
    user = get_request_user()
    if not user:
        return jsonify({'result': 'Error', 'message': 'API key or login required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result': 'Error', 'message': 'Daily scan limit reached for free tier.'}), 429

    payload = request.get_json(silent=True) or {}
    url = payload.get('url') or request.form.get('url', '')
    if not url:
        return jsonify({'result': 'Error', 'message': 'No URL provided'}), 400

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)[:3000]
        if not text:
            return jsonify({'result': 'Error', 'message': 'Could not extract content'}), 400

        gemini = analyze_with_gemini(f'This is content from URL: {url}\n\n{text}')
        wiki_ratio, keywords = get_wiki_score(text[:500])
        sensational_words = ['shocking', 'urgent', 'secret', 'bombshell', 'scandal', 'miracle', 'exposed']
        penalty = min(10, len([w for w in sensational_words if w in text.lower()]) * 3)
        base = 70 if gemini['result'] == 'REAL' else 45 if gemini['result'] == 'UNVERIFIED' else 15
        final_score = min(100, max(0, base + int(wiki_ratio * 20) - penalty))
        result = 'REAL' if final_score >= 70 else 'UNVERIFIED' if final_score >= 40 else 'FAKE'
        scan_id = f'AG-{uuid.uuid4().hex[:8].upper()}'
        record = ScanRecord(scan_id=scan_id, user_id=user.id, type='URL', result=result, confidence=final_score, details=json.dumps({'url': url, 'gemini': gemini, 'wiki': wiki_ratio}))
        db.session.add(record)
        db.session.commit()
        return jsonify({
            'result': result,
            'confidence': final_score,
            'summary': gemini['reasoning'],
            'forensic_details': [
                f"URL Scanned: {url[:50]}...",
                f"AI Analysis: {gemini['result']} ({gemini['confidence']}% confidence)",
                f"Fact Check Match: {wiki_ratio * 100}%",
                f"Flags: {', '.join(gemini['flags']) if gemini['flags'] else 'None'}"
            ],
            'scan_id': scan_id
        })
    except Exception as e:
        return jsonify({'result': 'Error', 'message': str(e)}), 500


@main.route('/api/scan/media', methods=['POST'])
@limiter.limit('500 per day')
def api_scan_media():
    user = get_request_user()
    if not user:
        return jsonify({'result': 'Error', 'message': 'API key or login required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result': 'Error', 'message': 'Daily scan limit reached for free tier.'}), 429

    if 'file' not in request.files:
        return jsonify({'result': 'Error', 'message': 'No Payload'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': 'Error', 'message': 'Empty Payload'}), 400

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    scores = []
    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov'))
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        file.save(tmp.name)
        if is_video:
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int((total / 5) * i))
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        scores.append(vision_model(preprocess(img).unsqueeze(0)).item())
            cap.release()
        else:
            img = Image.open(tmp.name).convert('RGB')
            with torch.no_grad():
                scores.append(vision_model(preprocess(img).unsqueeze(0)).item())
        os.unlink(tmp.name)
        avg = sum(scores) / len(scores) if scores else 0.5
        result, confidence = ('FAKE', avg * 100) if avg > 0.5 else ('REAL', (1 - avg) * 100)
        scan_id = f'AG-{uuid.uuid4().hex[:8].upper()}'
        record = ScanRecord(scan_id=scan_id, user_id=user.id, type='MEDIA', result=result, confidence=confidence, details='{}')
        db.session.add(record)
        db.session.commit()
        return jsonify({'result': result, 'confidence': round(confidence, 2), 'scan_id': scan_id})
    except Exception as e:
        return jsonify({'result': 'Error', 'message': str(e)}), 500


@main.route('/webhook/whatsapp', methods=['POST'])
def whatsapp_webhook():
    body = request.form.get('Body', '').strip()
    sender = request.form.get('From', '')
    response = MessagingResponse()
    if not body:
        response.message('AntiGravity Bot: No message detected. Send text or URL for analysis.')
        return Response(str(response), mimetype='application/xml')

    if body.lower().startswith('http'):
        try:
            url = body
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(r.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)[:3000]
            if not text:
                raise ValueError('Could not extract URL content')
            gemini = analyze_with_gemini(f"This is content from URL: {url}\n\n{text}")
            wiki_ratio, keywords = get_wiki_score(text[:500])
            sensational_words = ['shocking', 'urgent', 'secret', 'bombshell', 'scandal', 'miracle', 'exposed']
            penalty = min(10, len([w for w in sensational_words if w in text.lower()]) * 3)
            base = 70 if gemini['result'] == 'REAL' else 45 if gemini['result'] == 'UNVERIFIED' else 15
            final_score = min(100, max(0, base + int(wiki_ratio * 20) - penalty))
            message = f"Result: {('FAKE' if final_score < 40 else 'UNVERIFIED' if final_score < 70 else 'REAL')}\nConfidence: {final_score}%\n{gemini.get('reasoning', '')}"
        except Exception as e:
            message = f'Error processing URL: {e}'
    else:
        try:
            gemini = analyze_with_gemini(body)
            wiki_ratio, keywords = get_wiki_score(body)
            sensational_words = ['shocking', 'urgent', 'secret', 'bombshell', 'scandal', 'miracle', 'exposed']
            penalty = min(10, len([w for w in sensational_words if w in body.lower()]) * 3)
            base = 70 if gemini['result'] == 'REAL' else 45 if gemini['result'] == 'UNVERIFIED' else 15
            final_score = min(100, max(0, base + int(wiki_ratio * 20) - penalty))
            message = f"Result: {('FAKE' if final_score < 40 else 'UNVERIFIED' if final_score < 70 else 'REAL')}\nConfidence: {final_score}%\n{gemini.get('reasoning', '')}"
        except Exception as e:
            message = f'Error processing text: {e}'

    response.message(message)
    return Response(str(response), mimetype='application/xml')


@main.route('/report/<scan_id>')
def report(scan_id):
    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    record = ScanRecord.query.filter_by(scan_id=scan_id).first()
    if not record:
        return jsonify({'message': 'Scan ID not found.'}), 404

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(f'AntiGravity Report - {scan_id}')
    pdf.setFont('Helvetica-Bold', 18)
    pdf.drawString(40, 750, 'ANTIGRAVITY V4.0 FORENSIC REPORT')
    pdf.setFont('Helvetica', 12)
    pdf.drawString(40, 720, f'Scan ID: {record.scan_id}')
    pdf.drawString(40, 700, f'Type: {record.type}')
    pdf.drawString(40, 680, f'Result: {record.result}')
    pdf.drawString(40, 660, f'Confidence: {round(record.confidence,2)}%')
    pdf.drawString(40, 640, f"Timestamp: {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if record.user:
        pdf.drawString(40, 620, f'User: {record.user.email}')
    details = json.loads(record.details) if record.details else {}
    pdf.drawString(40, 600, 'Details:')
    y = 580
    for key, value in details.items():
        if y < 80:
            pdf.showPage()
            y = 750
        pdf.drawString(50, y, f'- {key}: {value}')
        y -= 20
    pdf.save()
    buffer.seek(0)
    return Response(buffer.getvalue(), mimetype='application/pdf', headers={
        'Content-Disposition': f'attachment; filename=antigravity_report_{scan_id}.pdf'
    })


@main.route('/history')
def get_history():
    user = get_request_user()
    if not user:
        return jsonify({'message': 'Login required to view history.'}), 401
    records = ScanRecord.query.filter_by(user_id=user.id).order_by(ScanRecord.timestamp.desc()).limit(10).all()
    return jsonify([r.to_dict() for r in records])

@main.route('/predict_news', methods=['POST'])
def predict_news():
    user = get_request_user()
    if not user:
        return jsonify({'result': 'Error', 'message': 'Authentication required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result': 'Error', 'message': 'Daily scan limit reached for free tier.'}), 429
    payload = request.get_json(silent=True) or {}
    text = request.form.get('text') or payload.get('text', '')
    if not text:
        return jsonify({'result': 'Error', 'message': 'Empty Stream'})

    gemini = analyze_with_gemini(text)
    wiki_ratio, keywords = get_wiki_score(text)

    sensational_words = ['shocking','urgent','secret','bombshell','scandal','miracle','exposed']
    penalty = min(10, len([w for w in sensational_words if w in text.lower()]) * 3)

    base = 70 if gemini["result"]=="REAL" else 45 if gemini["result"]=="UNVERIFIED" else 15
    final_score = min(100, max(0, base + int(wiki_ratio*20) - penalty))
    result = "REAL" if final_score>=70 else "UNVERIFIED" if final_score>=40 else "FAKE"

    scan_id = f"AG-{uuid.uuid4().hex[:8].upper()}"
    record = ScanRecord(scan_id=scan_id, user_id=user.id, type='TEXT', result=result, confidence=final_score,
        details=json.dumps({"gemini":gemini,"wiki":wiki_ratio,"keywords":keywords,"penalty":penalty}))
    db.session.add(record)
    db.session.commit()

    return jsonify({
        'result': result, 'confidence': final_score,
        'summary': gemini["reasoning"],
        'forensic_details': [
            f"AI Analysis: {gemini['result']} ({gemini['confidence']}% confidence)",
            f"Fact Check Match: {wiki_ratio*100}%",
            f"Sensationalism Penalty: -{penalty}pt",
            f"Flags: {', '.join(gemini['flags']) if gemini['flags'] else 'None'}"
        ],
        'scan_id': scan_id
    })

@main.route('/predict_deepfake', methods=['POST'])
def predict_deepfake():
    user = get_request_user()
    if not user:
        return jsonify({'result':'Error','message':'Authentication required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result':'Error','message':'Daily scan limit reached for free tier.'}), 429
    if 'file' not in request.files: return jsonify({'result':'Error','message':'No Payload'})
    file = request.files['file']
    if file.filename == '': return jsonify({'result':'Error','message':'Empty Payload'})

    preprocess = transforms.Compose([
        transforms.Resize((128,128)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    scores = []
    is_video = file.filename.lower().endswith(('.mp4','.avi','.mov'))
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        file.save(tmp.name)
        if is_video:
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, (total//5)*i)
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        scores.append(vision_model(preprocess(img).unsqueeze(0)).item())
            cap.release()
        else:
            img = Image.open(tmp.name).convert('RGB')
            with torch.no_grad():
                scores.append(vision_model(preprocess(img).unsqueeze(0)).item())
        os.unlink(tmp.name)
        avg = sum(scores)/len(scores) if scores else 0.5
        result, confidence = ("FAKE", avg*100) if avg>0.5 else ("REAL", (1-avg)*100)
        scan_id = f"AG-{uuid.uuid4().hex[:8].upper()}"
        record = ScanRecord(scan_id=scan_id, user_id=user.id, type='MEDIA', result=result, confidence=confidence, details="{}")
        db.session.add(record)
        db.session.commit()
        return jsonify({'result':result,'confidence':round(confidence,2),'scan_id':scan_id})
    except Exception as e:
        return jsonify({'result':'Error','message':str(e)})

@main.route('/predict_url', methods=['POST'])
def predict_url():
    user = get_request_user()
    if not user:
        return jsonify({'result': 'Error', 'message': 'Authentication required.'}), 401
    if not enforce_quota(user):
        return jsonify({'result': 'Error', 'message': 'Daily scan limit reached for free tier.'}), 429
    payload = request.get_json(silent=True) or {}
    url = request.form.get('url') or payload.get('url', '')
    if not url:
        return jsonify({'result': 'Error', 'message': 'No URL provided'})
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Extract main text
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)[:3000]
        
        if not text:
            return jsonify({'result': 'Error', 'message': 'Could not extract content'})
        
        # Use Gemini to analyze
        gemini = analyze_with_gemini(f"This is content from URL: {url}\n\n{text}")
        wiki_ratio, keywords = get_wiki_score(text[:500])
        
        sensational_words = ['shocking','urgent','secret','bombshell','scandal','miracle','exposed']
        penalty = min(10, len([w for w in sensational_words if w in text.lower()]) * 3)
        
        base = 70 if gemini["result"]=="REAL" else 45 if gemini["result"]=="UNVERIFIED" else 15
        final_score = min(100, max(0, base + int(wiki_ratio*20) - penalty))
        result = "REAL" if final_score>=70 else "UNVERIFIED" if final_score>=40 else "FAKE"
        
        scan_id = f"AG-{uuid.uuid4().hex[:8].upper()}"
        record = ScanRecord(scan_id=scan_id, user_id=user.id, type='URL', result=result, confidence=final_score,
            details=json.dumps({"url": url, "gemini": gemini, "wiki": wiki_ratio}))
        db.session.add(record)
        db.session.commit()
        
        return jsonify({
            'result': result,
            'confidence': final_score,
            'summary': gemini["reasoning"],
            'forensic_details': [
                f"URL Scanned: {url[:50]}...",
                f"AI Analysis: {gemini['result']} ({gemini['confidence']}% confidence)",
                f"Fact Check Match: {wiki_ratio*100}%",
                f"Flags: {', '.join(gemini['flags']) if gemini['flags'] else 'None'}"
            ],
            'scan_id': scan_id
        })
    except Exception as e:
        return jsonify({'result': 'Error', 'message': str(e)})

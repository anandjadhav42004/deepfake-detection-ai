import os
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, redirect, url_for
from flask_login import login_user, logout_user, current_user
from jwt import encode as jwt_encode, decode as jwt_decode, ExpiredSignatureError, InvalidTokenError
from flask_dance.contrib.google import make_google_blueprint
from . import db, login_manager, limiter
from .models import User, ScanRecord

auth = Blueprint('auth', __name__)

def _optional_env(name):
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    if not normalized or normalized.startswith('your_') or normalized.startswith('replace_with_'):
        return None

    return normalized


GOOGLE_CLIENT_ID = _optional_env('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = _optional_env('GOOGLE_CLIENT_SECRET')

google_bp = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    google_bp = make_google_blueprint(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scope=['profile', 'email'],
        redirect_url='/login/google'
    )

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
    except (ExpiredSignatureError, InvalidTokenError):
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

    api_user = get_api_user()
    if api_user:
        return api_user

    # Local demo fallback so the product remains usable without showing auth UI.
    from . import get_default_admin_email
    return User.query.filter_by(email=get_default_admin_email()).first()

def enforce_quota(user):
    if user is None: return False
    if user.is_premium: return True
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    scan_count = ScanRecord.query.filter(ScanRecord.user_id == user.id, ScanRecord.timestamp >= today_start).count()
    return scan_count < 10

def get_user_info():
    user = get_request_user()
    if not user: return None
    total_scans = ScanRecord.query.filter_by(user_id=user.id).count()
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    scans_today = ScanRecord.query.filter(ScanRecord.user_id == user.id, ScanRecord.timestamp >= today_start).count()
    return {
        'user': user.to_dict(),
        'scans_today': scans_today,
        'daily_limit': 10 if not user.is_premium else 'unlimited',
        'total_scans': total_scans
    }

@auth.route('/me')
def me():
    user = get_request_user()
    if not user:
        return jsonify({'authenticated': False}), 401
    data = get_user_info()
    data['authenticated'] = True
    return jsonify(data)

@auth.route('/register', methods=['POST'])
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

@auth.route('/login', methods=['POST'])
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

@auth.route('/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out.'})

@auth.route('/refresh_api_key', methods=['POST'])
def refresh_api_key():
    user = get_request_user()
    if not user: return jsonify({'message': 'Unauthorized'}), 401
    user.refresh_api_key()
    db.session.commit()
    return jsonify({'success': True, 'api_key': user.api_key})

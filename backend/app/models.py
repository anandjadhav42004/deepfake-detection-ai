import json
import uuid
import bcrypt
from datetime import datetime
from flask_login import UserMixin
from . import db


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(180), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    api_key = db.Column(db.String(64), unique=True, nullable=False, default=lambda: uuid.uuid4().hex)
    is_premium = db.Column(db.Boolean, default=False, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    scans = db.relationship('ScanRecord', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.email}>'

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        if not self.password_hash:
            return False
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
        except ValueError:
            return False

    def refresh_api_key(self):
        self.api_key = uuid.uuid4().hex
        return self.api_key

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'api_key': self.api_key,
            'is_premium': self.is_premium,
            'is_admin': self.is_admin,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }


class ScanRecord(db.Model):
    __tablename__ = 'scan_records'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    type = db.Column(db.String(20), nullable=False)
    result = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    details = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<ScanRecord {self.scan_id} {self.type} {self.result}>'

    def to_dict(self):
        return {
            'id': self.id,
            'scan_id': self.scan_id,
            'user_id': self.user_id,
            'type': self.type,
            'result': self.result,
            'confidence': round(self.confidence, 2),
            'details': json.loads(self.details) if self.details else {},
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }

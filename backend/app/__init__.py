import os
import pickle
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / 'backend' / '.env')


def _get_optional_env(name):
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    # Ignore README placeholder values so local development falls back cleanly.
    placeholder_values = {
        'your_gemini_api_key',
        'replace_with_secure_random_string',
        'your_google_client_id',
        'your_google_client_secret',
        'your_twilio_sid',
        'your_twilio_auth_token',
        'whatsapp:+1234567890',
        'postgresql://user:pass@host:port/dbname',
        'redis://localhost:6379/0',
    }
    if normalized in placeholder_values or normalized.startswith('your_') or normalized.startswith('replace_with_'):
        return None

    return normalized


def get_database_uri():
    db_url = _get_optional_env('DATABASE_URL')
    if db_url:
        return db_url

    db_path = (BASE_DIR / 'backend_data.db').resolve()
    return f"sqlite:///{db_path.as_posix()}"


def get_rate_limit_storage_uri():
    redis_url = _get_optional_env('REDIS_URL')
    if redis_url:
        return redis_url
    return 'memory://'


def get_default_admin_email():
    return (_get_optional_env('ADMIN_EMAIL') or 'admin@gmail.com').strip()


def get_default_admin_password():
    return (_get_optional_env('ADMIN_PASSWORD') or 'admin123').strip()


def _ensure_default_admin():
    from .models import User

    default_admin_email = get_default_admin_email()
    default_admin_password = get_default_admin_password()

    if not default_admin_email or not default_admin_password:
        return

    default_user = User.query.filter_by(email=default_admin_email).first()
    if default_user is None:
        admin_user = User(
            name='Administrator',
            email=default_admin_email,
            is_admin=True,
            is_premium=True
        )
        admin_user.set_password(default_admin_password)
        db.session.add(admin_user)
        db.session.commit()
        print(f"Created default admin user: {default_admin_email}")
    else:
        default_user.is_admin = True
        default_user.is_premium = True
        if not default_user.password_hash:
            default_user.set_password(default_admin_password)
        db.session.commit()
        print(f"Ensured default admin user: {default_admin_email}")


def _migrate_database():
    try:
        inspector = inspect(db.engine)
        if inspector.has_table('scan_records'):
            columns = [col['name'] for col in inspector.get_columns('scan_records')]
            if 'user_id' not in columns:
                print('Migrating scan_records: adding user_id column')
                with db.engine.begin() as connection:
                    connection.execute(text('ALTER TABLE scan_records ADD COLUMN user_id INTEGER'))
    except Exception as exc:
        print(f'Could not migrate database schema: {exc}')


db = SQLAlchemy()
login_manager = LoginManager()
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=get_rate_limit_storage_uri(),
    in_memory_fallback_enabled=True
)

# Load NLP fake-news models at module import time
_NLP_DIR = BASE_DIR / 'models' / 'fake_news'
nlp_model = None
nlp_vectorizer = None
try:
    with open(_NLP_DIR / 'model.pkl', 'rb') as _f:
        nlp_model = pickle.load(_f)
    with open(_NLP_DIR / 'vectorizer.pkl', 'rb') as _f:
        nlp_vectorizer = pickle.load(_f)
    print('NLP fake-news models loaded OK.')
except Exception as _nlp_err:
    print(f'NLP model load skipped: {_nlp_err}')


class BaseConfig:
    SECRET_KEY = _get_optional_env('SECRET_KEY') or 'antigravity-v4-secret'
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    UPLOAD_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.mp4', '.mov', '.avi']
    UPLOAD_PATH = str(BASE_DIR / 'backend' / 'tmp_uploads')
    JWT_SECRET_KEY = _get_optional_env('SECRET_KEY') or 'antigravity-v4-secret'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    CORS_RESOURCES = {r"/api/*": {"origins": "*"}, r"/webhook/*": {"origins": "*"}, r"/*": {"origins": "*"}}


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


def create_app(config_name=None):
    template_folder = str(BASE_DIR / 'frontend' / 'templates')
    static_folder = str(BASE_DIR / 'frontend' / 'static')

    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder,
        static_url_path='/static'
    )

    config_name = config_name or os.getenv('FLASK_ENV', 'development')
    if config_name.lower() == 'production':
        app.config.from_object(ProductionConfig)
    else:
        app.config.from_object(DevelopmentConfig)

    database_uri = get_database_uri()
    app.config['SQLALCHEMY_DATABASE_URI'] = database_uri

    os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)

    # The application is normally served by Flask itself, but these settings also
    # allow the static client to be hosted separately during development.
    CORS(app, resources=app.config['CORS_RESOURCES'], supports_credentials=False)
    db.init_app(app)
    login_manager.init_app(app)
    limiter.init_app(app)

    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access AntiGravity.'

    from app.routes import main
    from app.auth import auth, google_bp, load_user
    app.register_blueprint(main)
    app.register_blueprint(auth, url_prefix='/auth')

    if google_bp is not None:
        app.register_blueprint(google_bp, url_prefix='/login')

    with app.app_context():
        db.create_all()
        _migrate_database()
        _ensure_default_admin()

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        """Keep API failures machine-readable instead of returning Flask HTML."""
        if request.path.startswith(('/api/', '/auth/', '/predict_', '/history', '/admin', '/dashboard', '/status')):
            app.logger.warning('%s %s: %s', request.method, request.path, error)
            return jsonify({'success': False, 'result': 'Error', 'message': error.description}), error.code
        return error

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception('Unhandled error while processing %s %s', request.method, request.path)
        if request.path.startswith(('/api/', '/auth/', '/predict_', '/history', '/admin', '/dashboard', '/status')):
            return jsonify({'success': False, 'result': 'Error', 'message': 'An internal server error occurred.'}), 500
        return jsonify({'success': False, 'message': 'An internal server error occurred.'}), 500

    return app

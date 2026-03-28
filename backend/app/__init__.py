import os
from datetime import timedelta
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from flask_login import LoginManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(BASE_DIR, 'backend', '.env'))


def get_database_uri():
    db_url = os.getenv('DATABASE_URL')
    if db_url and db_url.strip():
        return db_url.strip()
    return 'sqlite:///' + os.path.join(BASE_DIR, 'backend_data.db')


def get_rate_limit_storage_uri():
    redis_url = os.getenv('REDIS_URL')
    if redis_url and redis_url.strip():
        return redis_url.strip()
    return 'memory://'


def _ensure_default_admin():
    from .models import User

    default_admin_email = os.getenv('ADMIN_EMAIL', 'admin@antigravity.local').strip()
    default_admin_password = os.getenv('ADMIN_PASSWORD', 'Admin123!').strip()

    if not default_admin_email or not default_admin_password:
        return

    current_admin = User.query.filter_by(is_admin=True).first()
    if current_admin:
        return

    existing = User.query.filter_by(email=default_admin_email).first()
    if existing is None:
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
        existing.is_admin = True
        db.session.commit()
        print(f"Updated existing user to admin: {default_admin_email}")


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


class BaseConfig:
    SECRET_KEY = os.getenv('SECRET_KEY', 'antigravity-v4-secret')
    SQLALCHEMY_DATABASE_URI = get_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    UPLOAD_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.mp4', '.mov', '.avi']
    UPLOAD_PATH = os.path.join(BASE_DIR, 'backend', 'tmp_uploads')
    JWT_SECRET_KEY = os.getenv('SECRET_KEY', 'antigravity-v4-secret')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=8)
    CORS_RESOURCES = {r"/api/*": {"origins": "*"}, r"/webhook/*": {"origins": "*"}, r"/*": {"origins": "*"}}


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


def create_app(config_name=None):
    template_folder = os.path.join(BASE_DIR, 'frontend', 'templates')
    static_folder = os.path.join(BASE_DIR, 'frontend', 'static')

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

    CORS(app, resources=app.config['CORS_RESOURCES'])
    db.init_app(app)
    login_manager.init_app(app)
    limiter.init_app(app)

    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access AntiGravity.'

    from app.routes import main, google_bp
    app.register_blueprint(main)
    if google_bp is not None:
        app.register_blueprint(google_bp, url_prefix='/login')

    with app.app_context():
        db.create_all()
        _migrate_database()
        _ensure_default_admin()

    return app

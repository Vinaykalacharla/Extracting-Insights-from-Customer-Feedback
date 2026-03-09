from flask import Flask, request
from flask_jwt_extended import JWTManager
from markupsafe import Markup, escape
import os
import sys
from .postgres import PostgresCompat
from .background import init_background_queue

# Add parent directory to Python path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

mysql = PostgresCompat()
jwt = JWTManager()
analysis_queue = None

def create_app():
    app = Flask(__name__)

    # Load configuration
    app.config['DATABASE_URL'] = Config.DATABASE_URL
    app.config['REDIS_URL'] = Config.REDIS_URL
    app.config['SLACK_WEBHOOK_URL'] = Config.SLACK_WEBHOOK_URL
    app.config['SMTP_HOST'] = Config.SMTP_HOST
    app.config['SMTP_PORT'] = Config.SMTP_PORT
    app.config['SMTP_USERNAME'] = Config.SMTP_USERNAME
    app.config['SMTP_PASSWORD'] = Config.SMTP_PASSWORD
    app.config['SMTP_FROM_EMAIL'] = Config.SMTP_FROM_EMAIL
    app.config['ALERT_EMAIL_TO'] = Config.ALERT_EMAIL_TO
    app.config['DB_HOST'] = Config.DB_HOST
    app.config['DB_PORT'] = Config.DB_PORT
    app.config['DB_USER'] = Config.DB_USER
    app.config['DB_PASSWORD'] = Config.DB_PASSWORD
    app.config['DB_NAME'] = Config.DB_NAME
    app.config['DB_SSLMODE'] = Config.DB_SSLMODE
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    app.config['MIN_PASSWORD_LENGTH'] = Config.MIN_PASSWORD_LENGTH

    # JWT Token stored in cookies
    app.config['JWT_TOKEN_LOCATION'] = ['cookies']
    app.config['JWT_COOKIE_SECURE'] = os.environ.get('JWT_COOKIE_SECURE', 'false').lower() == 'true'
    app.config['JWT_COOKIE_CSRF_PROTECT'] = os.environ.get('JWT_COOKIE_CSRF_PROTECT', 'true').lower() == 'true'
    app.config['JWT_CSRF_CHECK_FORM'] = True
    app.secret_key = os.environ.get('FLASK_SECRET_KEY') or os.environ.get('JWT_SECRET_KEY', 'dev-secret')
    
    # Initialize extensions
    mysql.init_app(app)
    jwt.init_app(app)
    init_background_queue(app)

    @app.context_processor
    def inject_template_helpers():
        csrf_cookie_name = app.config.get('JWT_ACCESS_CSRF_COOKIE_NAME', 'csrf_access_token')
        csrf_token = request.cookies.get(csrf_cookie_name, '')

        def csrf_form_input():
            if not csrf_token:
                return ''
            return Markup(
                f'<input type="hidden" name="csrf_token" value="{escape(csrf_token)}">'
            )

        return {
            'csrf_form_input': csrf_form_input,
            'jwt_csrf_token': csrf_token,
            'min_password_length': app.config.get('MIN_PASSWORD_LENGTH', 8),
        }

    # Import routes
    from .routes import main
    app.register_blueprint(main)

    return app


import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DATABASE_URL = os.getenv('DATABASE_URL', '').strip() or None
    REDIS_URL = os.getenv('REDIS_URL', '').strip() or None
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '').strip() or None
    SMTP_HOST = os.getenv('SMTP_HOST', '').strip() or None
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '').strip() or None
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '').strip() or None
    SMTP_FROM_EMAIL = os.getenv('SMTP_FROM_EMAIL', '').strip() or None
    ALERT_EMAIL_TO = os.getenv('ALERT_EMAIL_TO', '').strip() or None

    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '5432'))
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    DB_SSLMODE = os.getenv('DB_SSLMODE', 'require')

    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-key-12345')
    MIN_PASSWORD_LENGTH = int(os.getenv('MIN_PASSWORD_LENGTH', '8'))

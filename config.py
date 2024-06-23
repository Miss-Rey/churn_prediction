import os
from dotenv import load_dotenv

# Determine the folder of the top-level directory of this project
BASEDIR = os.path.abspath(os.path.dirname(__file__))

# Load environment variables from .env file if it exists
load_dotenv(os.path.join(BASEDIR, '.env'))

class Config:
    # Secret key setup
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    
    # Debug mode
    DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Model file paths
    COMBINED_MODEL_PATH = os.environ.get('COMBINED_MODEL_PATH') or os.path.join(BASEDIR, 'app', 'models', 'combined_model.pkl')
    BEST_MODEL_PATH = os.environ.get('BEST_MODEL_PATH') or os.path.join(BASEDIR, 'app', 'models', 'best_model.pkl')

    # Dataset path
    DATASET_PATH = os.environ.get('DATASET_PATH') or os.path.join(BASEDIR, 'first_telc.csv')

    # Flask-WTF settings
    CSRF_ENABLED = True
    CSRF_SESSION_KEY = os.environ.get('CSRF_SESSION_KEY') or 'hard-to-guess-key'

    # Logging setup
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT', 'False').lower() in ('true', '1', 't')

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Log to stderr
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    COMBINED_MODEL_PATH = os.environ.get('COMBINED_MODEL_PATH') or os.path.join(basedir, 'app', 'models', 'combined_model.pkl')
    BEST_MODEL_PATH = os.environ.get('BEST_MODEL_PATH') or os.path.join(basedir, 'app', 'models', 'best_model.pkl')
    DATASET_PATH = os.environ.get('DATASET_PATH') or os.path.join(basedir, 'first_telc.csv')
    
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
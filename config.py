# config.py

import os

class Config:
    DEBUG = False
    TESTING = False
    # Tambahkan konfigurasi lain jika diperlukan

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False


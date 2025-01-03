# app/__init__.py

from flask import Flask
from app.routes import main as main_blueprint

def create_app():
    app = Flask(__name__)

    # Register Blueprints
    app.register_blueprint(main_blueprint)

    return app


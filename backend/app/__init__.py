from flask import Flask
from app.clasificador_imagenes import clasificar_imagen

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    return app

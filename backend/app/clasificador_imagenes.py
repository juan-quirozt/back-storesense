import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Cargar el modelo preentrenado
modelo_path = os.path.join(os.path.dirname(__file__), '../modelo/modelo_imagenes.h5')
modelo = load_model(modelo_path)

# Lista de clases
clases = ['Jeans', 'Sofa', 'Tshirt', 'Tv']

def clasificar_imagen(imagen_path):
    img = image.load_img(imagen_path, target_size=(224, 224), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = modelo.predict(img_array)
    indice_clase = np.argmax(prediccion)
    
    return clases[indice_clase], prediccion[0][indice_clase]

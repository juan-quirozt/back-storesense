import os
import pandas as pd
import joblib
import numpy as np

# Cargar los archivos necesarios
vectorizer_path = os.path.join(os.path.dirname(__file__), '../modelo/vectorizer.pkl')
similarity_matrix_path = os.path.join(os.path.dirname(__file__), '../modelo/similarity_matrix.pkl')
data_path = os.path.join(os.path.dirname(__file__), '../modelo/data.csv')

vectorizer = joblib.load(vectorizer_path)
similarity_matrix = joblib.load(similarity_matrix_path)
df = pd.read_csv(data_path, sep=",")

def recomendar_productos(producto, n_recomendaciones=7):
    """
    Dado un producto, devuelve una lista de los productos más similares ponderados por rating y número de reseñas.
    """
    if producto not in df['name'].values:
        return {"error": "Producto no encontrado"}

    # Obtener el índice del producto en el DataFrame
    idx = df[df['name'] == producto].index[0]

    # Obtener las similitudes con los demás productos
    similitudes = similarity_matrix[idx]

    # Ponderar por ratings y número de calificaciones (log transform)
    weighted_scores = similitudes * df['ratings'] * np.log1p(df['no_of_ratings'])

    # Obtener los índices de los productos más relevantes
    indices_similares = np.argsort(weighted_scores)[::-1][1:n_recomendaciones + 1]

    # Obtener los productos recomendados con su información completa
    productos_recomendados = df.iloc[indices_similares][['name', 'main_category', 'sub_category', 'ratings', 'no_of_ratings']].to_dict(orient='records')

    return {"producto": producto, "recomendaciones": productos_recomendados}

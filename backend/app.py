from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin # Permitir peticiones desde otro dominio (Next.js)
import os
from werkzeug.utils import secure_filename
from app.clasificador_imagenes import clasificar_imagen
from app.clasificador_demanda import predecir_demanda
from app.recomendacion import recomendar_productos

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#-------------------------
#Prediccion de demanda
#-------------------------
@app.route('/api/predecir_demanda', methods=['POST'])
@cross_origin()
def api_predecir_demanda():
    """
    API para predecir demanda basado en store_id y dept_id.
    """
    print("\n📩 **Solicitud recibida en /api/predecir_demanda**")
    data = request.json
    store_id = data.get("store_id")
    dept_id = data.get("dept_id")

    if not store_id or not dept_id:
        print("❌ Error: store_id o dept_id faltantes")
        return jsonify({"error": "Faltan store_id o dept_id"}), 400

    try:
        predicciones = predecir_demanda(store_id, dept_id)
        print(f"✅ Predicción exitosa para Store {store_id}, Dept {dept_id}")
        print(f"🔍 Predicciones generadas: {predicciones}")
        return jsonify(predicciones)
    except Exception as e:
        print(f"❌ Error en la predicción de demanda: {e}")
        return jsonify({"error": str(e)}), 500


#-------------------------
#Clasificacion de imagenes
#-------------------------

@app.route('/api/clasificar', methods=['POST'])
def clasificar():
    print("\n📩 **Solicitud recibida en /api/clasificar**")
    
    # Imprimir detalles de la petición
    print("🔍 Headers:", request.headers)
    print("🔍 Content-Type:", request.content_type)
    print("🔍 request.files:", request.files)
    print("🔍 request.form:", request.form)

    if 'imagen' not in request.files:
        print("❌ Error: No se envió ninguna imagen")
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    imagen = request.files['imagen']
    
    if imagen.filename == '':
        print("❌ Error: Archivo vacío")
        return jsonify({"error": "Archivo vacío"}), 400

    filename = secure_filename(imagen.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    imagen.save(filepath)

    print(f"✅ Imagen guardada en: {filepath}")

    try:
        clase_predicha, confianza = clasificar_imagen(filepath)
        print(f"✅ Clasificación exitosa: Clase - {clase_predicha}, Confianza - {confianza}")

        return jsonify({
            "clase": clase_predicha,
            "confianza": float(confianza)
        })
    except Exception as e:
        print(f"❌ Error al clasificar la imagen: {e}")
        return jsonify({"error": "Error interno al procesar la imagen"}), 500


#-------------------------
# Recomendación de productos
#-------------------------
@app.route('/api/recomendar', methods=['POST'])
@cross_origin()
def api_recomendar():
    """
    API para recomendar productos basados en similitud.
    """
    print("\n📩 **Solicitud recibida en /api/recomendar**")
    data = request.json
    print(f"📜 Datos recibidos: {data}")  # <-- Agregado para depuración
    
    producto = data.get("producto")

    if not producto:
        print("❌ Error: Falta el nombre del producto")
        return jsonify({"error": "Falta el nombre del producto"}), 400

    try:
        recomendaciones = recomendar_productos(producto)
        print(f"✅ Recomendación exitosa para {producto}: {recomendaciones}")
        return jsonify(recomendaciones)
    except Exception as e:
        print(f"❌ Error en la recomendación: {e}")
        return jsonify({"error": str(e)}), 500



#----------------
#------------
#-------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Flask correrá en el puerto 5000

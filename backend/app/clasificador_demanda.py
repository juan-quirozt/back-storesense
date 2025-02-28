import os
import pandas as pd
import joblib
from datetime import date

# Cargar modelo y encoders
modelo_ventas_path = os.path.join(os.path.dirname(__file__), '../modelo/modelo_demanda.pkl')
encoder_store_path = os.path.join(os.path.dirname(__file__), '../modelo/encoder_store.pkl')
encoder_dept_path = os.path.join(os.path.dirname(__file__), '../modelo/encoder_dept.pkl')
historial_path = os.path.join(os.path.dirname(__file__), '../modelo/datos_historicos.csv')  # Ruta del dataset

modelo_ventas = joblib.load(modelo_ventas_path)
encoder_store = joblib.load(encoder_store_path)
encoder_dept = joblib.load(encoder_dept_path)

# Cargar datos históricos
df_historial = pd.read_csv(historial_path)
df_historial["ds"] = pd.to_datetime(df_historial["ds"])  # Asegurar que la columna de fecha es datetime

def predecir_demanda(store_id, dept_id):
    """
    Predice las ventas para los próximos 4 domingos de una tienda y departamento específicos.
    """
    # 🔹 Manejo de valores desconocidos en la codificación
    store_encoded = encoder_store.transform([store_id])[0] if store_id in encoder_store.classes_ else -1
    dept_encoded = encoder_dept.transform([dept_id])[0] if dept_id in encoder_dept.classes_ else -1

    # 🔹 Generar fechas futuras (4 semanas)
    future_dates = pd.date_range(start=df_historial["ds"].max(), periods=4, freq="W")
    future_final = pd.DataFrame({"ds": future_dates})

    # 🔹 Convertir fechas a ordinal
    future_final["ds"] = future_final["ds"].map(pd.Timestamp.toordinal)
    future_final["Store"] = store_encoded
    future_final["Dept"] = dept_encoded

    # 🔹 Calcular promedios de las últimas 4 semanas para las variables externas
    df_historial["week_of_year"] = df_historial["ds"].dt.isocalendar().week
    last_weeks_avg = df_historial.groupby("week_of_year")[["Temperature", "Fuel_Price", "CPI", "Unemployment"]].mean()

    for i, feature in enumerate(["Temperature", "Fuel_Price", "CPI", "Unemployment"]):
        future_final[feature] = last_weeks_avg.iloc[-4:].values[:, i]  # Tomar valores de las últimas 4 semanas

    # 🔹 Hacer predicción
    future_final["yhat"] = modelo_ventas.predict(future_final)

    # 🔹 Convertir fechas de vuelta a datetime
    future_final["ds"] = pd.to_datetime(future_final["ds"].map(date.fromordinal))

    return future_final[['ds', 'Store', 'Dept', 'yhat']].to_dict(orient="records")
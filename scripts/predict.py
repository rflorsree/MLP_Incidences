# scripts/predict.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime

# Cargar modelos y transformadores
model = tf.keras.models.load_model("models/mlp_model.keras")
ohe = joblib.load("models/encoder_causa.pkl")
scaler_y = joblib.load("models/scaler_y.pkl")

# Entradas
fecha_str = "2025-04-02"  # Fecha futura
causa = "Robo de CTO"      # Causa conocida

#Procesar fechas y dias
fecha = pd.to_datetime(fecha_str)
dia_semana = fecha.weekday()

# Codificar causa
causa_encoded = ohe.transform([[causa]])
X_input = np.hstack([[dia_semana], causa_encoded[0]]).reshape(1, -1)

# predecir
y_pred_scaled = model.predict(X_input)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# imprimir resultados
print(f"Predicción para {fecha_str} - {causa}")
print(f"- Número de incidencias:         {y_pred[0]:.2f}")
print(f"- Clientes afectados:           {y_pred[1]:.2f}")
print(f"- Tiempo promedio muerto (min): {y_pred[2]:.2f}")
print(f"- Tiempo promedio resolución:   {y_pred[3]:.2f}")

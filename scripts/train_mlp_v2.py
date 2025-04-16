import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_excel("data/EXCELINCIDENCIAS.xlsx", sheet_name="Sheet1")

# Procesar fechas
df["INICIO INCIDENCIA"] = pd.to_datetime(df["INICIO INCIDENCIA"])
df["HORA DE LLEGADA"] = pd.to_datetime(df["HORA DE LLEGADA"])
df["CIERRE DE INCIDENCIA"] = pd.to_datetime(df["CIERRE DE INCIDENCIA"])

# Agrupación por día
df["FECHA"] = df["INICIO INCIDENCIA"].dt.date
data_grouped = df.groupby("FECHA").agg({
    "INICIO INCIDENCIA": "min",
    "HORA DE LLEGADA": "min",
    "CLIENTES": "sum",
    "TIEMPO MUERTO (MIN)": "mean",
    "TIEMPO RESOLUCION (MIN)": "mean"
}).reset_index()

data_grouped["incidencias"] = df.groupby("FECHA").size().values

# Variables temporales
data_grouped["hora_inicio"] = data_grouped["INICIO INCIDENCIA"].dt.hour
data_grouped["mes"] = data_grouped["INICIO INCIDENCIA"].dt.month
data_grouped["dia_semana"] = data_grouped["INICIO INCIDENCIA"].dt.weekday
data_grouped["semana_del_anio"] = data_grouped["INICIO INCIDENCIA"].dt.isocalendar().week.astype(int)
data_grouped["minutos_respuesta"] = (
    data_grouped["HORA DE LLEGADA"] - data_grouped["INICIO INCIDENCIA"]
).dt.total_seconds() / 60

# Entradas (X)
X = data_grouped[[
    "hora_inicio", "mes", "dia_semana", "semana_del_anio", "minutos_respuesta",
    "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"
]].values

# Salidas (y)
y = data_grouped[[
    "incidencias", "CLIENTES", "TIEMPO MUERTO (MIN)", "TIEMPO RESOLUCION (MIN)"
]].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# División
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# Modelo MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4)  # 4 salidas
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# Entrenamiento
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Guardar modelo y escaladores
os.makedirs("models", exist_ok=True)
model.save("models/mlp_model.keras")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

# Crear carpeta outputs
os.makedirs("outputs", exist_ok=True)

# Gráfica pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('MLP - Pérdida durante entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig("outputs/loss_curve.png")
plt.close()

# Predicción
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_orig = scaler_y.inverse_transform(y_test)

labels = ["Incidencias", "Clientes", "TM Muerto", "TM Resolución"]
maes, r2s = [], []

for i in range(4):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_orig[:, i], y_pred[:, i], alpha=0.5)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"MLP - Real vs Predicho - {labels[i]}")
    plt.grid()
    plt.savefig(f"outputs/real_vs_pred_{i}_{labels[i].replace(' ', '_').lower()}.png")
    plt.close()

    maes.append(mean_absolute_error(y_test_orig[:, i], y_pred[:, i]))
    r2s.append(r2_score(y_test_orig[:, i], y_pred[:, i]))

# MAE
plt.figure(figsize=(10, 5))
plt.bar(labels, maes, color='skyblue')
plt.title('MLP - MAE por variable de salida')
plt.ylabel('Mean Absolute Error')
plt.grid(axis='y')
plt.savefig("outputs/mae_comparativo.png")
plt.close()

# R²
plt.figure(figsize=(10, 5))
plt.bar(labels, r2s, color='lightgreen')
plt.title('MLP - R² Score por variable de salida')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig("outputs/r2_comparativo.png")
plt.close()

print("Modelo MLP entrenado, guardado y gráficas exportadas a 'outputs'.")

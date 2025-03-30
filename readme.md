# Proyecto: Predicción de Incidencias con MLP

Este proyecto utiliza una red neuronal MLP (Multilayer Perceptron) para predecir información relacionada con incidencias técnicas, basada en la fecha futura y la causa de la incidencia.

## 📊 Estructura del Proyecto
```
MLP_Prediccion_Incidencias/
|
├── data/              # Datos originales (Excel)
│   └── EXCELINCIDENCIAS.xlsx
|
├── scripts/           # Códigos de entrenamiento y predicción
│   ├── train_mlp.py
│   └── predict.py
|
├── models/            # Modelo entrenado y transformadores
│   ├── mlp_model/     # Modelo de TensorFlow
│   ├── encoder_causa.pkl
│   └── scaler_y.pkl
|
├── outputs/           # Resultados, gráficas o logs (opcional)
|
└── notebooks/         # Jupyter notebooks de exploración (opcional)
```

---

## 🚀 Entrenamiento del Modelo

El archivo `scripts/train_mlp.py` entrena una red MLP con los siguientes datos:
- **Input**: día de la semana (de la fecha), y causa (codificada)
- **Output**:
  - Número de incidencias
  - Número de clientes afectados
  - Tiempo promedio muerto (min)
  - Tiempo promedio de resolución (min)

### Ejecución:
```bash
cd scripts
python train_mlp.py
```
El modelo y los preprocesadores se guardarán en la carpeta `../models/`.

---

## 🤖 Predicción

El archivo `scripts/predict.py` permite hacer una predicción ingresando:
- Fecha futura (YYYY-MM-DD)
- Causa de incidencia (texto)

### Ejecución:
```bash
cd scripts
python predict.py
```
Los resultados se mostrarán en consola.

---

## 📁 Requisitos

Instala las dependencias necesarias:

```bash
pip install tensorflow pandas numpy scikit-learn openpyxl joblib
```

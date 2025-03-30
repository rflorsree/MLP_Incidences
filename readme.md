# Proyecto: Predicción de Incidencias con MLP

Este proyecto utiliza una red neuronal MLP (Multilayer Perceptron) para predecir información relacionada con incidencias en la red elecrtica con una fecha y causa futura.

## 📊 Estructura del Proyecto
```
MLP_Prediccion_Incidencias/
|
├── data/              
│   └── EXCELINCIDENCIAS.xlsx
|
├── scripts/           # scripts de entrenamiento y prediccion
│   ├── train_mlp.py
│   └── predict.py
|
├── models/            # Modelo 
│   ├── mlp_model/     
│   ├── encoder_causa.pkl
│   └── scaler_y.pkl
|
└── outputs/           # Resultados de los modelos

```

---

## 🚀 Entrenamiento del Modelo

El archivo `scripts/train_mlp.py` entrena una red MLP con los siguientes datos:
- **Input**: Fecha  y causa 
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


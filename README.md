# Práctica 04 · Maldad pedagógica
## Diabetes Predictor — Sistema ML end-to-end

**DataSet Médica → ML Keras/Adam → Lambda → Frontend**

---

## Integrantes

| Nombre | Código |
|--------|--------|
|        |        |
|        |        |
|        |        |

---

## URLs de entrega

- **Vercel (frontend):** `https://practica04-diabetes-predictor.vercel.app`
- **Lambda API:**        `https://practica04-diabetes-predictor.vercel.app/api/predict`
- **GitHub:**            `https://github.com/<usuario>/practica04`

---

## Arquitectura

```
DataSet Médica  →  ML Keras/Adam  →  Lambda Vercel  →  Frontend
  Pima 768 obs     8→32→16→1 sig     POST /api/predict   index.html
  UCI / Kaggle     binary_xentropy   Solo stdlib Python   Sliders UI
```

---

## Dataset

**Pima Indians Diabetes Database**
- **Fuente:** [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) · [UCI](https://archive.ics.uci.edu/datasets?search=health)
- **Origen:** National Institute of Diabetes and Digestive and Kidney Diseases
- **Publicación:** Smith et al. (1988)
- **768 mujeres** de herencia Pima Indian, ≥ 21 años
- **Positivos:** 268 (34.9%)  **Negativos:** 500 (65.1%)

| Feature | μ | σ | Rango |
|---------|---|---|-------|
| Pregnancies | 3.8 | 3.4 | 0–17 |
| Glucose (mg/dL) | 120.9 | 32.0 | 0–199 |
| BloodPressure (mmHg) | 69.1 | 19.4 | 0–122 |
| SkinThickness (mm) | 20.5 | 16.0 | 0–99 |
| Insulin (μU/mL) | 79.8 | 115.2 | 0–846 |
| BMI (kg/m²) | 32.0 | 7.9 | 0–67.1 |
| DiabetesPedigreeFunction | 0.47 | 0.33 | 0.078–2.42 |
| Age | 33.2 | 11.8 | 21–81 |

---

## Estructura del proyecto

```
practica04/
├── api/
│   └── predict.py       ← Lambda serverless (solo stdlib Python)
├── public/
│   └── index.html       ← Frontend completo
├── model.py             ← Entrenamiento Keras/Adam (local)
├── requirements.txt     ← Sin dependencias (stdlib)
├── vercel.json          ← Config de deploy
└── README.md
```

---

## Paso 1 — Subir a GitHub

```bash
# Clonar / iniciar repo
git init
git add .
git commit -m "feat: Práctica 04 - Diabetes Predictor"

# Crear repo en github.com y luego:
git remote add origin https://github.com/<TU-USUARIO>/practica04.git
git branch -M main
git push -u origin main
```

---

## Paso 2 — Deploy en Vercel

### Opción A — Dashboard web (más fácil, recomendada)

1. Entrar a [vercel.com](https://vercel.com) → **Add New Project**
2. **Import Git Repository** → seleccionar `practica04`
3. Framework Preset: **Other**
4. Root Directory: dejar en `/`
5. Click **Deploy** ✅

### Opción B — CLI

```bash
npm install -g vercel
vercel login
vercel deploy --prod
```

> **¿Por qué no necesita TensorFlow en Vercel?**
> El Lambda (`api/predict.py`) usa un modelo logístico calibrado con
> los coeficientes reales del dataset Pima, implementado con `math`
> (stdlib de Python). Esto evita el límite de 50MB de Vercel y hace
> el deploy instantáneo. El modelo Keras con TensorFlow se entrena
> **localmente** con `model.py`.

---

## Paso 3 — Entrenar el modelo (local, opcional)

```bash
# Instalar dependencias de entrenamiento
pip install tensorflow pandas scikit-learn matplotlib joblib numpy

# Descargar diabetes.csv desde Kaggle y colocarlo aquí

# Entrenar
python model.py
# Genera: diabetes_model.h5, scaler.pkl,
#         training_curves.png, roc_curve.png
```

---

## API Reference

### `POST /api/predict`

**Request:**
```json
{
  "pregnancies":   4,
  "glucose":       121,
  "bloodpressure": 69,
  "skinthickness": 20,
  "insulin":       80,
  "bmi":           32.0,
  "dpf":           0.47,
  "age":           33
}
```

**Response 200:**
```json
{
  "prob":          0.3241,
  "diabetes":      false,
  "riesgo":        "BAJO",
  "recomendacion": "Mantener hábitos saludables. Control rutinario anual suficiente.",
  "model_info": {
    "type":      "Logistic Regression (Pima calibrated)",
    "dataset":   "Pima Indians Diabetes — UCI/Kaggle",
    "features":  8,
    "accuracy":  "77.6%",
    "auc_roc":   "0.834"
  }
}
```

---

## Métricas del modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | ~78% |
| AUC-ROC | ~0.84 |
| Optimizer | Adam (lr=0.001) |
| Loss | binary_crossentropy |
| Arquitectura | Dense(8→32→16→1) |

---

## Referencias

- Smith, J.W. et al. (1988). *ADAP learning algorithm for diabetes mellitus.* SCAMC.
- [Kaggle · Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [UCI · Health Datasets](https://archive.ics.uci.edu/datasets?search=health)
- [TensorFlow / Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Vercel Python Serverless](https://vercel.com/docs/functions/serverless-functions/runtimes/python)

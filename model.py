"""
model.py — Práctica 04 · Maldad pedagógica
Entrena una red neuronal con Keras + Adam sobre el
dataset Pima Indians Diabetes (UCI / Kaggle).

EJECUTAR LOCALMENTE antes del deploy:
  pip install tensorflow pandas scikit-learn matplotlib joblib numpy
  python model.py

DESCARGA el dataset desde:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
  → guardar como diabetes.csv en la raíz del proyecto
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── 1. Cargar dataset ─────────────────────────────────────────
df = pd.read_csv('diabetes.csv')
print(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Positivos: {df['Outcome'].sum()}  Negativos: {(df['Outcome']==0).sum()}\n")

# ── 2. Imputar zeros (missing values reales en Pima) ──────────
for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
    n = (df[col] == 0).sum()
    med = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, med)
    print(f"  {col}: {n} zeros imputados con mediana={med:.1f}")

# ── 3. Features y target ──────────────────────────────────────
FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
X = df[FEATURES].values
y = df['Outcome'].values

# ── 4. Split 80/20 con stratify ───────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 5. Normalizar ─────────────────────────────────────────────
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)
joblib.dump(sc, 'scaler.pkl')
print(f"\nScaler guardado: scaler.pkl")
print(f"Medias: {sc.mean_.round(3)}")
print(f"Stds:   {sc.scale_.round(3)}\n")

# ── 6. Red neuronal Keras ─────────────────────────────────────
tf.random.set_seed(42)
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(8,), name='dense_1'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', name='dense_2'),
    layers.Dropout(0.2),
    layers.Dense(1,  activation='sigmoid', name='output')
], name='diabetes_model')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
model.summary()

# ── 7. Entrenar ───────────────────────────────────────────────
early = keras.callbacks.EarlyStopping(
    monitor='val_auc', patience=15,
    restore_best_weights=True, mode='max')

history = model.fit(
    X_tr, y_tr,
    epochs=100, batch_size=32,
    validation_data=(X_te, y_te),
    callbacks=[early], verbose=1)

# ── 8. Evaluar ────────────────────────────────────────────────
loss, acc, auc = model.evaluate(X_te, y_te, verbose=0)
print(f"\n{'='*45}")
print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Loss     : {loss:.4f}")
print(f"{'='*45}\n")

y_prob = model.predict(X_te).flatten()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_te, y_pred,
      target_names=['No diabetes','Diabetes']))

# ── 9. Guardar modelo ─────────────────────────────────────────
model.save('diabetes_model.h5')
print("✅ diabetes_model.h5 guardado")

# ── 10. Gráficas de entrenamiento ─────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['accuracy'],     label='Train')
ax[0].plot(history.history['val_accuracy'], label='Val')
ax[0].set_title('Accuracy por epoch')
ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Accuracy')
ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(history.history['loss'],     label='Train')
ax[1].plot(history.history['val_loss'], label='Val')
ax[1].set_title('Loss por epoch')
ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Loss')
ax[1].legend(); ax[1].grid(alpha=.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=120)
print("✅ training_curves.png guardado")
plt.show()

# ── 11. Curva ROC ─────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_te, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='#D85A30', lw=2,
         label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve — Diabetes Model')
plt.legend(); plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=120)
print("✅ roc_curve.png guardado")
plt.show()

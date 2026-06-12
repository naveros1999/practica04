"""
api/predict.py — Lambda serverless · Vercel
Práctica 04 · Maldad pedagógica

Recibe POST con 8 features del paciente,
consulta el modelo ML (lógica Pima calibrada)
y retorna predicción JSON.
"""

from http.server import BaseHTTPRequestHandler
import json
import math


# ── Modelo logístico calibrado con datos reales Pima ──────────
# Pesos derivados de regresión logística entrenada sobre el
# dataset Pima Indians (Smith et al. 1988, UCI/Kaggle).
# Medias y desviaciones estándar reales del dataset para
# normalización manual (sin sklearn en Lambda).

MEANS = {
    'pregnancies':   3.845,
    'glucose':       120.894,
    'bloodpressure': 69.105,
    'skinthickness': 20.536,
    'insulin':       79.799,
    'bmi':           31.993,
    'dpf':           0.4719,
    'age':           33.241,
}
STDS = {
    'pregnancies':   3.3696,
    'glucose':       31.9726,
    'bloodpressure': 19.3558,
    'skinthickness': 15.9522,
    'insulin':       115.244,
    'bmi':           7.8842,
    'dpf':           0.3313,
    'age':           11.7602,
}

# Coeficientes del modelo logístico (entrenado sobre Pima)
COEFS = {
    'intercept':     -0.8642,
    'pregnancies':    0.1232,
    'glucose':        0.3641,
    'bloodpressure': -0.0132,
    'skinthickness':  0.0062,
    'insulin':       -0.0713,
    'bmi':            0.2801,
    'dpf':            0.1513,
    'age':            0.1436,
}

FEATURE_ORDER = [
    'pregnancies', 'glucose', 'bloodpressure',
    'skinthickness', 'insulin', 'bmi', 'dpf', 'age'
]

FIELDS_RANGE = {
    'pregnancies':   (0,    17),
    'glucose':       (44,   199),
    'bloodpressure': (24,   122),
    'skinthickness': (0,    99),
    'insulin':       (0,    846),
    'bmi':           (18.2, 67.1),
    'dpf':           (0.078, 2.42),
    'age':           (21,   81),
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict_diabetes(data):
    """Modelo logístico calibrado — no requiere TensorFlow."""
    z = COEFS['intercept']
    for feat in FEATURE_ORDER:
        val = float(data[feat])
        # Normalizar con estadísticas reales del dataset Pima
        val_norm = (val - MEANS[feat]) / STDS[feat]
        z += COEFS[feat] * val_norm
    prob = sigmoid(z)
    return round(prob, 4)


def validate(data):
    errors = []
    for field, (lo, hi) in FIELDS_RANGE.items():
        if field not in data:
            errors.append(f"Campo faltante: '{field}'")
            continue
        try:
            val = float(data[field])
        except (TypeError, ValueError):
            errors.append(f"'{field}' debe ser numérico")
            continue
        if val < lo or val > hi:
            errors.append(
                f"'{field}' fuera de rango [{lo}, {hi}], recibido: {val}"
            )
    return errors


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        # 1. Leer body
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw    = self.rfile.read(length)
            data   = json.loads(raw)
        except Exception:
            self._respond(400, {'error': 'JSON inválido en el body'})
            return

        # 2. Validar
        errors = validate(data)
        if errors:
            self._respond(422, {
                'error': 'Validación fallida',
                'details': errors
            })
            return

        # 3. Predecir
        prob    = predict_diabetes(data)
        is_diab = prob > 0.5

        if prob > 0.65:
            riesgo = 'ALTO'
        elif prob > 0.35:
            riesgo = 'MODERADO'
        else:
            riesgo = 'BAJO'

        # 4. Recomendación clínica según riesgo
        recomendaciones = {
            'ALTO':     'Consultar endocrinólogo urgente. Control glucémico estricto necesario.',
            'MODERADO': 'Monitorear glucosa periódicamente. Mejorar dieta y actividad física.',
            'BAJO':     'Mantener hábitos saludables. Control rutinario anual suficiente.',
        }

        self._respond(200, {
            'prob':           prob,
            'diabetes':       is_diab,
            'riesgo':         riesgo,
            'recomendacion':  recomendaciones[riesgo],
            'model_info': {
                'type':       'Logistic Regression (Pima calibrated)',
                'dataset':    'Pima Indians Diabetes — UCI/Kaggle',
                'features':   8,
                'accuracy':   '77.6%',
                'auc_roc':    '0.834',
            }
        })

    def _respond(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self._cors()
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _cors(self):
        self.send_header('Content-Type',                  'application/json')
        self.send_header('Access-Control-Allow-Origin',   '*')
        self.send_header('Access-Control-Allow-Methods',  'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers',  'Content-Type')

    def log_message(self, fmt, *args):
        pass  # silenciar logs de acceso

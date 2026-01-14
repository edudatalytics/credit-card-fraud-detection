# ============================
# PREDIÇÃO DE FRAUDE
# ============================

import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path

# ============================
# CONFIGURAÇÕES
# ============================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "fraud_detection_model"
MODEL_ALIAS = "Production"
THRESHOLD = 0.1

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ============================
# LOAD FEATURE COLUMNS
# ============================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS_PATH = (
    BASE_DIR / "Analise" / "artifacts" / "feature_columns.pkl"
)

FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)
# ============================
# LOAD MODEL
# ============================

def load_model():
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)

model = load_model()

# ============================
# FUNÇÃO DE PREDIÇÃO
# ============================

def predict_fraud(input_dict: dict) -> pd.DataFrame:
    """
    Recebe input parcial (Time, Amount),
    reconstrói o DataFrame completo
    e retorna a predição.
    """

    # cria dataframe vazio com TODAS as colunas
    input_df = pd.DataFrame(columns=FEATURE_COLUMNS)

    # inicializa com zeros
    input_df.loc[0] = 0.0

    # preenche apenas o que o usuário informou
    for col, value in input_dict.items():
        input_df.loc[0, col] = value

    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    return pd.DataFrame({
        "prob_fraude": proba,
        "fraude_predita": pred
    })

# ============================
# PREDIÇÃO DE FRAUDE
# Modelo: Random Forest com class_weight='balanced'
# Threshold: 0.5
# Métricas: ROC-AUC 97.7% | Recall 82.7% | Precision 81.8%
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

# MUDANÇA PRINCIPAL: Threshold otimizado
THRESHOLD = 0.5  # Antes: 0.1 | Agora: 0.5 (melhor F1-Score)

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
    """
    Carrega modelo Random Forest do MLflow
    Modelo de produção: RF com class_weight='balanced'
    """
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)

model = load_model()

print("="*60)
print("🔒 SISTEMA DE DETECÇÃO DE FRAUDE")
print("="*60)
print(f"Modelo: Random Forest com class_weight='balanced'")
print(f"Threshold: {THRESHOLD}")
print(f"ROC-AUC: 97.7% | Recall: 82.7% | Precision: 81.8%")
print("="*60 + "\n")

# ============================
# FUNÇÃO DE PREDIÇÃO
# ============================

def predict_fraud(input_dict: dict) -> pd.DataFrame:
    """
    Recebe input parcial (Time, Amount, etc),
    reconstrói o DataFrame completo
    e retorna a predição.
    
    Args:
        input_dict: Dicionário com features da transação
        
    Returns:
        DataFrame com probabilidade e predição
    """

    # cria dataframe vazio com TODAS as colunas
    input_df = pd.DataFrame(columns=FEATURE_COLUMNS)

    # inicializa com zeros
    input_df.loc[0] = 0.0

    # preenche apenas o que o usuário informou
    for col, value in input_dict.items():
        input_df.loc[0, col] = value

    # Predição
    input_df = input_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    proba = model.predict_proba(input_df)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    return pd.DataFrame({
        "prob_fraude": proba,
        "fraude_predita": pred
    })


# ============================
# EXEMPLO DE USO
# ============================

if __name__ == "__main__":
    
    print("📝 EXEMPLO DE USO:\n")
    
    # Transação de teste
    transacao_teste = {
        'Time': 12345,
        'Amount': 149.62,
        'V1': -1.35,
        'V2': -0.07
    }
    
    print("Input:")
    for k, v in transacao_teste.items():
        print(f"  {k}: {v}")
    
    # Fazer predição
    resultado = predict_fraud(transacao_teste)
    
    print("\n📊 Resultado:")
    print(resultado.to_string(index=False))
    
    # Interpretação
    prob = resultado['prob_fraude'].iloc[0]
    fraude = resultado['fraude_predita'].iloc[0]
    
    print(f"\n{'🚨 FRAUDE DETECTADA' if fraude == 1 else '✅ TRANSAÇÃO LEGÍTIMA'}")
    print(f"Probabilidade: {prob:.2%}")
    print(f"Threshold: {THRESHOLD}")
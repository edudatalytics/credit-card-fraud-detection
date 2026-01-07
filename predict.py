#%% IMPORTS
import joblib
import pandas as pd

THRESHOLD = 0.001

model = joblib.load("C:\\Users\\User\\Desktop\\ANALISE DE FRAUDES DE CRED\\models\\modelo_fraude_producao.pkl")

def predict_fraud(X):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    return pd.DataFrame({
        "prob_fraude": proba,
        "fraude_predita": pred
    })



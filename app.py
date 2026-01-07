# ================================
# 1️⃣ IMPORTS
# ================================
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ================================
# 2️⃣ CONFIGURAÇÃO DA PÁGINA
# ================================
st.set_page_config(
    page_title="Detector de Fraude",
    page_icon="🚨",
    layout="centered"
)

# ================================
# 3️⃣ CONSTANTES
# ================================
THRESHOLD_PRODUCAO = 0.005
MODEL_PATH = Path("models/modelo_fraude_producao.pkl")

# ================================
# 4️⃣ LOAD DO MODELO
# ================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Modelo não encontrado na pasta models/")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ================================
# 5️⃣ INTERFACE
# ================================
st.title("🚨 Detector de Fraude em Cartão de Crédito")

st.markdown("""
Este aplicativo utiliza **Machine Learning** para detectar **transações suspeitas**  
com base em dados históricos de fraude.

👉 Preencha os dados da transação abaixo.
""")

st.divider()

st.subheader("📥 Dados da Transação")

time = st.number_input(
    "Tempo desde a primeira transação (em segundos)",
    min_value=0.0,
    value=10000.0
)

amount = st.number_input(
    "Valor da transação (R$)",
    min_value=0.0,
    value=100.0
)

# ================================
# 6️⃣ PREDIÇÃO
# ================================
if st.button("🔍 Analisar Transação"):
    input_dict = {}

    for feature in model.feature_names_in_:
        if feature == "Time":
            input_dict[feature] = time
        elif feature == "Amount":
            input_dict[feature] = amount
        else:
            # V1–V28 (simulados como zero)
            input_dict[feature] = 0.0

    # DataFrame na ordem correta
    input_data = pd.DataFrame([input_dict])[model.feature_names_in_]

    proba_fraude = model.predict_proba(input_data)[:, 1][0]
    is_fraud = proba_fraude >= THRESHOLD_PRODUCAO

    st.divider()
    st.subheader("📊 Resultado da Análise")

    st.metric(
        label="Probabilidade de Fraude",
        value=f"{proba_fraude:.2%}"
    )

    if is_fraud:
        st.error("🚨 **Transação classificada como FRAUDE**")
    else:
        st.success("✅ **Transação classificada como LEGÍTIMA**")

    st.caption(f"Threshold utilizado: {THRESHOLD_PRODUCAO}")

# ================================
# 7️⃣ RODAPÉ
# ================================
st.divider()
st.caption(
    "Projeto desenvolvido para fins educacionais • Ciência de Dados • Machine Learning"
)


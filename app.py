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

import joblib
import streamlit as st

# ================================
# 3️⃣ CONSTANTES
# ================================
THRESHOLD_PRODUCAO = 0.005
MODEL_PATH = "models/modelo_fraude_producao.pkl"

# ================================
# 4️⃣ LOAD DO MODELO
# ================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Modelo não encontrado na pasta models/ \nErro: {e}")
        st.stop()


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
proba = model.predict_proba(input_data)[0][1]  # probabilidade da classe de fraude
pred = 1 if proba >= THRESHOLD_PRODUCAO else 0

st.write(f"🔢 Probabilidade estimada de fraude: **{proba:.2%}**")

if pred == 1:
    st.error("⚠️ Transação suspeita! Existe chance de ser fraude.")
else:
    st.success("🛡️ Transação normal. Sem sinais de fraude.")

if proba < 0.01:
    st.success("🟢 Baixíssimo risco")
elif proba < 0.05:
    st.warning("🟡 Risco moderado")
else:
    st.error("🔴 Alto risco de fraude")
    st.caption(f"Threshold utilizado: {THRESHOLD_PRODUCAO}")

# ================================
# 7️⃣ RODAPÉ
# ================================
st.divider()
st.caption(
    "Projeto desenvolvido para fins educacionais • Ciência de Dados • Machine Learning"
)


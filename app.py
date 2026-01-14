import streamlit as st

from predict import predict_fraud

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="Detector de Fraude em Cartão de Crédito",
    page_icon="🚨",
    layout="centered"
)

# =========================
# TÍTULO
# =========================

st.title("🚨 Detector de Fraude em Cartão de Crédito")

st.markdown(
    """
    Este aplicativo utiliza um modelo de **Machine Learning**
    treinado para detectar **transações fraudulentas**.

    O modelo foi selecionado com base em **recall, precision,
    custo operacional e ajuste de threshold**.
    """
)

st.divider()

# =========================
# INPUTS DO USUÁRIO
# =========================

st.subheader("📥 Dados da Transação")

time = st.number_input(
    "Tempo desde a primeira transação (segundos)",
    min_value=0.0,
    value=10000.0
)

amount = st.number_input(
    "Valor da transação",
    min_value=0.0,
    value=100.0
)

# =========================
# BOTÃO DE PREDIÇÃO
# =========================

if st.button("🔍 Analisar Transação"):

    input_dict = {
        "Time": time,
        "Amount": amount
    }

    result = predict_fraud(input_dict)

    prob_fraude = result.loc[0, "prob_fraude"]
    fraude_predita = result.loc[0, "fraude_predita"]

    st.divider()
    st.subheader("📊 Resultado da Análise")

    st.metric(
        label="Probabilidade de Fraude",
        value=f"{prob_fraude:.2%}"
    )

    if fraude_predita == 1:
        st.error("🚨 **Transação classificada como FRAUDE**")
    else:
        st.success("✅ **Transação classificada como LEGÍTIMA**")

    st.caption("Threshold de decisão: 0.1")

# =========================
# RODAPÉ
# =========================

st.divider()
st.caption(
    "Projeto educacional • Ciência de Dados • Machine Learning • MLflow"
)

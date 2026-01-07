# 🚨 Detecção de Fraudes em Cartões de Crédito com Machine Learning

## 📌 Visão Geral

Este projeto tem como objetivo **identificar transações fraudulentas em cartões de crédito** utilizando técnicas de **Machine Learning**, com foco em **aplicação prática, organização de código e uso em produção**.

O trabalho simula um cenário real de negócio, abordando desde a análise exploratória até a disponibilização de um modelo treinado para consumo via aplicação interativa.

---

## 🎯 Objetivo do Projeto

* Detectar transações potencialmente fraudulentas
* Lidar com **dados altamente desbalanceados**
* Avaliar modelos com métricas adequadas ao contexto de fraude
* Criar um **pipeline pronto para produção**
* Disponibilizar uma interface simples para uso do modelo

---

## 🧠 Abordagem Utilizada

* Análise Exploratória de Dados (EDA)
* Tratamento de desbalanceamento (class_weight / SMOTE)
* Treinamento e comparação de modelos
* Avaliação com métricas como **Recall, Precision e ROC-AUC**
* Ajuste de **threshold de decisão** visando minimizar falsos negativos
* Persistência do modelo com Joblib
* Aplicação interativa com Streamlit

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib / Seaborn
* Streamlit
* Joblib

---

## 📂 Estrutura do Projeto

```
ANALISE_DE_FRAUDES/
│
├── Analise/
│   └── credit_card_fraud_analysis.py
│
├── models/
│   └── modelo_fraude_producao.pkl
│
├── Streamlit/
│   └── app.py
│
├── credit_card_fraud_synthetic.csv
├── predict.py
├── requirements.txt
└── README.md
```

---

## 🤖 Modelo Final

Após a comparação entre diferentes algoritmos, o modelo selecionado foi:

* **Regressão Logística com class_weight balanceado**

Motivos da escolha:

* Maior estabilidade em diferentes thresholds
* Melhor controle do trade-off entre Recall e Precision
* Facilidade de interpretação
* Maior confiabilidade para uso em produção

O modelo foi salvo como um **pipeline completo**, incluindo todas as etapas de pré-processamento.

---

## 🚀 Como Executar o Projeto

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run Streamlit/app.py
```

---

## 📊 Resultado Esperado

A aplicação retorna:

* Probabilidade estimada de fraude
* Classificação final da transação (Fraude ou Legítima)
* Decisão baseada em threshold configurável

O foco está em **não perder fraudes**, característica essencial em problemas financeiros.

---

## 🔮 Próximos Passos

* Deploy em nuvem (Heroku / Render)
* Criação de API REST para consumo externo
* Monitoramento de desempenho do modelo
* Re-treinamento automático

---

## 👤 Autor

**Eduardo Matos**
Formado em Ciência de Dados
Foco em Machine Learning, Análise de Dados e Aplicações em Produção

---

📌 *Projeto desenvolvido para fins educacionais e demonstração de habilidades técnicas em Ciência de Dados.*

# 🚨 Detecção de Fraudes em Cartões de Crédito

Projeto de **Machine Learning end-to-end** para detecção de transações fraudulentas,
com foco em **dados desbalanceados**, **ajuste de threshold baseado em custo operacional**
e **deploy com MLflow + Streamlit**.

---

## 📌 Objetivo
Desenvolver um modelo capaz de identificar transações fraudulentas,
minimizando perdas financeiras e falsos positivos,
utilizando boas práticas de ciência de dados e MLOps.

---

## 🧠 Principais Desafios
- Forte desbalanceamento de classes (fraude < 2%)
- Threshold padrão (0.5) inadequado para o negócio
- Necessidade de controle de custo operacional
- Versionamento e rastreabilidade do modelo

---

## 🧪 Modelos Avaliados
- Regressão Logística (baseline)
- Regressão Logística com `class_weight`
- Random Forest (baseline, SMOTE e class_weight)
- Gradient Boosting
- XGBoost (challenger)

---

## 📊 Métricas Utilizadas
- Recall (Fraude)
- Precision (Fraude)
- F1-score
- ROC-AUC
- PR-AUC
- **Custo Operacional (FP x FN)**

---

## 🏆 Modelo Final (Produção)
- **Modelo:** Regressão Logística com `class_weight=balanced`
- **Threshold:** `0.1`
- Escolhido por apresentar melhor equilíbrio entre:
  - Recall elevado
  - Menor custo operacional
  - Estabilidade e interpretabilidade

---

## 🔁 MLflow
- Rastreamento de experimentos
- Comparação de modelos e thresholds
- Registro e versionamento do modelo
- Uso de alias (`Production`)

---

## 🚀 Deploy com Streamlit
Interface interativa para simular transações e obter:
- Probabilidade de fraude
- Classificação final (fraude / legítima)

Para rodar o app:
```bash
streamlit run app.py

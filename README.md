# 🔒 Sistema de Detecção de Fraude em Cartão de Crédito

Sistema completo de Machine Learning para detecção de transações fraudulentas em tempo real usando Random Forest.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.3%2B-blue)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Banner do Projeto](docs/banner.png)

---

## 📊 Métricas do Modelo

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **ROC-AUC** | 97.7% | Excelente capacidade de discriminação |
| **Recall** | 82.7% | Detecta 83 de cada 100 fraudes |
| **Precision** | 81.8% | 82% das alertas são fraudes reais |
| **F1-Score** | 0.82 | Ótimo equilíbrio precision/recall |
| **PR-AUC** | 81.8% | Robusto para dados desbalanceados |

**Threshold otimizado:** 0.5 (via análise de custo-benefício)

---

## 🚀 Demonstração

### Interface Web (Streamlit)

```bash
streamlit run app.py
```

![Demo da Aplicação](docs/demo.gif)

### Análise Individual
Teste transações específicas com probabilidade em tempo real.

### Análise em Lote
Faça upload de CSV com múltiplas transações para análise em massa.

### Dashboard
Visualize métricas, distribuições e matriz de confusão.

---

## 💰 Impacto de Negócio

- **Fraudes Detectadas:** 81 de 98 (82.7%)
- **Falsos Positivos:** Apenas 18 em 56,864 transações legítimas (0.03%)
- **Economia Estimada:** R$ 80,820 por período
- **ROI:** 15:1 (para cada R$ 1 investido, retorno de R$ 15)

---

## 🛠️ Tecnologias Utilizadas

### Machine Learning
- **Scikit-learn** - Algoritmos de ML
- **Random Forest** - Modelo principal
- **XGBoost** - Modelo alternativo testado
- **Imbalanced-learn** - SMOTE para balanceamento

### MLOps
- **MLflow** - Tracking de experimentos
- **Joblib** - Serialização de modelos

### Visualização
- **Streamlit** - Interface web interativa
- **Plotly** - Gráficos interativos
- **Matplotlib/Seaborn** - Visualizações estáticas

### Desenvolvimento
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Python 3.8+** - Linguagem base

---

## 📁 Estrutura do Projeto

```
credit-card-fraud-detection/
├── app.py                          # Interface Streamlit
├── predict.py                      # Sistema de predição
├── fraud_detection_v2.py           # Script de treinamento
├── requirements.txt                # Dependências
├── README.md                       # Este arquivo
├── .gitignore                      # Arquivos ignorados
│
├── artifacts/                      # Artefatos do modelo
│   ├── feature_columns.pkl         # Colunas do dataset
│   └── fraud_model.pkl             # Modelo treinado (não versionado)
│
├── data/                           # Datasets
│   ├── .gitkeep                    # Manter pasta no Git
│   └── creditcard.csv              # Dataset (não versionado - 150MB)
│
├── plots/                          # Gráficos gerados
│   ├── .gitkeep
│   ├── confusion_matrix.png        # Matriz de confusão
│   └── model_comparison.png        # Comparação de modelos
│
├── notebooks/                      # Jupyter notebooks
│   └── exploratory_analysis.ipynb  # Análise exploratória
│
└── docs/                           # Documentação
    ├── banner.png
    ├── demo.gif
    └── metodologia.md
```

---

## ⚙️ Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Criar Ambiente Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar Dataset

O dataset **Credit Card Fraud Detection** deve ser baixado do Kaggle:

**Link:** https://kaggle.com/datasets/mlg-ulb/creditcardfraud

1. Baixe o arquivo `creditcard.csv` (150 MB)
2. Coloque em `data/creditcard.csv`

**Alternativa via Kaggle API:**

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

---

## 🎯 Uso

### 1. Treinar o Modelo

```bash
python fraud_detection_v2.py
```

Este script:
- Carrega e processa o dataset
- Treina múltiplos modelos (LogReg, RF, XGBoost)
- Otimiza threshold
- Salva modelo em `artifacts/fraud_model.pkl`
- Registra experimentos no MLflow

### 2. Fazer Predições

```python
from predict import predict_fraud

# Transação de exemplo
transacao = {
    'Time': 12345,
    'Amount': 149.62,
    'V1': -1.35,
    'V2': -0.07
}

# Predição
resultado = predict_fraud(transacao)
print(resultado)

# Output:
#    prob_fraude  fraude_predita
#    0.034        0
```

### 3. Executar Interface Web

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

---

## 🔬 Metodologia

### 1. Análise Exploratória
- Dataset: 284,807 transações
- Fraudes: 492 (0.17%)
- Forte desbalanceamento (99.83% legítimas)

### 2. Pré-processamento
- Features V1-V28: componentes PCA (já anonimizadas)
- Time: segundos desde primeira transação
- Amount: valor em euros
- Sem missing values

### 3. Técnicas de Balanceamento Testadas
- **class_weight='balanced'** ✅ Escolhido
- SMOTE (Synthetic Minority Oversampling)
- Undersampling da classe majoritária

### 4. Modelos Testados

| Modelo | ROC-AUC | Recall | Precision | F1-Score |
|--------|---------|--------|-----------|----------|
| Logistic Regression (baseline) | 0.974 | 88.9% | 75.4% | 0.815 |
| Logistic Regression + class_weight | 0.976 | 88.9% | 86.4% | 0.876 |
| Random Forest + class_weight | **0.977** | **82.7%** | **81.8%** | **0.822** |
| XGBoost | 0.975 | 85.7% | 79.2% | 0.823 |

**Modelo escolhido:** Random Forest com class_weight='balanced'

### 5. Otimização de Threshold

Testamos thresholds de 0.01 a 0.5:
- **Threshold 0.1:** Recall alto (90%+), mas muitos falsos positivos
- **Threshold 0.5:** Melhor equilíbrio F1-Score (0.82) ✅
- **Custo operacional:** R$ 1,000/fraude perdida vs R$ 10/falso positivo

---

## 📈 Resultados

### Matriz de Confusão (Conjunto de Teste)

```
                Predito Legítimo    Predito Fraude
Real Legítimo        56,846             18
Real Fraude             17              81
```

### Interpretação
- **True Negatives:** 56,846 (99.97% das legítimas corretas)
- **True Positives:** 81 (82.7% das fraudes detectadas)
- **False Positives:** 18 (apenas 0.03% de alarmes falsos)
- **False Negatives:** 17 (17.3% de fraudes perdidas)

### Curva ROC-AUC

![ROC Curve](plots/roc_curve.png)

---

## 🧪 Testes

### Teste Unitário

```bash
python -m pytest tests/
```

### Teste de Predição

```python
# Transação legítima
predict_fraud({'Amount': 50.0, 'Time': 10000})
# Output: prob_fraude=0.02, fraude_predita=0

# Transação suspeita
predict_fraud({'Amount': 15000.0, 'Time': 80000})
# Output: prob_fraude=0.78, fraude_predita=1
```

---

## 🐛 Problemas Conhecidos

- [ ] MLflow pode falhar se não estiver rodando (`mlflow ui`)
- [ ] Dataset muito grande (150MB) - não versionado no Git
- [ ] Modelo precisa de retreinamento mensal para evitar drift

---

## 🔄 Roadmap

### Versão 2.0
- [ ] Deploy no Heroku/AWS
- [ ] API REST com FastAPI
- [ ] Retreinamento automático
- [ ] Monitoramento de drift
- [ ] Explicabilidade com SHAP values
- [ ] Ensemble (RF + XGBoost)

### Versão 1.1
- [x] Interface Streamlit
- [x] Sistema de predição
- [x] MLflow tracking
- [x] Threshold otimizado

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Autor

**Eduardo Matos**  
Cientista de Dados

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Eduardo_Matos-blue)](https://www.linkedin.com/in/matos-eduardo)
[![GitHub](https://img.shields.io/badge/GitHub-edudatalytics-black)](https://github.com/edudatalytics)
[![Email](https://img.shields.io/badge/Email-eduardomatos2399@gmail.com-red)](mailto:eduardomatos2399@gmail.com)

---

## 🙏 Agradecimentos

- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspiração: Projetos da comunidade de Data Science
- Bibliotecas open-source incríveis: Scikit-learn, Streamlit, MLflow

---

## 📚 Referências

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. [Handling Imbalanced Datasets](https://imbalanced-learn.org/)
3. [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
4. [Credit Card Fraud Detection Paper](https://www.researchgate.net/publication/...)

---

**⭐ Se este projeto foi útil, considere dar uma estrela!**
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sympy import resultant

# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================
# Dataset sintético de transações de cartão de crédito

df = pd.read_csv('C:\\Users\\User\\Desktop\\ANALISE DE FRAUDES DE CRED\\credit_card_fraud_synthetic.csv')
df.head()
#%%
df.shape
df.info()
#%%
df['Class'].value_counts()
df['Class'].value_counts(normalize=True)

# O dataset apresenta severo desbalanceamento entre classes, 
# exigindo técnicas específicas de avaliação e balanceamento para evitar modelos enviesados.

#%% EXPLORE
# Distribuição da Variável Alvo
# Análise da proporção entre transações legítimas e fraudulentas.

sns.countplot(x='Class', data=df)
plt.title("Distribuição das Classes (0 = Legítima, 1 = Fraude)")
plt.show()
#%% 
# Avaliação da distribuição dos valores das transações.
plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Distribuição de valores das transações')
plt.xlabel("Valor da Transação")
plt.ylabel("Frequência")
plt.show()
#%%
# Comparação dos valores das transações entre classes.
plt.figure(figsize=(8,4))
sns.boxplot( x = 'Class', y = 'Amount', data = df)
plt.xlabel("Classe")
plt.ylabel("Valor da Transação")
plt.show()

# Fraudes tendem a ocorrer em valores específicos
#%% 
# Análise do comportamento temporal das transações.
plt.figure(figsize=(8,4))
sns.histplot(df['Time'], bins = 50, kde= True)
plt.title('Distribuição da variavel Time')
plt.xlabel("Tempo")
plt.ylabel("Frequência")
plt.show()


#%%
# Avaliação das correlações lineares entre as variáveis.
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap = 'coolwarm', center = 0)
plt.title("Mapa de correlação entre as variáveis")
plt.show()

#%%
# Correlação de Spearman com a variável alvo
df.corr(method='spearman')['Class']

#%% MODIFY
# separando features e target
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2,
                                                      random_state = 42,
                                                      stratify = y)

# A separação dos dados foi realizada antes de qualquer transformação para evitar vazamento de dados
#%%
# Padronização das variáveis Time e Amount
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(
    X_train[['Time', 'Amount']])

X_test[['Time', 'Amount']] = scaler.transform(
    X_test[['Time', 'Amount']])

X_train[['Amount', 'Time']].describe()

# Após a padronização com StandardScaler, 
# as variáveis apresentaram média aproximadamente zero e desvio padrão próximo de um

#%% Verificando se esta normalizado
X_train[['Amount', 'Time']].mean().round(6)
X_train[['Amount', 'Time']].std().round(6)

# SMOTE para balanceamento da classe minoritária
smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)


print(y_train_sm.value_counts())
#%% MODEL
# Definição inicial de modelos candidatos.
# O objetivo é estabelecer um baseline simples,
# sem qualquer técnica de balanceamento, para entender
# o impacto do desbalanceamento dos dados.

models = {
    'logistic_regression' : LogisticRegression(random_state=42,
                                               max_iter=1000),
    'random_forest' : RandomForestClassifier(random_state=42,
                                             n_estimators=200,)

}

# Treinamento da Regressão Logística sem balanceamento.
pipe_log_base = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])
# Treinamento da Regressão Logística com class_weight='balanced'.
pipe_log_bal = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    ))
])

# Treinamento da Regressão Logística com SMOTE.
pipe_log_smote = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(
        random_state=42,
        sampling_strategy=0.5
    )),
    ('model', LogisticRegression(
        random_state=42,
        max_iter=1000
    ))
])
# Treinamento do Random Forest sem balanceamento.
pipe_rf_base = Pipeline(steps=[
    ('model', RandomForestClassifier(
        random_state=42,
        n_estimators=200
    ))
])

# Treinamento do Random Forest com class_weight='balanced'.
pipe_rf_bal = Pipeline(steps=[
    ('model', RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        class_weight='balanced'
    ))
])
# Treinamento do Random Forest com SMOTE.
pipe_rf_smote = Pipeline(steps=[
    ('smote', SMOTE(
        random_state=42,
        sampling_strategy=0.5
    )),
    ('model', RandomForestClassifier(
        random_state=42,
        n_estimators=200
    ))
])
#%%
# ============================
# AVALIAÇÃO DOS MODELOS
# ============================

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    print("Recall Fraude:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Não Fraude', 'Fraude'],
        yticklabels=['Não Fraude', 'Fraude']
    )
    plt.title(f"Matriz de Confusão — {name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

models = {
    "Logística — Baseline": pipe_log_base,
    "Logística — class_weight": pipe_log_bal,
    "Logística — SMOTE": pipe_log_smote,
    "Random Forest — Baseline": pipe_rf_base,
    "Random Forest — class_weight": pipe_rf_bal,
    "Random Forest — SMOTE": pipe_rf_smote
}

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, name)

#%% 
# Função para avaliar o impacto de diferentes thresholds
# sobre Recall e Precision da classe fraude.
# Essencial em problemas desbalanceados, onde o threshold padrão (0.5)
# não representa a melhor decisão de negócio.
#%%# 

# FUNÇÃO DE AVALIAÇÃO DO MODELO
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        print(f"⚠️ Modelo {name} não possui predict_proba")
        return None

    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))
    print("Recall Fraude:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    return y_proba

# FUNÇÃO PARA AVALIAR THRESHOLDS

def avaliar_thresholds(y_true, y_proba, thresholds, model_name):
    resultados = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        resultados.append({
            "Modelo": model_name,
            "Threshold": t,
            "Recall": recall_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)
        })

    return pd.DataFrame(resultados)


# TREINO + AVALIAÇÃO DOS MODELOS
y_probas = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = evaluate_model(model, X_test, y_test, name)

    # PROTEÇÃO CONTRA None
    if y_proba is not None:
        y_probas[name] = y_proba

# DEFININDO THRESHOLDS

thresholds_testados = [0.001, 0.005, 0.01, 0.05]

# RANDOM FOREST + SMOTE

df_rf_smote_thr = avaliar_thresholds(
    y_test,
    y_probas["Random Forest — SMOTE"],
    thresholds_testados,
    model_name="RF + SMOTE"
)

# REGRESSÃO LOGÍSTICA BALANCEADA

df_lr_bal_thr = avaliar_thresholds(
    y_test,
    y_probas["Logística — class_weight"],
    thresholds_testados,
    model_name="Logística Balanceada"
)

# COMPARAÇÃO FINAL
comparacao_thresholds = pd.concat(
    [df_rf_smote_thr, df_lr_bal_thr],
    ignore_index=True
)

print(comparacao_thresholds)


# PLOT — RECALL x THRESHOLD

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=comparacao_thresholds,
    x="Threshold",
    y="Recall",
    hue="Modelo",
    marker="o"
)

plt.title("Recall × Threshold")
plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.grid(True)
plt.show()

#%% FINAL MODEL TRAINING
# Treinamento final do modelo escolhido para produção:
pipe_log_bal.fit(X_train, y_train)


# Apesar do bom desempenho inicial, o Random Forest apresentou alta sensibilidade ao threshold,
# tornando-o menos adequado para produção

# Modelo final: Regressão Logística com class_weight='balanced'
# Threshold inicial: 0.005
# Recall máximo (não perder fraude)

#%% SAVE MODEL
import joblib
from pathlib import Path

# Raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Pasta models (na raiz)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Caminho final do modelo
MODEL_PATH = MODEL_DIR / "modelo_fraude_producao.pkl"

# Salvando o pipeline treinado
joblib.dump(pipe_log_bal, MODEL_PATH)

print(f"✅ Modelo salvo com sucesso em: {MODEL_PATH}")


# Salvando o modelo treinado para uso em produção

#%% LOAD MODEL
# Carregando o modelo salvo para verificação
from pathlib import Path
import joblib

# Diretório raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "modelo_fraude_producao.pkl"

model = joblib.load(MODEL_PATH)

print("✅ Modelo carregado com sucesso!")
print(model)
# Modelo carregado com sucesso para uso em produção

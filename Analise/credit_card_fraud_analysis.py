#%%

# IMPORTS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.sklearn

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

# Métrica essencial para fraude
from sklearn.metrics import average_precision_score


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "credit-card-fraud-detection"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

#%%
# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================
# Dataset sintético de transações de cartão de crédito

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_card_fraud_synthetic.csv"

df = pd.read_csv(DATA_PATH)
#%%
df.shape
df.info()
#%%
df['Class'].value_counts()
df['Class'].value_counts(normalize=True)

# O dataset apresenta severo desbalanceamento entre classes, 
# exigindo técnicas específicas de avaliação e balanceamento para evitar modelos enviesados.

#%% EXPLORE
# 1. Distribuição da variável alvo
print(df["Class"].value_counts(normalize=True))

sns.countplot(x="Class", data=df)
plt.title("Distribuição das Classes")
plt.show()

# 2. Amount por classe
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Amount por Classe")
plt.show()

# 3. Distribuição temporal
sns.histplot(df["Time"], bins=50)
plt.title("Distribuição de Time")
plt.show()
#%% MODIFY
# separando features e target
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2,
                                                      random_state = 42,
                                                      stratify = y)

# A separação dos dados foi realizada antes de qualquer transformação para evitar vazamento de dados

#%% MODEL

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
# Treinamento do Gradient Boosting como modelo adicional
# para comparação de desempenho.
gb_model = GradientBoostingClassifier(
    random_state=42
)

# XGBOOST — CHALLENGER

# cálculo do scale_pos_weight (fundamental para fraude)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

pipe_xgb = Pipeline(steps=[
    ('model', XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='aucpr',        # métrica correta para fraude
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    ))
])


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
models = {
    "Logística — Baseline": pipe_log_base,
    "Logística — class_weight": pipe_log_bal,
    "Logística — SMOTE": pipe_log_smote,
    "Random Forest — Baseline": pipe_rf_base,
    "Random Forest — class_weight": pipe_rf_bal,
    "Random Forest — SMOTE": pipe_rf_smote
}

# Treinamento e avaliação dos modelos

#%% 
thresholds_testados = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

def run_mlflow_experiment_with_thresholds(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    thresholds
):
    # RUN PAI
    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        for t in thresholds:
            # RUN FILHO
            with mlflow.start_run(
                run_name=f"{model_name} | thr={t}",
                nested=True
            ):
                y_pred = (y_proba >= t).astype(int)

                mlflow.log_param("model", model_name)
                mlflow.log_param("threshold", t)

                mlflow.log_metric(
                    "recall_fraude",
                    recall_score(y_test, y_pred)
                )
                mlflow.log_metric(
                    "precision_fraude",
                    precision_score(y_test, y_pred, zero_division=0)
                )
                mlflow.log_metric(
                    "f1_score",
                    f1_score(y_test, y_pred)
                )
                mlflow.log_metric(
                    "roc_auc",
                    roc_auc_score(y_test, y_proba)
                )
                mlflow.log_metric(
                    "pr_auc",
                    average_precision_score(y_test, y_proba)
                )

                print(f"✔️ {model_name} | threshold={t}")


# Executando experimentos para todos os modelos definidos
mlflow_models = {
    "LogReg - class_weight": pipe_log_bal,
    "LogReg - SMOTE": pipe_log_smote,
    "RF - class_weight": pipe_rf_bal,
    "RF - SMOTE": pipe_rf_smote,
    "Gradient Boosting": gb_model,
     "XGBoost - Challenger": pipe_xgb
}

#%% 
# Executando experimentos MLflow com diferentes thresholds
for model_name, model in mlflow_models.items():
    run_mlflow_experiment_with_thresholds(
        model=model,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        thresholds=thresholds_testados
    )
#%% 
# PRECISION × RECALL CURVE
# Função para plotar curvas Precision × Recall
def plot_precision_recall(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 7))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall (Fraude)")
    plt.ylabel("Precision (Fraude)")
    plt.title("Precision × Recall — Comparação dos Modelos")
    plt.legend()
    plt.grid(True)
    plt.show()
#%%
models_pr = {
    "LogReg - class_weight": pipe_log_bal,
    "XGBoost": pipe_xgb
}
#%% 
# Plotando a curva Precision × Recall para os modelos selecionados
plot_precision_recall(models_pr, X_test, y_test)


#%% CUSTO OPERACIONAL
# Função para calcular o custo operacional baseado em FP e FN
def calcular_custo_operacional(
    model,
    X_test,
    y_test,
    thresholds,
    custo_fn=1000,
    custo_fp=10
):
    resultados = []

    y_proba = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        custo_total = (fn * custo_fn) + (fp * custo_fp)

        resultados.append({
            "threshold": t,
            "FP": fp,
            "FN": fn,
            "custo_total": custo_total
        })

    return pd.DataFrame(resultados)

thresholds_testados = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

custo_logreg = calcular_custo_operacional(
    model=pipe_log_bal,
    X_test=X_test,
    y_test=y_test,
    thresholds=thresholds_testados
)

custo_logreg["modelo"] = "LogReg - class_weight"

custo_xgb = calcular_custo_operacional(
    model=pipe_xgb,
    X_test=X_test,
    y_test=y_test,
    thresholds=thresholds_testados
)

custo_xgb["modelo"] = "XGBoost"
#%%
df_custo = pd.concat([custo_logreg, custo_xgb], ignore_index=True)
#%%
plt.figure(figsize=(10, 6))

for modelo in df_custo["modelo"].unique():
    subset = df_custo[df_custo["modelo"] == modelo]
    plt.plot(
        subset["threshold"],
        subset["custo_total"],
        marker="o",
        label=modelo
    )

plt.xlabel("Threshold")
plt.ylabel("Custo Operacional Total (R$)")
plt.title("Custo Operacional × Threshold — Comparação de Modelos")
plt.legend()
plt.grid(True)
plt.show()

#%% MODEL FINAL PARA PRODUÇÃO
# Definição do threshold para produção baseado na análise de custo operacional

from pathlib import Path
import joblib

best_model = pipe_log_bal
best_threshold = 0.1

FEATURE_COLUMNS = X_train.columns.tolist()

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(FEATURE_COLUMNS, "artifacts/feature_columns.pkl")


# RUN DE PRODUÇÃO

with mlflow.start_run(run_name="Production | LogReg class_weight"):

    best_model.fit(X_train, y_train)

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)

    # ===== MÉTRICAS =====
    mlflow.log_metric("recall_fraude", recall_score(y_test, y_pred))
    mlflow.log_metric("precision_fraude", precision_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    mlflow.log_metric("pr_auc", average_precision_score(y_test, y_proba))

    # ===== PARÂMETROS =====
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("threshold", best_threshold)

    # ===== REGISTRO DO MODELO =====
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="fraud_detection_model"
    )

    # ===== REGISTRAR SCHEMA =====
    mlflow.log_artifact(
        "artifacts/feature_columns.pkl",
        artifact_path="schema"
    )

    # ===== REGISTRAR GRÁFICOS =====
    plt.savefig("precision_recall.png")
    mlflow.log_artifact("precision_recall.png", artifact_path="plots")


# Analisando o custo operacional,
# o modelo de Regressão Logística com class_weight='balanced'

#%% FINAL MODEL TRAINING
# Treinamento final do modelo escolhido para produção:
#pipe_log_bal.fit(X_train, y_train)


# Apesar do bom desempenho inicial, o Random Forest apresentou alta sensibilidade ao threshold,
# tornando-o menos adequado para produção

# Modelo final: Regressão Logística com class_weight='balanced'
# Threshold inicial: 0.005
# Recall máximo (não perder fraude)

#%% SAVE MODEL
#import joblib
#from pathlib import Path

# Raiz do projeto
#BASE_DIR = Path(__file__).resolve().parent.parent

# Pasta models (na raiz)
#MODEL_DIR = BASE_DIR / "models"
#MODEL_DIR.mkdir(exist_ok=True)

# Caminho final do modelo
#MODEL_PATH = MODEL_DIR / "modelo_fraude_producao.pkl"

# Salvando o pipeline treinado
#joblib.dump(pipe_log_bal, MODEL_PATH)

#print(f"✅ Modelo salvo com sucesso em: {MODEL_PATH}")


# Salvando o modelo treinado para uso em produção

#%% LOAD MODEL
# Carregando o modelo salvo para verificação
#from pathlib import Path
#import joblib

# Diretório raiz do projeto
#BASE_DIR = Path(__file__).resolve().parent.parent

#MODEL_PATH = BASE_DIR / "models" / "modelo_fraude_producao.pkl"

#model = joblib.load(MODEL_PATH)

#print("✅ Modelo carregado com sucesso!")
#print(model)
# Modelo carregado com sucesso para uso em produção

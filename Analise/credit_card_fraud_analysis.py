# ============================================================
# DETECÇÃO DE FRAUDE EM CARTÃO DE CRÉDITO
# Versão Atualizada com Dataset Real do Kaggle
# ============================================================

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
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

# Imbalanced learning
from imblearn.over_sampling import SMOTE

# XGBoost
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DO MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "../data/creditcard.csv"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print("="*60)
print("DETECÇÃO DE FRAUDE EM CARTÃO DE CRÉDITO")
print("Dataset: Kaggle Real Credit Card Fraud")
print("="*60)

# ============================================================
# CARREGAMENTO DO DATASET REAL
# ============================================================

print("\n📥 Carregando dataset...")

# DATASET REAL DO KAGGLE
df = pd.read_csv("../data/creditcard.csv")


print(f"✅ Dataset carregado!")
print(f"   Shape: {df.shape}")
print(f"   Colunas: {df.columns.tolist()}")

# ============================================================
# ANÁLISE EXPLORATÓRIA RÁPIDA
# ============================================================

print("\n" + "="*60)
print("ANÁLISE EXPLORATÓRIA")
print("="*60)

print(f"\n📊 Distribuição das Classes:")
print(df['Class'].value_counts())
print(f"\nProporção de fraudes: {df['Class'].mean():.4%}")

print(f"\n📈 Estatísticas de Amount:")
print(df.groupby('Class')['Amount'].describe())

# Verificar valores nulos
print(f"\n🔍 Valores nulos: {df.isnull().sum().sum()}")

# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

print("\n" + "="*60)
print("PREPARAÇÃO DOS DADOS")
print("="*60)

# Separar features e target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split train/test com stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n📦 Split realizado:")
print(f"   Train: {X_train.shape[0]:,} amostras")
print(f"   Test:  {X_test.shape[0]:,} amostras")
print(f"   Fraudes no train: {y_train.sum()} ({y_train.mean():.4%})")
print(f"   Fraudes no test:  {y_test.sum()} ({y_test.mean():.4%})")

# ============================================================
# DEFINIÇÃO DOS MODELOS
# ============================================================

print("\n" + "="*60)
print("DEFINIÇÃO DOS MODELOS")
print("="*60)

# Cálculo do scale_pos_weight para XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nscale_pos_weight para XGBoost: {scale_pos_weight:.1f}")

# Dicionário de modelos
models = {
    "LogReg - Baseline": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    
    "LogReg - class_weight": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        ))
    ]),
    
    "LogReg - SMOTE": Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ]),
    
    "RF - class_weight": Pipeline([
        ('model', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ]),
    
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='aucpr'
        ))
    ])
}

# ============================================================
# TREINAMENTO E AVALIAÇÃO
# ============================================================

print("\n" + "="*60)
print("TREINAMENTO DOS MODELOS")
print("="*60)

results = []
thresholds = [0.01, 0.05, 0.1, 0.3, 0.5]

for model_name, model in models.items():
    print(f"\n🔄 Treinando: {model_name}")
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Predições
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Testar diferentes thresholds
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calcular métricas
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        results.append({
            'Model': model_name,
            'Threshold': threshold,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc
        })
        
        # Log no MLflow
        with mlflow.start_run(run_name=f"{model_name} | thr={threshold}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)

# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "="*60)
print("RESULTADOS CONSOLIDADOS")
print("="*60)

df_results = pd.DataFrame(results)

# Encontrar melhor combinação para cada métrica
print("\n🏆 MELHORES RESULTADOS:")

best_recall = df_results.loc[df_results['Recall'].idxmax()]
print(f"\nMelhor Recall: {best_recall['Recall']:.2%}")
print(f"  Modelo: {best_recall['Model']}")
print(f"  Threshold: {best_recall['Threshold']}")
print(f"  Precision: {best_recall['Precision']:.2%}")
print(f"  F1-Score: {best_recall['F1-Score']:.4f}")

best_f1 = df_results.loc[df_results['F1-Score'].idxmax()]
print(f"\nMelhor F1-Score: {best_f1['F1-Score']:.4f}")
print(f"  Modelo: {best_f1['Model']}")
print(f"  Threshold: {best_f1['Threshold']}")
print(f"  Recall: {best_f1['Recall']:.2%}")
print(f"  Precision: {best_f1['Precision']:.2%}")

best_roc = df_results.loc[df_results['ROC-AUC'].idxmax()]
print(f"\nMelhor ROC-AUC: {best_roc['ROC-AUC']:.4f}")
print(f"  Modelo: {best_roc['Model']}")

# Salvar resultados
df_results.to_csv('model_comparison_results.csv', index=False)
print("\n✅ Resultados salvos em: model_comparison_results.csv")

# ============================================================
# MODELO FINAL PARA PRODUÇÃO
# ============================================================

print("\n" + "="*60)
print("MODELO DE PRODUÇÃO")
print("="*60)

# Escolher melhor modelo (maior F1-Score)
best_model_name = best_f1['Model']
best_threshold = best_f1['Threshold']

print(f"\n🎯 Modelo selecionado: {best_model_name}")
print(f"   Threshold: {best_threshold}")

# Retreinar modelo escolhido
final_model = models[best_model_name]
final_model.fit(X_train, y_train)

# Predições finais
y_proba_final = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_proba_final >= best_threshold).astype(int)

# Métricas finais
print("\n📊 MÉTRICAS FINAIS:")
print(f"   Recall:    {recall_score(y_test, y_pred_final):.2%}")
print(f"   Precision: {precision_score(y_test, y_pred_final):.2%}")
print(f"   F1-Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_proba_final):.4f}")
print(f"   PR-AUC:    {average_precision_score(y_test, y_proba_final):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n📋 CONFUSION MATRIX:")
print(f"   True Negatives:  {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]}")
print(f"   True Positives:  {cm[1,1]}")

# Análise de custo
custo_fn = 1000  # custo de perder uma fraude
custo_fp = 10    # custo de bloquear cliente legítimo
custo_total = (cm[1,0] * custo_fn) + (cm[0,1] * custo_fp)

print(f"\n💰 IMPACTO DE NEGÓCIO:")
print(f"   Fraudes detectadas: {cm[1,1]} de {cm[1,1] + cm[1,0]}")
print(f"   Taxa de detecção: {cm[1,1]/(cm[1,1] + cm[1,0]):.1%}")
print(f"   Custo operacional: R$ {custo_total:,.2f}")

# ============================================================
# VISUALIZAÇÕES
# ============================================================

print("\n" + "="*60)
print("GERANDO VISUALIZAÇÕES")
print("="*60)

# Criar pasta para plots
from pathlib import Path
Path("plots").mkdir(exist_ok=True)

# 1. Comparação de modelos por threshold
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for model in df_results['Model'].unique():
    data = df_results[df_results['Model'] == model]
    axes[0, 0].plot(data['Threshold'], data['Recall'], marker='o', label=model)
    axes[0, 1].plot(data['Threshold'], data['Precision'], marker='s', label=model)
    axes[1, 0].plot(data['Threshold'], data['F1-Score'], marker='^', label=model)
    axes[1, 1].plot(data['Threshold'], data['ROC-AUC'], marker='d', label=model)

axes[0, 0].set_title('Recall vs Threshold', fontsize=14, weight='bold')
axes[0, 0].set_xlabel('Threshold')
axes[0, 0].set_ylabel('Recall')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_title('Precision vs Threshold', fontsize=14, weight='bold')
axes[0, 1].set_xlabel('Threshold')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].set_title('F1-Score vs Threshold', fontsize=14, weight='bold')
axes[1, 0].set_xlabel('Threshold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].set_title('ROC-AUC vs Threshold', fontsize=14, weight='bold')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('ROC-AUC')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: plots/model_comparison.png")

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}\nThreshold: {best_threshold}', 
          fontsize=16, weight='bold')
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: plots/confusion_matrix.png")

plt.show()

print("\n" + "="*60)
print("PROCESSO CONCLUÍDO!")
print("="*60)
print("\n✅ Todos os resultados foram salvos")
print("✅ Verifique a pasta 'plots/' para visualizações")
print("✅ Verifique 'model_comparison_results.csv' para métricas")
print("✅ Acesse o MLflow UI para explorar experimentos")
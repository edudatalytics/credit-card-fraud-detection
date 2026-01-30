# ============================================================
# APP DE DETECÇÃO DE FRAUDE EM CARTÃO DE CRÉDITO
# Interface Web com Streamlit
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Adicionar diretório ao path para importar predict
sys.path.append(str(Path(__file__).parent))

# Importar função de predição
from predict import predict_fraud, THRESHOLD

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Detecção de Fraude",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO CSS CUSTOMIZADO - CORRIGIDO PARA LEGIBILIDADE
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown('<h1 class="main-header">🔒 Sistema de Detecção de Fraude</h1>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR - INFORMAÇÕES DO MODELO
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=100)
    
    st.markdown("### 📊 Informações do Modelo")
    st.markdown(f"""
    **Modelo:** Random Forest  
    **Threshold:** {THRESHOLD}  
    **ROC-AUC:** 97.7%  
    **Recall:** 82.7%  
    **Precision:** 81.8%  
    **F1-Score:** 0.82
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Como Funciona")
    st.markdown("""
    1. **Inserir dados** da transação
    2. **Analisar** com IA
    3. **Receber resultado** em segundos
    
    O modelo foi treinado com 284 mil transações reais.
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Interpretação")
    st.markdown(f"""
    - **Prob < {THRESHOLD*100:.0f}%**: Transação legítima
    - **Prob ≥ {THRESHOLD*100:.0f}%**: Possível fraude
    
    O sistema detecta **82.7%** das fraudes reais com apenas **18.2%** de falsos positivos.
    """)

# ============================================================
# TABS PRINCIPAIS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Análise Individual", 
    "📊 Análise em Lote", 
    "📈 Dashboard",
    "ℹ️ Sobre o Modelo"
])

# ============================================================
# TAB 1: ANÁLISE INDIVIDUAL
# ============================================================

with tab1:
    st.header("🔍 Análise de Transação Individual")
    
    st.markdown("""
    Insira os dados de uma transação para verificar se ela é potencialmente fraudulenta.
    **Nota:** Apenas `Amount` e `Time` são obrigatórios. Outras features são opcionais.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Dados da Transação")
        
        # Campos principais
        amount = st.number_input(
            "💰 Valor da Transação (R$)",
            min_value=0.0,
            max_value=100000.0,
            value=150.0,
            step=10.0,
            help="Valor em reais da transação"
        )
        
        time = st.number_input(
            "⏰ Tempo (segundos desde primeira transação)",
            min_value=0,
            max_value=200000,
            value=12345,
            step=100,
            help="Tempo em segundos desde a primeira transação do dataset"
        )
        
        st.markdown("---")
        st.markdown("#### Features Adicionais (Opcional)")
        st.markdown("*Valores das componentes PCA (V1-V28)*")
        
        # Expandir para features opcionais
        with st.expander("➕ Adicionar features V1-V28"):
            v_features = {}
            
            # Criar grid 4x7 para V1-V28
            for i in range(0, 28, 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    if i + j < 28:
                        v_num = i + j + 1
                        v_features[f'V{v_num}'] = col.number_input(
                            f'V{v_num}',
                            value=0.0,
                            step=0.1,
                            format="%.4f",
                            key=f'v{v_num}'
                        )
    
    with col2:
        st.subheader("🎯 Resultado da Análise")
        
        # Botão de análise
        if st.button("🔍 ANALISAR TRANSAÇÃO", type="primary", use_container_width=True):
            
            # Preparar input
            transaction = {
                'Amount': amount,
                'Time': time
            }
            
            # Adicionar V features se preenchidas
            if 'v_features' in locals():
                transaction.update(v_features)
            
            # Fazer predição
            with st.spinner("🔄 Analisando transação..."):
                result = predict_fraud(transaction)
                
                prob = result['prob_fraude'].iloc[0]
                is_fraud = result['fraude_predita'].iloc[0]
            
            # ============================================================
            # EXIBIR RESULTADO - VERSÃO CORRIGIDA COM LEGIBILIDADE
            # ============================================================
            
            st.markdown("---")
            
            if is_fraud == 1:
                # FRAUDE DETECTADA - TEXTO ESCURO EM FUNDO VERMELHO CLARO
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
                    padding: 2rem;
                    border-radius: 1rem;
                    border: 3px solid #c62828;
                    box-shadow: 0 4px 6px rgba(198, 40, 40, 0.3);
                    margin: 1rem 0;
                ">
                    <h2 style="
                        color: #b71c1c;
                        margin: 0 0 0.8rem 0;
                        font-size: 2rem;
                        font-weight: bold;
                        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
                    ">
                        🚨 ALERTA DE FRAUDE DETECTADO
                    </h2>
                    <h3 style="
                        color: #c62828;
                        margin: 0.5rem 0;
                        font-size: 1.6rem;
                        font-weight: 600;
                    ">
                        Probabilidade: {prob:.1%}
                    </h3>
                    <p style="
                        color: #212121;
                        margin: 0.5rem 0;
                        font-size: 1.2rem;
                        font-weight: 500;
                    ">
                        <strong>⚠️ Recomendação:</strong> Bloquear transação e contactar cliente imediatamente
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart para fraude
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={
                        'text': "Probabilidade de Fraude",
                        'font': {'size': 24, 'color': '#212121', 'family': 'Arial Black'}
                    },
                    number={
                        'suffix': "%",
                        'font': {'size': 50, 'color': '#d32f2f', 'family': 'Arial Black'}
                    },
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': '#212121'},
                        'bar': {'color': "#d32f2f", 'thickness': 0.8},
                        'steps': [
                            {'range': [0, 30], 'color': "#a5d6a7"},
                            {'range': [30, 50], 'color': "#fff176"},
                            {'range': [50, 70], 'color': "#ffb74d"},
                            {'range': [70, 100], 'color': "#ef5350"}
                        ],
                        'threshold': {
                            'line': {'color': "#212121", 'width': 5},
                            'thickness': 0.8,
                            'value': THRESHOLD * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=350,
                    font={'color': '#212121', 'family': 'Arial', 'size': 14},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # TRANSAÇÃO LEGÍTIMA - TEXTO ESCURO EM FUNDO VERDE CLARO
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
                    padding: 2rem;
                    border-radius: 1rem;
                    border: 3px solid #388e3c;
                    box-shadow: 0 4px 6px rgba(56, 142, 60, 0.3);
                    margin: 1rem 0;
                ">
                    <h2 style="
                        color: #1b5e20;
                        margin: 0 0 0.8rem 0;
                        font-size: 2rem;
                        font-weight: bold;
                        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
                    ">
                        ✅ TRANSAÇÃO LEGÍTIMA
                    </h2>
                    <h3 style="
                        color: #2e7d32;
                        margin: 0.5rem 0;
                        font-size: 1.6rem;
                        font-weight: 600;
                    ">
                        Probabilidade de fraude: {prob:.1%}
                    </h3>
                    <p style="
                        color: #212121;
                        margin: 0.5rem 0;
                        font-size: 1.2rem;
                        font-weight: 500;
                    ">
                        <strong>✓ Recomendação:</strong> Aprovar transação
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart para legítima
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={
                        'text': "Probabilidade de Fraude",
                        'font': {'size': 24, 'color': '#212121', 'family': 'Arial Black'}
                    },
                    number={
                        'suffix': "%",
                        'font': {'size': 50, 'color': '#388e3c', 'family': 'Arial Black'}
                    },
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': '#212121'},
                        'bar': {'color': "#388e3c", 'thickness': 0.8},
                        'steps': [
                            {'range': [0, 30], 'color': "#a5d6a7"},
                            {'range': [30, 50], 'color': "#fff176"},
                            {'range': [50, 70], 'color': "#ffb74d"},
                            {'range': [70, 100], 'color': "#ef5350"}
                        ],
                        'threshold': {
                            'line': {'color': "#212121", 'width': 5},
                            'thickness': 0.8,
                            'value': THRESHOLD * 100
                        }
                    }
                ))
                fig.update_layout(
                    height=350,
                    font={'color': '#212121', 'family': 'Arial', 'size': 14},
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detalhes técnicos
            with st.expander("🔬 Detalhes Técnicos"):
                st.json({
                    "Probabilidade": f"{prob:.4f}",
                    "Threshold": THRESHOLD,
                    "Predição": "FRAUDE" if is_fraud == 1 else "LEGÍTIMA",
                    "Confiança": f"{max(prob, 1-prob):.2%}",
                    "Modelo": "Random Forest com class_weight='balanced'"
                })
            
            # Informações do input
            with st.expander("📋 Dados da Transação"):
                st.write(transaction)

# ============================================================
# TAB 2: ANÁLISE EM LOTE
# ============================================================

with tab2:
    st.header("📊 Análise em Lote")
    
    st.markdown("""
    Faça upload de um arquivo CSV com múltiplas transações para análise em massa.
    
    **Formato esperado:** CSV com colunas `Time`, `Amount`, e opcionalmente `V1`-`V28`
    """)
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "📁 Selecione um arquivo CSV",
        type=['csv'],
        help="Arquivo CSV com transações para análise"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar CSV
            df_input = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Arquivo carregado: {len(df_input)} transações")
            
            # Mostrar preview
            with st.expander("👁️ Preview dos Dados"):
                st.dataframe(df_input.head(10))
            
            # Botão de análise
            if st.button("🔍 ANALISAR TODAS AS TRANSAÇÕES", type="primary"):
                
                with st.spinner(f"🔄 Analisando {len(df_input)} transações..."):
                    # Fazer predições
                    results = []
                    for idx, row in df_input.iterrows():
                        transaction = row.to_dict()
                        result = predict_fraud(transaction)
                        results.append({
                            'ID': idx,
                            'Amount': transaction.get('Amount', 0),
                            'Probabilidade_Fraude': result['prob_fraude'].iloc[0],
                            'Fraude_Predita': result['fraude_predita'].iloc[0]
                        })
                    
                    df_results = pd.DataFrame(results)
                
                st.success("✅ Análise concluída!")
                
                # Estatísticas
                n_total = len(df_results)
                n_frauds = df_results['Fraude_Predita'].sum()
                n_legit = n_total - n_frauds
                fraud_rate = n_frauds / n_total * 100
                
                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total de Transações", n_total)
                
                with col2:
                    st.metric("🚨 Fraudes Detectadas", n_frauds, 
                             delta=f"{fraud_rate:.1f}%", delta_color="inverse")
                
                with col3:
                    st.metric("✅ Transações Legítimas", n_legit)
                
                with col4:
                    avg_prob = df_results['Probabilidade_Fraude'].mean()
                    st.metric("📈 Probabilidade Média", f"{avg_prob:.1%}")
                
                # Gráficos
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pizza
                    fig_pie = px.pie(
                        values=[n_legit, n_frauds],
                        names=['Legítimas', 'Fraudes'],
                        title='Distribuição de Transações',
                        color_discrete_sequence=['#4caf50', '#f44336']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Histograma
                    fig_hist = px.histogram(
                        df_results,
                        x='Probabilidade_Fraude',
                        nbins=50,
                        title='Distribuição de Probabilidades',
                        labels={'Probabilidade_Fraude': 'Probabilidade de Fraude'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_hist.add_vline(
                        x=THRESHOLD, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Threshold ({THRESHOLD})"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Tabela de resultados
                st.markdown("---")
                st.subheader("📋 Resultados Detalhados")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    show_filter = st.selectbox(
                        "Filtrar por:",
                        ["Todas", "Apenas Fraudes", "Apenas Legítimas"]
                    )
                
                with col2:
                    sort_by = st.selectbox(
                        "Ordenar por:",
                        ["ID", "Probabilidade (Maior)", "Probabilidade (Menor)", "Amount"]
                    )
                
                # Aplicar filtros
                df_display = df_results.copy()
                
                if show_filter == "Apenas Fraudes":
                    df_display = df_display[df_display['Fraude_Predita'] == 1]
                elif show_filter == "Apenas Legítimas":
                    df_display = df_display[df_display['Fraude_Predita'] == 0]
                
                if sort_by == "Probabilidade (Maior)":
                    df_display = df_display.sort_values('Probabilidade_Fraude', ascending=False)
                elif sort_by == "Probabilidade (Menor)":
                    df_display = df_display.sort_values('Probabilidade_Fraude', ascending=True)
                elif sort_by == "Amount":
                    df_display = df_display.sort_values('Amount', ascending=False)
                
                # Adicionar coluna de status
                df_display['Status'] = df_display['Fraude_Predita'].apply(
                    lambda x: "🚨 FRAUDE" if x == 1 else "✅ LEGÍTIMA"
                )
                
                # Formatar probabilidade
                df_display['Probabilidade_Fraude'] = df_display['Probabilidade_Fraude'].apply(
                    lambda x: f"{x:.2%}"
                )
                
                # Formatar Amount
                df_display['Amount'] = df_display['Amount'].apply(lambda x: f"R$ {x:.2f}")
                
                st.dataframe(
                    df_display[['ID', 'Amount', 'Probabilidade_Fraude', 'Status']],
                    use_container_width=True,
                    height=400
                )
                
                # Download dos resultados
                st.download_button(
                    label="📥 Download Resultados (CSV)",
                    data=df_results.to_csv(index=False).encode('utf-8'),
                    file_name=f"fraud_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {e}")
            st.info("Verifique se o arquivo CSV está no formato correto.")

# ============================================================
# TAB 3: DASHBOARD
# ============================================================

with tab3:
    st.header("📈 Dashboard do Sistema")
    
    st.info("💡 Esta seção mostra estatísticas do modelo com dados sintéticos de exemplo.")
    
    # Gerar dados sintéticos para demonstração
    np.random.seed(42)
    n_samples = 1000
    
    # Simular transações
    amounts = np.random.lognormal(4, 1.5, n_samples)
    times = np.random.randint(0, 172800, n_samples)
    
    # Simular probabilidades (com viés para distribuição realista)
    probs = np.random.beta(2, 20, n_samples)  # Maioria baixa, algumas altas
    predictions = (probs >= THRESHOLD).astype(int)
    
    # Criar DataFrame
    df_demo = pd.DataFrame({
        'Amount': amounts,
        'Time': times,
        'Probability': probs,
        'Prediction': predictions
    })
    
    # Métricas principais
    st.subheader("📊 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Threshold do Modelo",
            f"{THRESHOLD*100:.0f}%",
            help="Limiar de decisão otimizado"
        )
    
    with col2:
        st.metric(
            "🎯 ROC-AUC",
            "97.7%",
            delta="+2.7%",
            help="Área sob a curva ROC"
        )
    
    with col3:
        st.metric(
            "🎯 Recall",
            "82.7%",
            help="Taxa de detecção de fraudes"
        )
    
    with col4:
        st.metric(
            "🎯 Precision",
            "81.8%",
            help="Precisão das predições"
        )
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribuição de Probabilidades")
        fig1 = px.histogram(
            df_demo,
            x='Probability',
            nbins=50,
            title='Distribuição de Probabilidades de Fraude',
            labels={'Probability': 'Probabilidade'},
            color_discrete_sequence=['#1f77b4']
        )
        fig1.add_vline(
            x=THRESHOLD,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("💰 Valor vs Probabilidade")
        fig2 = px.scatter(
            df_demo.sample(200),  # Sample para performance
            x='Amount',
            y='Probability',
            color='Prediction',
            title='Relação entre Valor e Probabilidade de Fraude',
            labels={
                'Amount': 'Valor da Transação (R$)',
                'Probability': 'Probabilidade de Fraude',
                'Prediction': 'Predição'
            },
            color_discrete_map={0: '#4caf50', 1: '#f44336'}
        )
        fig2.add_hline(
            y=THRESHOLD,
            line_dash="dash",
            line_color="red"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Matriz de confusão (simulada)
    st.markdown("---")
    st.subheader("📋 Matriz de Confusão (Dados de Teste)")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Dados reais do seu modelo
        confusion_data = {
            'Predito Legítimo': [56846, 17],
            'Predito Fraude': [18, 81]
        }
        
        df_confusion = pd.DataFrame(
            confusion_data,
            index=['Real Legítimo', 'Real Fraude']
        )
        
        fig_conf = px.imshow(
            df_confusion,
            text_auto=True,
            aspect="auto",
            title="Matriz de Confusão",
            color_continuous_scale='Blues',
            labels={'x': 'Predição', 'y': 'Real'}
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        st.markdown("#### ✅ Acertos")
        st.metric("True Negatives", "56,846")
        st.metric("True Positives", "81")
        st.markdown("**Total:** 56,927")
    
    with col3:
        st.markdown("#### ❌ Erros")
        st.metric("False Positives", "18")
        st.metric("False Negatives", "17")
        st.markdown("**Total:** 35")

# ============================================================
# TAB 4: SOBRE O MODELO
# ============================================================

with tab4:
    st.header("ℹ️ Sobre o Modelo")
    
    st.markdown("""
    ## 🤖 Detecção de Fraude com Machine Learning
    
    Este sistema utiliza um modelo de **Random Forest** treinado com 284 mil transações 
    reais de cartão de crédito para identificar padrões de fraude.
    
    ### 📊 Especificações Técnicas
    
    **Modelo:** Random Forest Classifier  
    **Técnica de Balanceamento:** class_weight='balanced'  
    **Threshold:** 0.5 (otimizado via análise de custo-benefício)  
    **Dataset:** 284,807 transações (0.17% fraudes)  
    **Features:** 30 variáveis (Time, Amount, V1-V28)
    
    ### 🎯 Métricas de Performance
    
    | Métrica | Valor | Interpretação |
    |---------|-------|---------------|
    | **ROC-AUC** | 97.7% | Excelente capacidade de discriminação |
    | **Recall** | 82.7% | Detecta 83 de cada 100 fraudes |
    | **Precision** | 81.8% | 82% das alertas são fraudes reais |
    | **F1-Score** | 0.82 | Ótimo equilíbrio precision/recall |
    
    ### 💰 Impacto de Negócio
    
    - **Fraudes Detectadas:** 81 de 98 (82.7%)
    - **Falsos Positivos:** Apenas 18 em 56,864 transações legítimas (0.03%)
    - **Economia Estimada:** R$ 80,820 por período
    - **ROI:** 15:1 (para cada R$ 1 investido, retorno de R$ 15)
    
    ### 🔬 Metodologia
    
    1. **Análise Exploratória:** Identificação de padrões e desbalanceamento
    2. **Feature Engineering:** Utilização de componentes PCA (V1-V28)
    3. **Balanceamento:** class_weight para lidar com 99.83% de transações legítimas
    4. **Otimização:** Análise de múltiplos thresholds (0.01 a 0.5)
    5. **Validação:** Estratificação para manter proporção de classes
    
    ### ⚠️ Limitações
    
    - Modelo treinado em dados históricos (pode haver drift)
    - Features V1-V28 são resultado de PCA (sem interpretabilidade direta)
    - Necessita retreinamento periódico (recomendado: mensal)
    
    ### 🔄 Próximos Passos
    
    - [ ] Monitoramento de drift do modelo
    - [ ] A/B testing com modelo ensemble
    - [ ] Retreinamento automático
    - [ ] Explicabilidade com SHAP values
    
    ### 👨‍💻 Desenvolvedor
    
    **Eduardo Matos**  
    Cientista de Dados  
    [LinkedIn](https://www.linkedin.com/in/matos-eduardo) | [GitHub](https://github.com/edudatalytics)
    
    ---
    
    *Sistema desenvolvido como projeto de portfólio em Ciência de Dados*
    """)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔒 Sistema de Detecção de Fraude v1.0</p>
    <p>Desenvolvido por Eduardo Matos | 2026</p>
    <p>Modelo: Random Forest | ROC-AUC: 97.7% | Threshold: 0.5</p>
</div>
""", unsafe_allow_html=True)
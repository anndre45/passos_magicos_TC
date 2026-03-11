import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("📊 Monitoramento do Modelo - Passos Mágicos")

predictions_file = Path("logs/predictions.csv")
api_log_file = Path("logs/api.log")
system_log_file = Path("logs/system.log")

# =========================
# Carregar predições
# =========================

if predictions_file.exists():

    df = pd.read_csv(predictions_file)
    # =========================
    # Métricas principais
    # =========================
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Total de Predições",
        len(df)
    )
    
    col2.metric(
        "Confiança Média",
        round(df["confidence"].mean(), 2)
    )
    
    col3.metric(
        "Última Classe Prevista",
        df["prediction"].iloc[-1]
    )
    col1, col2 = st.columns(2)

    # ---------------------
    # gráfico 1
    # ---------------------

    with col1:

        st.subheader("Distribuição das Predições")

        fig, ax = plt.subplots(figsize=(4,3))

        df["prediction"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.tick_params(axis='x', labelrotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    # ---------------------
    # gráfico 2
    # ---------------------

    with col2:

        st.subheader("Distribuição da Confiança")

        fig2, ax2 = plt.subplots(figsize=(4,3))

        ax2.hist(df["confidence"], bins=10)

        fig2.tight_layout()

        st.pyplot(fig2)

    # ---------------------
    # tabela
    # ---------------------

    st.subheader("Últimas Predições")

    st.dataframe(df.tail(20))

else:

    st.warning("Arquivo predictions.csv não encontrado")

# =========================
# Logs da API
# =========================

st.subheader("📡 Logs da API")

if api_log_file.exists():

    with open(api_log_file) as f:
        logs = f.readlines()

    st.text("".join(logs[-20:]))

else:

    st.warning("api.log não encontrado")

# =========================
# Logs do Sistema
# =========================

st.subheader("⚙️ Logs do Sistema")

if system_log_file.exists():

    with open(system_log_file) as f:
        logs = f.readlines()

    st.text("".join(logs[-20:]))

else:

    st.warning("system.log não encontrado")
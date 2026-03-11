import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from monitoring.drift_monitor import detect_drift

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
st.subheader("📉 Monitoramento de Data Drift")

try:

    drift_report = detect_drift()

    if drift_report:

        drift_data = []

        for feature, result in drift_report.items():

            drift_data.append({
                "feature": feature,
                "p_value": result["p_value"],
                "drift_detected": result["drift_detected"]
            })

        drift_df = pd.DataFrame(drift_data)

        st.dataframe(drift_df)

        # gráfico
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6,3))

        ax.bar(
            drift_df["feature"],
            drift_df["p_value"]
        )

        ax.axhline(
            y=0.05,
            linestyle="--"
        )

        ax.set_ylabel("p-value")
        ax.set_title("Detecção de Drift")

        st.pyplot(fig)

        # alerta
        if drift_df["drift_detected"].any():

            st.error("⚠ Drift detectado em algumas features!")

        else:

            st.success("✅ Nenhum drift detectado")

    else:

        st.info("Dados insuficientes para calcular drift.")

except Exception as e:

    st.warning("Não foi possível calcular drift.")

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
import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Monitoreo de Modelos", layout="wide")
st.title("Visualización de Rendimiento de Modelos")

metrics_dir = "metrics"
metric_files = [f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")]

if not metric_files:
    st.warning("No se encontraron archivos de métricas en la carpeta 'metrics/'.")
else:
    modelos = [f.replace("_metrics.json", "") for f in metric_files]
    modelo_seleccionado = st.selectbox("Selecciona un modelo", modelos)

    with open(os.path.join(metrics_dir, f"{modelo_seleccionado}_metrics.json")) as f:
        metrics = json.load(f)
        
    st.subheader("📌 Métricas de Evaluación")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['precision']:.2f}")
    col3.metric("Recall", f"{metrics['recall']:.2f}")
    st.metric("F1 Score", f"{metrics['f1_score']:.2f}")

    st.subheader("Matriz de Confusión")
    fig, ax = plt.subplots()
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)
        

    st.subheader("📊 Comparación entre Modelos")
    comparar = st.checkbox("Mostrar comparación general de métricas entre modelos")

    if comparar:
        modelos_data = []
        for file in metric_files:
            nombre = file.replace("_metrics.json", "")
            with open(os.path.join(metrics_dir, file)) as f:
                m = json.load(f)
                modelos_data.append({
                    "Modelo": nombre,
                    "Accuracy": m.get("accuracy", 0),
                    "Precision": m.get("precision", 0),
                    "Recall": m.get("recall", 0),
                    "F1 Score": m.get("f1_score", 0)
                })

        df_comparacion = pd.DataFrame(modelos_data).set_index("Modelo")

        st.dataframe(df_comparacion.style.highlight_max(axis=0, color="lightgreen"))

        for metrica in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            fig, ax = plt.subplots()
            ordenado = df_comparacion.sort_values(by=metrica)
            ax.barh(ordenado.index, ordenado[metrica], color="skyblue")
            ax.set_title(f"{metrica} por Modelo")
            ax.set_xlabel(metrica)
            ax.set_ylabel("Modelo")
            st.pyplot(fig)
            
        
    st.title("📡 Estadísticas de Inferencia en Producción")

    log_path = "logs/inferencia.jsonl"
    if not os.path.exists(log_path):
        st.info("Aún no hay registros de inferencia.")
    else:
        with open(log_path, "r") as f:
            registros = [json.loads(line) for line in f if line.strip()]
        df_logs = pd.DataFrame(registros)
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        df_logs = df_logs.sort_values("timestamp", ascending=False)

        st.subheader("Últimas Inferencias Realizadas")
        st.dataframe(df_logs[["timestamp", "modelo", "latencia_ms", "prediccion"]].head(10))

        st.subheader("Distribución de Latencias")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_logs["latencia_ms"], bins=20, kde=True, ax=ax1, color="orchid")
        ax1.set_xlabel("Latencia (ms)")
        ax1.set_ylabel("Frecuencia")
        st.pyplot(fig1)

        st.subheader("Distribución de Clases Predichas")
        pred_labels = df_logs["prediccion"].apply(lambda x: x[0] if isinstance(x, list) else x)
        fig2, ax2 = plt.subplots()
        pred_labels.value_counts().plot(kind="bar", ax=ax2, color="lightgreen")
        ax2.set_xlabel("Clase Predicha")
        ax2.set_ylabel("Cantidad")
        st.pyplot(fig2)


import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Monitoreo de Modelos", layout="wide")
st.title("Plataforma de Aprendizaje Supervisado Distribuido")

#rutas 
metrics_dir = "metrics"
log_path = "logs/inferencia.jsonl"
metric_files = [f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")] if os.path.exists(metrics_dir) else []

tabs = st.tabs(["🏋️‍♀️ Entrenamiento", "📡 Inferencia en Producción", "📈 Estadísticas Avanzadas"])

with tabs[0]:
    st.header("🏋️‍♀️ Métricas de Rendimiento durante el Entrenamiento")

    if not metric_files:
        st.warning("No se encontraron archivos de métricas en la carpeta 'metrics/'.")
    else:
        modelos = [f.replace("_metrics.json", "") for f in metric_files]
        modelo_seleccionado = st.selectbox("Selecciona un modelo", modelos)

        with open(os.path.join(metrics_dir, f"{modelo_seleccionado}_metrics.json")) as f:
            metrics = json.load(f)

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

with tabs[1]:
    st.header("📡 Estadísticas de Inferencia en Producción")

    if not os.path.exists(log_path):
        st.info("Aún no hay registros de inferencia.")
    else:
        with open(log_path, "r") as f:
            registros = [json.loads(line) for line in f if line.strip()]
        df_logs = pd.DataFrame(registros)
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        df_logs = df_logs.sort_values("timestamp", ascending=False)
        df_logs["pred_clase"] = df_logs["prediccion"].apply(lambda x: x[0] if isinstance(x, list) else x)

        st.subheader("Últimas Inferencias Realizadas")
        st.dataframe(df_logs[["timestamp", "modelo", "latencia_ms", "prediccion"]].head(10))

        st.subheader("Distribución de Latencias")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_logs["latencia_ms"], bins=20, kde=True, ax=ax1, color="orchid")
        ax1.set_xlabel("Latencia (ms)")
        ax1.set_ylabel("Frecuencia")
        st.pyplot(fig1)

        st.subheader("Distribución de Clases Predichas")
        fig2, ax2 = plt.subplots()
        df_logs["pred_clase"].value_counts().plot(kind="bar", ax=ax2, color="lightgreen")
        ax2.set_xlabel("Clase Predicha")
        ax2.set_ylabel("Cantidad")
        st.pyplot(fig2)
    
with tabs[2]:
    st.header("📈 Comparativas y Tendencias")

    if not os.path.exists(log_path):
        st.warning("No hay registros de inferencia para análisis avanzado.")
    else:
        st.subheader("Latencia Promedio por Modelo")
        lat_prom = df_logs.groupby("modelo")["latencia_ms"].mean().sort_values()
        fig, ax = plt.subplots()
        ax.barh(lat_prom.index, lat_prom.values, color="mediumslateblue")
        ax.set_xlabel("Latencia Promedio (ms)")
        ax.set_ylabel("Modelo")
        ax.set_title("Latencia Promedio por Modelo")
        st.pyplot(fig)

        st.subheader("Evolución de la Latencia en el Tiempo")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_logs, x="timestamp", y="latencia_ms", hue="modelo", marker="o", ax=ax)
        ax.set_ylabel("Latencia (ms)")
        ax.set_xlabel("Tiempo")
        plt.xticks(rotation=45)
        ax.set_title("Latencia por Inferencia")
        ax.legend(title="Modelo")
        st.pyplot(fig)

        st.subheader("Proporción de Clases Predichas por Modelo")
        conteo = df_logs.groupby(["modelo", "pred_clase"]).size().unstack(fill_value=0)
        conteo_percent = conteo.div(conteo.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots()
        conteo_percent.plot(kind="barh", stacked=True, ax=ax, colormap="Set2", edgecolor="black")
        ax.set_xlabel("Porcentaje (%)")
        ax.set_ylabel("Modelo")
        ax.set_title("Distribución de Clases Predichas por Modelo")
        st.pyplot(fig)

        st.subheader("Comparativa Global de Modelos")
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
        st.dataframe(df_comparacion.style.highlight_max(axis=0, color="lightblue"))

        for metrica in ["Accuracy", "Precision", "Recall", "F1 Score"]:
            fig, ax = plt.subplots()
            ordenado = df_comparacion.sort_values(by=metrica)
            ax.barh(ordenado.index, ordenado[metrica], color="cornflowerblue")
            ax.set_title(f"{metrica} por Modelo")
            st.pyplot(fig)


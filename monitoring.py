import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


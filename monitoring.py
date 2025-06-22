import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📈 Métricas de Entrenamiento")

metrics_dir = "metrics"
modelos = [f.replace("_metrics.json", "") for f in os.listdir(metrics_dir) if f.endswith(".json")]
modelo_seleccionado = st.selectbox("Selecciona un modelo", modelos)

with open(f"{metrics_dir}/{modelo_seleccionado}_metrics.json") as f:
    metrics = json.load(f)

st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
st.metric("Precision", f"{metrics['precision']:.2f}")
st.metric("Recall", f"{metrics['recall']:.2f}")
st.metric("F1 Score", f"{metrics['f1_score']:.2f}")

# Matriz de confusión
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

import streamlit as st
import pandas as pd

st.title("Dashboard de Modelos")
st.subheader("Estadísticas de Entrenamiento")
st.metric(label="Accuracy", value="0.92")
st.line_chart(pd.DataFrame({"accuracy": [0.85, 0.88, 0.91, 0.92]}))

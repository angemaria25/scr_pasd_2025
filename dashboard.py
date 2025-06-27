import streamlit as st
import requests
import pandas as pd
import json
import base64
import io
import time

# URL base de tu API de modelos (FastAPI)
# Usamos el nombre del servicio Docker 'model-api' y el puerto 8000
FASTAPI_BASE_URL = "http://model-api:8000"

st.set_page_config(layout="wide", page_title="Sistema de ML Distribuido con Ray")

st.title("📊 Sistema de Machine Learning Distribuido con Ray")
st.markdown("Una interfaz para gestionar, monitorizar y visualizar modelos de ML entrenados con Ray y servidos con FastAPI.")

# --- Función para llamar a la API ---
def call_api(endpoint, method="GET", data=None):
    url = f"{FASTAPI_BASE_URL}/{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión: No se pudo conectar con el servicio FastAPI en {FASTAPI_BASE_URL}. Asegúrate de que el contenedor 'model-api' esté corriendo.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error al llamar a la API en {endpoint}: {e}")
        if response and response.content:
            try:
                st.error(f"Detalles del error de la API: {response.json()}")
            except json.JSONDecodeError:
                st.error(f"Respuesta de error no JSON: {response.text}")
        return None

# --- Barra lateral para navegación ---
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a", ["Estado del Sistema", "Métricas de Entrenamiento", "Predicción de Modelos", "Monitoreo de Inferencia", "Visualización de Gráficas"])

# --- Sección: Estado del Sistema ---
if page == "Estado del Sistema":
    st.header("Estado General del Sistema")
    health_data = call_api("health")
    if health_data:
        st.success("Servicio FastAPI en línea y saludable.")
        st.json(health_data)

    st.subheader("Modelos Disponibles")
    models_data = call_api("models")
    if models_data and models_data.get("available_models"):
        st.write(f"Total de modelos cargados: {models_data['total_models']}")
        st.write("Lista de modelos:")
        for model_name in models_data['available_models']:
            st.markdown(f"- **{model_name}**")
    elif models_data:
        st.warning("No se encontraron modelos cargados. Asegúrate de que 'train.py' se haya ejecutado correctamente.")

# --- Sección: Métricas de Entrenamiento ---
elif page == "Métricas de Entrenamiento":
    st.header("Métricas de Entrenamiento de Modelos")
    all_metadata = call_api("all-models-metadata")
    if all_metadata:
        st.subheader("Resumen de Métricas por Modelo")
        metrics_df_data = []
        for model_name, metadata in all_metadata.items():
            if "error" in metadata:
                metrics_df_data.append({"Modelo": model_name, "Estado": "Error al cargar métricas", "Detalle": metadata["error"]})
                continue
            
            metrics = metadata.get("metrics", {})
            metrics_df_data.append({
                "Modelo": model_name,
                "Accuracy": f"{metrics.get('accuracy', 0):.4f}",
                "Precision": f"{metrics.get('precision', 0):.4f}",
                "Recall": f"{metrics.get('recall', 0):.4f}",
                "F1-Score": f"{metrics.get('f1_score', 0):.4f}",
                "ROC AUC": f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') is not None else "N/A",
                "Fecha Entrenamiento": metadata.get("training_date", "N/A").split('T')[0]
            })
        
        metrics_df = pd.DataFrame(metrics_df_data)
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Detalle de Métricas por Modelo")
        selected_model_metrics = st.selectbox("Selecciona un modelo para ver el detalle de sus métricas:", list(all_metadata.keys()))
        if selected_model_metrics:
            st.json(all_metadata[selected_model_metrics])

# --- Sección: Predicción de Modelos ---
elif page == "Predicción de Modelos":
    st.header("Realizar Predicciones")
    models_data = call_api("models")
    if models_data and models_data.get("available_models"):
        model_names = models_data['available_models']
        selected_model = st.selectbox("Selecciona un modelo para predecir:", model_names)

        if selected_model:
            st.subheader(f"Características para {selected_model}")
            # Aquí definimos las características esperadas para el dataset Titanic
            # Adapta esto si tu dataset es diferente
            features = {}
            st.write("Introduce las características del pasajero:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                features["Pclass"] = st.selectbox("Clase de Pasajero (Pclass):", [1, 2, 3], index=2)
                features["Sex"] = st.selectbox("Sexo:", ["male", "female"], index=0)
                features["Age"] = st.number_input("Edad:", min_value=0.0, max_value=100.0, value=25.0)
            with col2:
                features["SibSp"] = st.number_input("Número de hermanos/cónyuges a bordo (SibSp):", min_value=0, value=0)
                features["Parch"] = st.number_input("Número de padres/hijos a bordo (Parch):", min_value=0, value=0)
                features["Fare"] = st.number_input("Tarifa (Fare):", min_value=0.0, value=32.20)
            with col3:
                features["Embarked"] = st.selectbox("Puerto de Embarque (Embarked):", ["C", "Q", "S"], index=2)
            
            # Puedes añadir más características según tu dataset
            # Ejemplo: features["Cabin"] = st.text_input("Cabina (Cabin):", "N/A")
            # Ejemplo: features["Name"] = st.text_input("Nombre (Name):", "John Doe")

            if st.button("Obtener Predicción"):
                prediction_data = call_api(f"predict/{selected_model}", method="POST", data={"features": features})
                if prediction_data:
                    st.success("Predicción Exitosa!")
                    st.json(prediction_data)
                    st.info(f"Latencia de la predicción: {prediction_data.get('latency_ms', 0):.2f} ms")
    else:
        st.warning("No hay modelos disponibles para predicción. Entrena algunos modelos primero.")

# --- Sección: Monitoreo de Inferencia ---
elif page == "Monitoreo de Inferencia":
    st.header("Monitoreo de Inferencia en Producción")
    st.write("Estadísticas de uso de los modelos y latencia promedio.")

    inference_stats = call_api("inference-stats")
    if inference_stats:
        st.subheader("Resumen de Estadísticas de Inferencia")
        stats_df_data = []
        for model_name, stats in inference_stats.items():
            stats_df_data.append({
                "Modelo": model_name,
                "Total Solicitudes": stats.get('total_requests', 0),
                "Latencia Promedio (ms)": f"{stats.get('average_latency_ms', 0):.2f}"
            })
        stats_df = pd.DataFrame(stats_df_data)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Detalle de Latencias Recientes (Últimas 100)")
        selected_model_latency = st.selectbox("Selecciona un modelo para ver sus latencias recientes:", list(inference_stats.keys()))
        if selected_model_latency and inference_stats[selected_model_latency].get('last_100_latencies_ms'):
            latencies = inference_stats[selected_model_latency]['last_100_latencies_ms']
            st.line_chart(pd.DataFrame({"Latencia (ms)": latencies}))
        elif selected_model_latency:
            st.info("No hay datos de latencia recientes para este modelo aún.")
    else:
        st.info("No hay estadísticas de inferencia disponibles aún. Realiza algunas predicciones.")

# --- Sección: Visualización de Gráficas ---
elif page == "Visualización de Gráficas":
    st.header("Visualización de Gráficas de Rendimiento")
    models_data = call_api("models")
    if models_data and models_data.get("available_models"):
        model_names = models_data['available_models']
        selected_model_plot = st.selectbox("Selecciona un modelo para ver sus gráficas:", model_names)

        if selected_model_plot:
            st.subheader(f"Curva ROC para {selected_model_plot}")
            roc_png_data = call_api(f"model-roc-png/{selected_model_plot}")
            if roc_png_data and roc_png_data.get("image_base64"):
                st.image(base64.b64decode(roc_png_data["image_base64"]), caption=f"Curva ROC para {selected_model_plot}", use_column_width=True)
            else:
                st.warning(roc_png_data.get("error", "No se pudo cargar la Curva ROC. Asegúrate de que el modelo soporte predict_proba y que los datos de prueba estén disponibles."))

            st.subheader(f"Curva de Aprendizaje para {selected_model_plot}")
            learning_png_data = call_api(f"model-learning-curve-png/{selected_model_plot}")
            if learning_png_data and learning_png_data.get("image_base64"):
                st.image(base64.b64decode(learning_png_data["image_base64"]), caption=f"Curva de Aprendizaje para {selected_model_plot}", use_column_width=True)
            else:
                st.warning(learning_png_data.get("error", "No se pudo cargar la Curva de Aprendizaje. Asegúrate de que los datos de entrenamiento estén disponibles."))
    else:
        st.warning("No hay modelos disponibles para generar gráficas. Entrena algunos modelos primero.")

st.sidebar.markdown("---")
st.sidebar.info("Desarrollado con Ray, FastAPI y Streamlit.")

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import os 
import json

FASTAPI_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
st.sidebar.info(f"Conectando a FastAPI en: {FASTAPI_BASE_URL}")

def get_available_models():
    """Obtiene la lista de modelos disponibles desde la API de FastAPI."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/models")
        response.raise_for_status()  
        return response.json().get("available_models", [])
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión: Asegúrate de que la API de FastAPI esté corriendo en {FASTAPI_BASE_URL}")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener modelos: {e}")
        return []

def get_all_models_metadata():
    """Obtiene los metadatos de todos los modelos desde la API de FastAPI."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/all-models-metadata")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión: Asegúrate de que la API de FastAPI esté corriendo en {FASTAPI_BASE_URL}")
        return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener metadatos: {e}")
        return {}

def make_prediction(model_name: str, features: dict):
    """Realiza una predicción para un modelo dado con las características proporcionadas."""
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/predict/{model_name}", json={"features": features})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión: Asegúrate de que la API de FastAPI esté corriendo en {FASTAPI_BASE_URL}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Error de la API al predecir: {e.response.json().get('detail', str(e))}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error inesperado al predecir: {e}")
        return None

st.set_page_config(layout="wide", page_title="Métricas de Rendimiento de Modelos ML")

st.title("Métricas de Rendimiento de Modelos de ML entrenados con Ray")
st.markdown("""
Esta aplicación de Streamlit se conecta a la API de FastAPI para obtener y visualizar métricas de rendimiento y realizar predicciones con los modelos de Machine Learning desplegados en Ray.
""")

st.header("✨ Modelos Disponibles")
available_models = get_available_models()
if available_models:
    st.success(f"Modelos conectados y disponibles: {', '.join(available_models)}")
else:
    st.warning("No se encontraron modelos disponibles. Asegúrate de que la API de FastAPI esté funcionando y los actores de Ray estén registrados.")

st.header("📈 Metadatos de Entrenamiento y Rendimiento")
all_metadata = get_all_models_metadata()

if all_metadata:
    metadata_rows = []

    chart_metrics_data = {} 

    for model_name, data in all_metadata.items():
        if "error" in data:
            metadata_rows.append({
                "Modelo": model_name, 
                "Estado": data["error"], 
                "Precisión (Accuracy)": "N/A", 
                "Recall": "N/A",
                "F1 Score": "N/A",
            })
            chart_metrics_data[model_name] = {
                "Precisión (Accuracy)": None,
                "Recall": None,
                "F1 Score": None
            }
        else:
            metrics = data.get("metrics", {}) 
            
            accuracy = metrics.get("accuracy", "N/A")
            recall = metrics.get("recall", "N/A")
            f1_score = metrics.get("f1_score", "N/A")
            
            metadata_rows.append({
                "Modelo": model_name,
                "Estado": "Cargado",
                "Precisión (Accuracy)": f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else accuracy,
                "Recall": f"{recall:.4f}" if isinstance(recall, (int, float)) else recall,
                "F1 Score": f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else f1_score,
            })
            
            chart_metrics_data[model_name] = {
                "Precisión (Accuracy)": accuracy if isinstance(accuracy, (int, float)) else None,
                "Recall": recall if isinstance(recall, (int, float)) else None,
                "F1 Score": f1_score if isinstance(f1_score, (int, float)) else None
            }

    df_metadata = pd.DataFrame(metadata_rows)
    st.dataframe(df_metadata, use_container_width=True)

    st.subheader("Comparación de Métricas de Rendimiento")

    metric_options = ["Precisión (Accuracy)", "Recall", "F1 Score"]
    selected_metric = st.selectbox("Selecciona la métrica para comparar:", metric_options)

    chart_data_rows = []
    has_chart_data = False
    for model_name, metrics_values in chart_metrics_data.items():
        metric_value = metrics_values.get(selected_metric)
        if metric_value is not None: 
            chart_data_rows.append({"Modelo": model_name, "Valor de la Métrica": metric_value})
            has_chart_data = True

    if has_chart_data:
        df_chart = pd.DataFrame(chart_data_rows)
        
        fig = px.bar(df_chart, x='Modelo', y='Valor de la Métrica',
                        title=f'{selected_metric} de los Modelos Entrenados',
                        labels={'Valor de la Métrica': selected_metric},
                        color='Modelo',
                        template='plotly_white')
        fig.update_layout(xaxis_title="Modelo", yaxis_title=selected_metric)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No hay datos numéricos de '{selected_metric}' disponibles para graficar.")
else:
    st.info("No se pudieron cargar los metadatos de los modelos.")

st.header("🔮 Predicción")

if available_models:
    selected_model = st.selectbox("Selecciona un modelo para predecir:", available_models)

    st.markdown("""
    Introduce las características de entrada para la predicción.
    """)

    st.subheader("Input (JSON)")
    default_features_example = {
        "Pclass": 3,
        "Sex": "female",
        "Age": 20,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.50,
        "Embarked": "S"
    }
    
    features_input = st.text_area(
        "Introduce las características:",
        value=json.dumps(default_features_example, indent=2),
        height=200
    )

    predict_button = st.button("Realizar Predicción")

    if predict_button:
        try:
            features = json.loads(features_input)
            if not isinstance(features, dict):
                st.error("El input JSON debe ser un objeto (diccionario) de características.")
            else:
                with st.spinner(f"Realizando predicción con {selected_model}..."):
                    prediction_response = make_prediction(selected_model, features)
                    if prediction_response:
                        st.subheader("Resultados de la Predicción:")
                        st.json(prediction_response)
                        st.info(f"Latencia de la predicción: {prediction_response.get('latency_ms', 'N/A'):.2f} ms")
        except json.JSONDecodeError:
            st.error("Error: El formato JSON de las características es inválido. Por favor, revisa la sintaxis.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar la predicción: {e}")
else:
    st.info("No hay modelos disponibles para realizar predicciones.")

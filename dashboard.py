import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import os
import json
from io import StringIO

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

def start_training_api_call(data_json_string: str, target_column: str, models_to_train: list[str]):
    """Llama al endpoint de FastAPI para iniciar el entrenamiento, pasando datos como JSON."""
    payload = {
        "data": data_json_string,
        "target_column": target_column,
        "models_to_train": models_to_train
    }
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/train-models", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión al iniciar entrenamiento: Asegúrate de que la API de FastAPI esté corriendo en {FASTAPI_BASE_URL}")
        return {"status": "error", "message": "Error de conexión."}
    except requests.exceptions.HTTPError as e:
        st.error(f"Error de la API al iniciar entrenamiento: {e.response.json().get('detail', str(e))}")
        return {"status": "error", "message": e.response.json().get('detail', str(e))}
    except requests.exceptions.RequestException as e:
        st.error(f"Error inesperado al iniciar entrenamiento: {e}")
        return {"status": "error", "message": str(e)}

def get_training_status_api_call(task_id: str):
    """Consulta el estado de una tarea de entrenamiento específica."""
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/training-status/{task_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Error de conexión al consultar estado: Asegúrate de que la API de FastAPI esté corriendo en {FASTAPI_BASE_URL}")
        return {"status": "error", "message": "Error de conexión."}
    except requests.exceptions.HTTPError as e:
        st.error(f"Error de la API al consultar estado: {e.response.json().get('detail', str(e))}")
        return {"status": "error", "message": e.response.json().get('detail', str(e))}
    except requests.exceptions.RequestException as e:
        st.error(f"Error inesperado al consultar estado: {e}")
        return {"status": "error", "message": str(e)}

st.set_page_config(layout="wide", page_title="Plataforma de ML con Ray y FastAPI")

st.title("🚀 Plataforma de Entrenamiento y Predicción de Modelos ML")
st.markdown("""
Esta aplicación de Streamlit te permite subir datasets, **desencadenar el entrenamiento** de modelos con Ray a través de una API de FastAPI, y luego **visualizar el rendimiento** y **realizar predicciones** con los modelos entrenados.
""")

st.sidebar.header("Estado del Sistema")
health_check_response = requests.get(f"{FASTAPI_BASE_URL}/health")
if health_check_response.status_code == 200:
    health_data = health_check_response.json()
    st.sidebar.success(f"FastAPI API: {health_data['status']}")
    st.sidebar.info(f"Modelos cargados: {health_data['models_loaded']}")
    st.sidebar.info(f"Orquestador de entrenamiento: {health_data['training_orchestrator_status']}")
else:
    st.sidebar.error(f"FastAPI API: No disponible ({health_check_response.status_code})")

st.header("✨ Modelos Disponibles y Sus Métricas")
available_models_names = get_available_models()
if available_models_names:
    st.success(f"Modelos conectados y disponibles para predicción: {', '.join(available_models_names)}")
else:
    st.warning("No se encontraron modelos disponibles para predicción. Entrena algunos primero.")

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
                "Fecha Entrenamiento": "N/A",
                "Parámetros": "N/A"
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
            training_date = data.get("training_date", "N/A")
            params = data.get("params", {})

            metadata_rows.append({
                "Modelo": model_name,
                "Estado": "Cargado",
                "Precisión (Accuracy)": f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else accuracy,
                "Recall": f"{recall:.4f}" if isinstance(recall, (int, float)) else recall,
                "F1 Score": f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else f1_score,
                "Fecha Entrenamiento": training_date.split("T")[0] if isinstance(training_date, str) else training_date,
                "Parámetros": json.dumps(params)
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
    selected_metric = st.selectbox("Selecciona la métrica para comparar:", metric_options, key="metric_compare")

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

st.header("⚙️ Entrenamiento de Modelos")
st.markdown("""
Sube tu dataset, selecciona la columna objetivo y elige qué modelos deseas entrenar.
""")

uploaded_file = st.file_uploader("Sube un archivo CSV o JSON para entrenar", type=["csv", "json"])

if uploaded_file is not None:
    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    st.write(file_details)

    df_preview = None
    try:
        if uploaded_file.type == "text/csv":
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0) # Resetear la posición del stream después de leer con pandas
        elif uploaded_file.type == "application/json":
            df_preview = pd.read_json(uploaded_file)
            uploaded_file.seek(0) # Resetear la posición del stream después de leer con pandas

        st.subheader("Vista Previa del Dataset:")
        st.dataframe(df_preview.head())

    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        df_preview = None

    if df_preview is not None:
        all_columns = df_preview.columns.tolist()

        st.subheader("Configuración del Entrenamiento")
        target_column = st.selectbox("Selecciona la columna objetivo para el entrenamiento:", all_columns, key="target_col_select")

        configured_models_for_selection = ["LogisticRegression", "RandomForestClassifier", "SVC"]

        models_to_train_selected = st.multiselect(
            "Selecciona los modelos que deseas entrenar:",
            options=configured_models_for_selection,
            default=configured_models_for_selection,
            key="models_to_train_select"
        )

        training_status_placeholder = st.empty()

        if st.button("Iniciar Entrenamiento", key="start_training_button"):
            if not models_to_train_selected:
                st.warning("Por favor, selecciona al menos un modelo para entrenar.")
            elif not target_column:
                st.warning("Por favor, selecciona la columna objetivo.")
            else:
                # Convertir el DataFrame a una cadena JSON para enviar a través de HTTP
                json_data_string = df_preview.to_json(orient='records')

                training_response = start_training_api_call(
                    json_data_string,
                    target_column,
                    models_to_train_selected
                )

                if training_response and (training_response.get("status") == "initiated" or training_response.get("status") == "success"):
                    task_id = training_response.get("task_id")
                    if task_id:
                        training_status_placeholder.info(f"Entrenamiento iniciado (ID: {task_id}). Por favor, espera mientras se procesa...")

                        progress_bar = st.progress(0)
                        status_message = training_status_placeholder.empty()

                        for i in range(1, 11): # Bucle para verificar el estado
                            time.sleep(2)
                            status_data = get_training_status_api_call(task_id)
                            progress_bar.progress(i * 10)
                            status_message.info(f"Estado de la tarea {task_id}: {status_data.get('message', 'Consultando...')}")

                            if status_data.get("status") in ["completed", "failed", "error"]:
                                break

                        if status_data.get("status") == "completed":
                            st.success(f"¡Entrenamiento completado! Los modelos han sido actualizados. Puedes recargar la página para ver las nuevas métricas.")
                            st.balloons()
                        elif status_data.get("status") == "failed" or status_data.get("status") == "error":
                            st.error(f"El entrenamiento falló (ID: {task_id}): {status_data.get('message', 'Error desconocido')}")
                        else:
                            st.warning(f"El entrenamiento (ID: {task_id}) sigue en curso o no se pudo determinar el estado final. Recarga la página más tarde.")

                    else:
                        st.error("El entrenamiento se inició, pero no se recibió un ID de tarea.")
                else:
                    st.error(f"Fallo al iniciar el entrenamiento: {training_response.get('message', 'Error desconocido')}")


st.header("🔮 Predicción")

if available_models_names:
    selected_model_for_prediction = st.selectbox("Selecciona un modelo para predecir:", available_models_names, key="predict_model_select")

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
        height=200,
        key="features_input_text"
    )

    predict_button = st.button("Realizar Predicción", key="predict_button")

    if predict_button:
        try:
            features = json.loads(features_input)
            if not isinstance(features, dict):
                st.error("El input JSON debe ser un objeto (diccionario) de características.")
            else:
                with st.spinner(f"Realizando predicción con {selected_model_for_prediction}..."):
                    prediction_response = make_prediction(selected_model_for_prediction, features)
                    if prediction_response:
                        st.subheader("Resultados de la Predicción:")
                        st.json(prediction_response)
        except json.JSONDecodeError:
            st.error("Error al parsear el JSON de entrada. Asegúrate de que el formato sea válido.")
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
else:
    st.info("No hay modelos disponibles para realizar predicciones. Por favor, entrena algunos modelos primero.")
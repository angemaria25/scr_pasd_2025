import streamlit as st
import requests
import json
import os
import pandas as pd
import base64
import time

st.set_page_config(page_title="Plataforma distribuida de entrenamiento supervisado", layout="wide")

# --- Robust session state initialization ---
def ensure_session_state():
    """Initialize all session state variables to prevent KeyError crashes"""
    defaults = {
        'cluster': {'head': {'cpu': 2, 'ram': 4, 'running': False}},
        'uploaded_files': {},
        'file_configs': {},
        'last_training_results': None,
        'last_worker_count': None,
        'sidebar_section': 'Cluster'  # Default section
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state to prevent crashes
ensure_session_state()

# --- CLUSTER MANAGEMENT ---
st.sidebar.title("Menú principal")

# Ultra-safe radio widget to prevent crashes during scaling
section = 'Cluster'  # Default fallback
try:
    # Try to get current section safely
    current_index = 0
    if 'sidebar_section' in st.session_state:
        try:
            current_index = ["Cluster", "Training", "Predicción"].index(st.session_state['sidebar_section'])
        except (ValueError, KeyError):
            current_index = 0
    
    # Create radio widget with stable key
    section = st.sidebar.radio(
        "Selecciona una sección", 
        ["Cluster", "Training", "Predicción"],
        index=current_index,
        key="main_section_radio"  # Stable key for consistent behavior
    )
    st.session_state['sidebar_section'] = section
except Exception as e:
    st.sidebar.warning("🔄 Menú en modo recuperación")
    # Use selectbox as fallback
    try:
        section = st.sidebar.selectbox(
            "Selecciona una sección", 
            ["Cluster", "Training", "Predicción"],
            index=0,
            key="fallback_section_selector"
        )
    except:
        section = 'Cluster'  # Ultimate fallback

def check_backend_connectivity():
    """Check if backend is accessible and return status info"""
    try:
        response = requests.get('http://localhost:8000/health', timeout=5)
        if response.status_code == 200:
            return {"status": "connected", "message": "Backend is accessible"}
        else:
            return {"status": "error", "message": f"Backend returned status {response.status_code}"}
    except requests.exceptions.ConnectRefused:
        return {"status": "disconnected", "message": "Backend server is not running. Please start the backend with: docker-compose up"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "message": "Backend is running but not responding. Check if it's still starting up."}
    except Exception as e:
        return {"status": "error", "message": f"Connection error: {str(e)}"}

def get_workers_from_api():
    """Get workers information directly from API"""
    try:
        response = requests.get('http://localhost:8000/cluster/workers', timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": "API not available", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}

def get_cluster_status():
    """Get cluster status from backend API"""
    try:
        response = requests.get('http://localhost:8000/cluster/status', timeout=15)
        if response.status_code == 200:
            return response.json()
        return {"error": "Backend unavailable"}
    except Exception as e:
        return {"error": str(e)}



if section == "Cluster":
    st.header("Gestión del Clúster Ray Distribuido")
    
    # Add refresh button for cluster state
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("")  # Empty space
    with col2:
        if st.button("🔄 Refrescar workers", help="Consulta la API para obtener el estado actual de todos los workers"):
            # Force refresh by clearing any cached data and re-querying API
            st.cache_data.clear()
            workers_info = get_workers_from_api()
            if workers_info.get("success"):
                st.success(f"✅ Detectados {workers_info.get('total_workers', 0)} workers activos")
            else:
                st.error(f"❌ Error consultando workers: {workers_info.get('error', 'Unknown error')}")
            st.rerun()
    
    # Get real cluster status from backend
    cluster_status = get_cluster_status()
    
    # Detect changes in worker count
    if "error" not in cluster_status:
        nodes = cluster_status.get("node_details", [])
        current_worker_count = max(0, len(nodes) - 1) if nodes else 0
        
        # Removed info/warning messages for worker count changes
        
        st.session_state['last_worker_count'] = current_worker_count
    
    if "error" not in cluster_status:
        st.subheader("Estado Actual del Clúster")
        
        # Get worker details from backend API
        workers_api_response = get_workers_from_api()
        worker_details = []
        if workers_api_response.get("success"):
            worker_details = workers_api_response.get("workers", [])
        
        # Also try the original endpoint as fallback
        if not worker_details:
            try:
                workers_response = requests.get('http://localhost:8000/cluster/workers', timeout=15)
                if workers_response.status_code == 200:
                    worker_data = workers_response.json()
                    if worker_data.get('success'):
                        worker_details = worker_data.get('workers', [])
            except Exception as e:
                st.warning(f"Could not fetch worker details (timeout or error): {e}")
        
        # Create comprehensive cluster table
        st.markdown("### 📋 Nodos del Clúster")
        
        # Prepare table data
        table_data = []
        
        # Add head node
        nodes = cluster_status.get("node_details", [])
        head_node = None
        if nodes:
            head_node = nodes[0]  # First node is typically the head
        
        # Use realistic CPU values instead of Ray's over-reported values
        # Ray often reports virtual/logical cores, we'll cap at realistic values
        head_cpu_raw = head_node.get("Resources", {}).get("CPU", 2.0) if head_node else 2.0
        head_cpu = min(head_cpu_raw, 8)  # Cap at 8 cores for more realistic display
        head_memory = head_node.get("Resources", {}).get("memory", 4e9) / 1e9 if head_node else 4.0
        head_status = "🟢 Activo" if head_node and head_node.get("Alive") else "🔴 Inactivo"
        
        table_data.append({
            "Nodo": "🎯 Head Node (ray-head)",
            "CPU": f"{head_cpu}",
            "RAM (GB)": f"{head_memory:.1f}",
            "Estado": head_status,
            "Tipo": "Coordinador Principal"
        })
        
        # Add worker nodes - use same logic as metrics (all Ray nodes except head)
        worker_nodes = nodes[1:] if len(nodes) > 1 else []  # All nodes except head
        
        # Build worker table using API data
        if worker_details:
            for worker in worker_details:
                # Get CPU and memory from worker resources
                worker_cpu = 4  # Default
                worker_memory = 2.0  # Default
                
                if 'resources' in worker:
                    worker_cpu = min(worker['resources'].get('CPU', 4), 4)
                    worker_memory = worker['resources'].get('memory', 2e9) / 1e9
                
                status_icon = "🟢"
                status_text = "Activo"
                
                table_data.append({
                    "Nodo": f"⚙️ Worker {worker['number']} ({worker['name']})",
                    "CPU": f"{worker_cpu}",
                    "RAM (GB)": f"{worker_memory:.1f}",
                    "Estado": f"{status_icon} {status_text}",
                    "Tipo": "Nodo de Procesamiento"
                })
        else:
            # Fallback: use Ray cluster info if API data not available
            for i, worker_node in enumerate(worker_nodes):
                if worker_node.get("Alive"):
                    worker_cpu_raw = worker_node.get("Resources", {}).get("CPU", 2.0)
                    worker_memory = worker_node.get("Resources", {}).get("memory", 2e9) / 1e9
                    worker_cpu = min(worker_cpu_raw, 4)
                    
                    table_data.append({
                        "Nodo": f"⚙️ Worker {i+1} (ray-worker-{i+1})",
                        "CPU": f"{worker_cpu}",
                        "RAM (GB)": f"{worker_memory:.1f}",
                        "Estado": "🟢 Activo",
                        "Tipo": "Nodo de Procesamiento"
                    })
        
        # Display table with current worker count info
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No se pudo obtener información de los nodos del clúster")
        
        # Detailed cluster information
        with st.expander("🔍 Información Detallada del Clúster"):
            st.json(cluster_status)
    
    else:
        st.warning(f"⚠️ Estado del clúster no disponible: {cluster_status['error']}")
        st.info("💡 Esto puede ocurrir si Ray no está completamente inicializado. Verifica los logs del contenedor.")
    
    st.stop()

# --- API STATUS ---
st.sidebar.markdown("---")
st.sidebar.subheader("Estado del Backend API")
try:
    response = requests.get('http://localhost:8000/health', timeout=2)
    if response.status_code == 200:
        st.sidebar.success("🟢 API disponible")
    else:
        st.sidebar.warning("⚠️ API con problemas")
except Exception:
    st.sidebar.error("🔴 API no disponible")


st.title("Distributed ML Platform - Visual Interface")

# --- SECTION: TRAINING ---
if section == "Training":
    st.header("🚀 Entrenamiento Distribuido de Modelos ML")
    st.markdown("Suba archivos CSV/JSON para procesamiento y entrenamiento distribuido en el clúster Ray")
    
    # Check for existing trained models (no longer shown in UI, but still fetched for prediction section)
    try:
        models_response = requests.get('http://localhost:8000/models', timeout=5)
        if models_response.status_code == 200:
            existing_models = models_response.json()
            # No UI display here; models are used in prediction section only
    except Exception:
        pass  # If check fails, continue normally
    
    # Show recent training results if available
    if st.session_state.get('last_training_results'):
        last_results = st.session_state['last_training_results']
        time_ago = int(time.time() - last_results['timestamp'])
        
        if time_ago < 3600:  # Show if less than 1 hour ago
            minutes_ago = time_ago // 60
            with st.expander(f"📈 Recent Training Results ({minutes_ago} minutes ago)"):
                result = last_results['results']
                
                if 'results' in result:
                    for dataset_name, dataset_result in result['results'].items():
                        if dataset_result.get('status') == 'success' and 'results' in dataset_result:
                            st.write(f"**{dataset_name}:**")
                            for model_name, model_result in dataset_result['results'].items():
                                accuracy = model_result.get('accuracy')
                                if accuracy is None:
                                    accuracy = model_result.get('metrics', {}).get('accuracy')
                                if accuracy is None:
                                    accuracy = model_result.get('test_score')
                                
                                if accuracy is not None:
                                    try:
                                        # Ensure accuracy is a number before formatting
                                        accuracy_float = float(accuracy)
                                        st.write(f"  - {model_name}: {accuracy_float:.4f}")
                                    except (ValueError, TypeError):
                                        # If conversion fails, display as-is
                                        st.write(f"  - {model_name}: {accuracy}")
                                else:
                                    st.write(f"  - {model_name}: Training completed")
    
    # Session state is already initialized at the top of the file
    
    # Step 1: File Upload
    st.subheader("1. 📁 Subir Archivos CSV/JSON")

    # Add distributed memory clear button
    if st.button("🧹 Limpiar Memoria Distribuida", help="Limpia toda la memoria distribuida y los modelos entrenados para evitar desbordamientos y empezar de cero."):
        with st.spinner("Limpiando memoria distribuida y recursos del clúster..."):
            try:
                response = requests.post("http://localhost:8000/clear_memory", timeout=60)
                if response.status_code == 200:
                    st.session_state['uploaded_files'] = {}
                    st.session_state['file_configs'] = {}
                    st.session_state['last_training_results'] = None
                    st.success("✅ Memoria distribuida limpiada correctamente. Puedes subir nuevos archivos y entrenar modelos desde cero.")
                    st.rerun()
                else:
                    try:
                        error_msg = response.json().get('detail', response.text)
                    except Exception:
                        error_msg = response.text
                    st.error(f"❌ Error al limpiar memoria distribuida: {error_msg}")
            except Exception as e:
                st.error(f"❌ Error de conexión al limpiar memoria distribuida: {e}")

    # File uploader for multiple files
    uploaded_files = st.file_uploader(
        "Seleccione archivos CSV o JSON para procesamiento distribuido:",
        type=['csv', 'json'],
        accept_multiple_files=True,
        help="Seleccione uno o más archivos CSV/JSON para entrenar modelos de ML distribuido"
    )
    
    if uploaded_files:
        st.success(f"📊 {len(uploaded_files)} archivo(s) seleccionado(s)")
        
        # Display uploaded files info
        st.write("**Archivos seleccionados:**")
        for uploaded_file in uploaded_files:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            st.write(f"- {uploaded_file.name} ({file_size:.2f} MB)")
        
        # Auto-process files if not already processed
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state['uploaded_files']]
        
        # Debug info
        if not files_to_process and uploaded_files:
            st.warning("⚠️ Los archivos ya están en la sesión pero pueden no haberse procesado correctamente.")
            if st.button("🔄 Forzar reprocesamiento de todos los archivos"):
                # Clear session state for these files
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state['uploaded_files']:
                        del st.session_state['uploaded_files'][uploaded_file.name]
                st.rerun()
        
        if files_to_process:
            # Process and upload files to backend automatically
            st.info(f"🔄 {len(files_to_process)} archivo(s) nuevo(s) detectado(s). Procesando automáticamente...")
            uploaded_count = 0
            
            for i, uploaded_file in enumerate(files_to_process):
                filename = uploaded_file.name
                
                # Add delay between requests to prevent API overload
                if i > 0:
                    time.sleep(0.5)
                
                try:
                    # Read file content
                    file_content = uploaded_file.getvalue()
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    
                    # Upload to backend
                    upload_request = {
                        "filename": filename,
                        "content": encoded_content
                    }
                    
                    with st.spinner(f"Procesando {filename}..."):
                        response = requests.post(
                            "http://localhost:8000/upload",
                            json=upload_request,
                            timeout=60
                        )
                    
                    if response.status_code == 200:
                        upload_result = response.json()
                        st.session_state['uploaded_files'][filename] = upload_result
                        uploaded_count += 1
                        st.success(f"✅ {filename} procesado y distribuido ({upload_result.get('rows', 'N/A')} filas)")
                    else:
                        # Get detailed error information
                        try:
                            error_details = response.json()
                            error_msg = error_details.get('detail', response.text)
                        except:
                            error_msg = response.text
                        
                        st.error(f"❌ Error procesando {filename} (Código {response.status_code})")
                        
                        # Show specific error details
                        if response.status_code == 400:
                            st.error(f"🔍 Detalle del error: {error_msg}")
                            st.info("💡 Posibles causas: archivo corrupto, formato incorrecto, o problema con la codificación")
                        elif response.status_code == 403:
                            st.error(f"🔍 Error de permisos (403): {error_msg}")
                            st.info("💡 Posibles causas: problema de autenticación del backend, permisos de Ray, o configuración CORS")
                            st.warning("🔧 Solución: Reinicie el backend con `docker-compose restart ray-head`")
                        elif response.status_code == 500:
                            st.error(f"🔍 Error interno del servidor: {error_msg}")
                            st.info("💡 Posible causa: problema en el procesamiento del backend")
                        else:
                            st.error(f"🔍 Error: {error_msg}")
                        
                        # Store failed upload info to prevent showing configuration
                        st.session_state['uploaded_files'][filename] = {"error": error_msg}
                        
                except Exception as e:
                    st.error(f"❌ Error de conexión procesando {filename}: {e}")
                    st.info("💡 Verifique que el backend esté ejecutándose correctamente")
                    # Store failed upload info to prevent showing configuration
                    st.session_state['uploaded_files'][filename] = {"error": str(e)}
            
            if uploaded_count > 0:
                st.info(f"📤 {uploaded_count} archivo(s) distribuido(s) en el clúster Ray")
                st.success("🔄 Página actualizada. Los archivos están listos para configuración.")
            elif uploaded_count == 0 and files_to_process:
                st.error("❌ Ningún archivo pudo ser procesado. Verifique los errores arriba.")
                
                # Add diagnostic information
                with st.expander("🔧 Información de diagnóstico"):
                    st.write("**Estado del backend:**")
                    backend_status = check_backend_connectivity()
                    if backend_status["status"] == "connected":
                        st.success("✅ Backend accesible")
                    else:
                        st.error(f"❌ Problema con backend: {backend_status['message']}")
                    
                    st.write("**Archivos que fallaron:**")
                    for file in files_to_process:
                        file_size = len(file.getvalue()) / (1024 * 1024)
                        st.write(f"- {file.name}: {file_size:.2f} MB")
                        
                    st.write("**Posibles soluciones:**")
                    st.write("1. **Para errores 403**: Reinicie el backend: `docker-compose restart ray-head`")
                    st.write("2. **Para errores 400**: Verifique que los archivos CSV tengan formato correcto")
                    st.write("3. Asegúrese que los archivos no estén corruptos")
                    st.write("4. Verifique que el backend esté funcionando: `docker-compose logs ray-head`")
                    st.write("5. Si persiste, reconstruya: `docker-compose up --build -d`")
        else:
            # All files already processed
            st.success("✅ Todos los archivos ya han sido procesados y están listos para configuración")
            if st.button("🔄 Actualizar vista", help="Refresca la interfaz para mostrar la sección de configuración"):
                st.rerun()
    else:
        st.info("💡 Seleccione archivos CSV o JSON para comenzar el procesamiento distribuido")
    
    # Step 2: File Configuration and Training
    # Only show configuration if files are successfully uploaded and have valid data
    successfully_uploaded_files = {k: v for k, v in st.session_state['uploaded_files'].items() 
                                  if v and 'rows' in v and 'columns' in v}
    
    # Verify that uploaded files are still accessible in the backend
    if successfully_uploaded_files:
        try:
            response = requests.get('http://localhost:8000/uploaded_files', timeout=5)
            if response.status_code == 200:
                backend_response = response.json()
                backend_files = backend_response.get('files', [])
                
                # Only check for missing files if the backend returned a valid response
                if isinstance(backend_files, list):
                    missing_files = [f for f in successfully_uploaded_files.keys() if f not in backend_files]
                    if missing_files and len(backend_files) == 0:
                        # Only warn if ALL files are missing (indicating cluster restart)
                        st.warning(f"⚠️ Detectado que el clúster Ray se reinició. Los archivos necesitan ser vueltos a subir.")
                        # Clear missing files from session state
                        for missing_file in missing_files:
                            if missing_file in st.session_state['uploaded_files']:
                                del st.session_state['uploaded_files'][missing_file]
                            if missing_file in st.session_state['file_configs']:
                                del st.session_state['file_configs'][missing_file]
                        # Update the list of successfully uploaded files
                        successfully_uploaded_files = {k: v for k, v in st.session_state['uploaded_files'].items() 
                                                      if v and 'rows' in v and 'columns' in v}
                        if not successfully_uploaded_files:
                            st.info("💡 Por favor, vuelve a subir los archivos.")
                    elif missing_files:
                        # Some files missing but not all - show info message
                        st.info(f"ℹ️ Algunos archivos ({missing_files}) no están disponibles en el backend. Esto puede ocurrir si se han agregado/eliminado workers o si el clúster se ha reconfigurado. Si persisten los problemas, vuelve a subir los archivos.")
        except Exception:
            pass  # If backend check fails, continue with cached session state
    
    if successfully_uploaded_files:
        st.subheader("2. ⚙️ Configure Training Parameters")
        for filename, file_info in successfully_uploaded_files.items():
            st.markdown(f"#### 📄 Configure {filename}")
            st.caption(f"{file_info['rows']} rows, {len(file_info['columns'])} columns")

            # Show file preview
            if file_info.get('preview'):
                with st.expander("👁️ View Data Preview"):
                    preview_df = pd.DataFrame(file_info['preview'])
                    st.dataframe(preview_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Smart task type detection based on filename
                default_task_type = "classification"
                if "housing" in filename.lower() or "price" in filename.lower() or "regression" in filename.lower():
                    default_task_type = "regression"
                elif "classification" in filename.lower() or "cancer" in filename.lower():
                    default_task_type = "classification"
                # Task type selection with smart default
                default_index = 0 if default_task_type == "classification" else 1
                task_type = st.selectbox(
                    "Task Type",
                    ["classification", "regression"],
                    index=default_index,
                    key=f"task_{filename}",
                    help=f"Recommended: {default_task_type} (based on filename analysis)"
                )

            with col2:
                # Smart target column detection
                default_target = "target"
                if "target" in file_info['columns']:
                    default_target = "target"
                elif "price" in [col.lower() for col in file_info['columns']]:
                    default_target = next(col for col in file_info['columns'] if col.lower() == "price")
                elif "value" in [col.lower() for col in file_info['columns']]:
                    default_target = next(col for col in file_info['columns'] if col.lower() == "value")
                elif any("y" == col.lower() for col in file_info['columns']):
                    default_target = next(col for col in file_info['columns'] if col.lower() == "y")
                # Target column selection with smart default
                try:
                    default_index = file_info['columns'].index(default_target)
                except ValueError:
                    default_index = 0
                target_column = st.selectbox(
                    "Target Column",
                    file_info['columns'],
                    index=default_index,
                    key=f"target_{filename}",
                    help=f"The column to predict. Usually named 'target', 'price', 'value', or 'y'"
                )

            # Algorithm selection - Multiple selection
            if task_type == "classification":
                algorithms = ["Decision Tree", "Gradient Boosting", "SVM", "Logistic Regression", "K-Nearest Neighbors"]
            else:
                algorithms = ["Decision Tree Regressor", "Gradient Boosting Regressor", "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net"]
            selected_algorithms = st.multiselect(
                "Select Models to Train (you can select multiple)",
                algorithms,
                default=[algorithms[0]],  # Default to first algorithm
                key=f"algos_{filename}",
                help="You can select multiple models to train and compare their performance"
            )

            # Advanced parameters section (not nested in expander)
            st.markdown("**⚙️ Advanced Parameters:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, key=f"test_size_{filename}")
            with col2:
                random_state = st.number_input("Random State", 1, 1000, 42, key=f"random_state_{filename}")

            # Store configuration (cross_val_folds removed)
            st.session_state['file_configs'][filename] = {
                'task_type': task_type,
                'target_column': target_column,
                'algorithms': selected_algorithms,
                'test_size': test_size,
                'random_state': random_state
            }
            # Show current configuration status
            if selected_algorithms:
                st.success(f"✅ {len(selected_algorithms)} model(s) configured for {filename}")
            else:
                st.warning("⚠️ Please select at least one algorithm")
            st.markdown("---")  # Separator between datasets
        
        # Add single "Train All" button at the end
        st.subheader("3. 🚀 Train All Models")
        
        # Count total models across all datasets (only successfully uploaded ones)
        total_models = 0
        valid_configs = 0
        
        for filename, config in st.session_state['file_configs'].items():
            if filename in successfully_uploaded_files and config.get('algorithms'):
                total_models += len(config['algorithms'])
                valid_configs += 1
        
        if total_models > 0:
            st.info(f"📊 Ready to train {total_models} model(s) across {valid_configs} dataset(s)")
            
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                with st.spinner(f"Training {total_models} model(s) across {valid_configs} dataset(s)..."):
                    try:
                        # Create algorithm mapping function
                        def convert_algorithm_name(algo_name, task_type):
                            """Convert display name to API name"""
                            mapping = {
                                # Classification algorithms
                                "Decision Tree": "decision_tree",
                                "Gradient Boosting": "gradient_boosting", 
                                "SVM": "svm",
                                "Logistic Regression": "logistic_regression",
                                "K-Nearest Neighbors": "k_nearest_neighbors",
                                # Regression algorithms
                                "Decision Tree Regressor": "decision_tree_regressor",
                                "Gradient Boosting Regressor": "gradient_boosting_regressor",
                                "Linear Regression": "linear_regression",
                                "Ridge Regression": "ridge_regression",
                                "Lasso Regression": "lasso_regression",
                                "Elastic Net": "elastic_net"
                            }
                            return mapping.get(algo_name, algo_name.lower().replace(" ", "_"))
                        
                        # Prepare batch training request (only for successfully uploaded files)
                        datasets_config = {}
                        
                        for filename, config in st.session_state['file_configs'].items():
                            if filename in successfully_uploaded_files and config.get('algorithms'):
                                # Convert algorithm names to API format
                                api_algorithms = [
                                    convert_algorithm_name(algo, config['task_type']) 
                                    for algo in config['algorithms']
                                ]
                                
                                datasets_config[filename] = {
                                    "task_type": config['task_type'],
                                    "target_column": config['target_column'],
                                    "algorithms": api_algorithms,
                                    "test_size": config['test_size'],
                                    "random_state": config['random_state']
                                }
                        
                        # Send batch training request
                        response = requests.post(
                            "http://localhost:8000/train_all_datasets",
                            json={"datasets": datasets_config},
                            timeout=1200  # 20 minutes timeout for batch training
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store training results in session state
                            st.session_state['last_training_results'] = {
                                'timestamp': time.time(),
                                'results': result
                            }
                            
                            # Count actual successful models
                            total_successful = 0
                            if 'results' in result:
                                for dataset_result in result['results'].values():
                                    if dataset_result.get('status') == 'success' and 'results' in dataset_result:
                                        total_successful += len(dataset_result['results'])
                            
                            st.success(f"✅ Batch training completed! {total_successful} model(s) trained successfully!")
                            
                            # Show results for each dataset
                            if 'results' in result:
                                for dataset_name, dataset_result in result['results'].items():
                                    with st.expander(f"📊 Results for {dataset_name}"):
                                        if dataset_result.get('status') == 'success':
                                            models_count = len(dataset_result.get('results', {}))
                                            st.success(f"✅ {models_count} model(s) trained successfully")
                                            # Show results for each model in this dataset
                                            if 'results' in dataset_result:
                                                for model_name, model_result in dataset_result['results'].items():
                                                    st.markdown(f"**{model_name}:**")
                                                    # Improved model type detection
                                                    algo_name = model_result.get('algorithm', '').lower() if 'algorithm' in model_result else model_name.lower()
                                                    classification_keywords = [
                                                        'class', 'logistic', 'svm', 'knn', 'forest', 'tree', 'neighbor'
                                                    ]
                                                    regression_keywords = [
                                                        'regress', 'linear', 'ridge', 'lasso', 'elastic', 'svr', 'bayesian', 'huber', 'quantile'
                                                    ]
                                                    is_classification = any(word in algo_name for word in classification_keywords)
                                                    is_regression = any(word in algo_name for word in regression_keywords)

                                                    # Main metric display
                                                    if is_classification:
                                                        # Extract accuracy from different possible locations
                                                        accuracy = model_result.get('accuracy')
                                                        if accuracy is None:
                                                            accuracy = model_result.get('metrics', {}).get('accuracy')
                                                        if accuracy is None:
                                                            accuracy = model_result.get('test_score')
                                                        if accuracy is not None:
                                                            try:
                                                                accuracy_float = float(accuracy)
                                                                st.info(f"🎯 Accuracy: {accuracy_float:.4f}")
                                                            except (ValueError, TypeError):
                                                                st.info(f"🎯 Accuracy: {accuracy}")
                                                        else:
                                                            st.warning("🎯 Accuracy: Not available")
                                                    elif is_regression:
                                                        # Prefer RMSE, then MSE, then test_score
                                                        rmse = model_result.get('metrics', {}).get('rmse') if 'metrics' in model_result else None
                                                        if rmse is None:
                                                            rmse = model_result.get('rmse')
                                                        if rmse is not None:
                                                            try:
                                                                rmse_float = float(rmse)
                                                                st.info(f"📉 RMSE: {rmse_float:.4f}")
                                                            except (ValueError, TypeError):
                                                                st.info(f"📉 RMSE: {rmse}")
                                                        else:
                                                            st.warning("📉 RMSE: Not available")
                                                    else:
                                                        st.info("ℹ️ Model type not detected. Metrics below.")

                                                    # Show additional metrics if available
                                                    if 'metrics' in model_result and model_result['metrics']:
                                                        st.markdown("**Detailed metrics:**")
                                                        st.json(model_result['metrics'])

                                                    # Visualizations: ROC curve (classification) and learning curve (all)
                                                    col_viz1, col_viz2 = st.columns(2)
                                                    with col_viz1:
                                                        if is_classification:
                                                            try:
                                                                roc_response = requests.get(f'http://localhost:8000/visualization/{model_name}/roc_curve', timeout=15)
                                                                content_type = roc_response.headers.get('content-type', '')
                                                                content_len = len(roc_response.content)
                                                                if roc_response.status_code == 200 and content_type.startswith('image') and content_len > 100:
                                                                    st.image(roc_response.content, caption=f"ROC Curve - {model_name}")
                                                                elif roc_response.status_code == 200 and content_len > 0 and not content_type.startswith('image'):
                                                                    st.warning(f"ROC curve not available (backend returned non-image content).")
                                                                else:
                                                                    st.warning("ROC curve not available.")
                                                            except Exception as e:
                                                                st.warning(f"ROC curve error: {e}")
                                                    with col_viz2:
                                                        try:
                                                            learning_response = requests.get(f'http://localhost:8000/visualization/{model_name}/learning_curve', timeout=60)
                                                            content_type = learning_response.headers.get('content-type', '')
                                                            content_len = len(learning_response.content)
                                                            if learning_response.status_code == 200 and content_type.startswith('image') and content_len > 100:
                                                                st.image(learning_response.content, caption=f"Learning Curve - {model_name}")
                                                            elif learning_response.status_code == 200 and content_len > 0 and not content_type.startswith('image'):
                                                                st.warning(f"Learning curve not available (backend returned non-image content).")
                                                            else:
                                                                st.warning("Learning curve not available.")
                                                        except Exception as e:
                                                            st.warning(f"Learning curve error: {e}")
                                        else:
                                            st.error(f"❌ Training failed for {dataset_name}: {dataset_result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"❌ Batch training failed: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during batch training: {e}")
                        
                        # Additional error handling: check if cluster is healthy
                        cluster_status = get_cluster_status()
                        if "error" in cluster_status:
                            st.error("🔴 El clúster Ray no está disponible. Verifica que los contenedores estén ejecutándose.")
                            st.info("Para resolver el problema, ejecuta: `docker-compose restart`")
                        else:
                            st.info("El clúster está funcionando. El error puede ser temporal. Intenta nuevamente en unos momentos.")
        else:
            st.warning("⚠️ Please configure and select algorithms for at least one dataset before training")
    else:
        # No successfully uploaded files
        if st.session_state['uploaded_files']:
            st.error("❌ Los archivos seleccionados no se han podido procesar correctamente.")
            
            # Check backend connectivity and provide troubleshooting info
            backend_status = check_backend_connectivity()
            
            if backend_status["status"] == "connected":
                st.info("✅ El backend está funcionando. El problema puede estar en el formato de los archivos o en el procesamiento.")
                st.markdown("""
                **Posibles soluciones:**
                - Verifica que los archivos CSV tengan el formato correcto (con encabezados)
                - Asegúrate que los archivos JSON tengan estructura de array de objetos
                - Revisa los logs del backend para más detalles: `docker-compose logs backend`
                """)
            else:
                st.error(f"🔴 Problema de conectividad: {backend_status['message']}")
                st.markdown("""
                **Para solucionar el problema:**
                1. Asegúrate que Docker esté ejecutándose
                2. Ejecuta: `docker-compose up -d` en la carpeta del proyecto
                3. Espera unos segundos para que los contenedores se inicien completamente
                4. Recarga esta página
                """)
        else:
            st.info("💡 Primero selecciona y procesa archivos CSV o JSON para continuar con la configuración de entrenamiento.")



# --- SECTION: PREDICTION ---
if section == "Predicción":
    st.header("🔮 Model Prediction Interface")
    # Get uploaded files and trained models
    try:
        files_response = requests.get('http://localhost:8000/uploaded_files', timeout=10)
        models_response = requests.get('http://localhost:8000/models', timeout=10)
        if files_response.status_code == 200 and models_response.status_code == 200:
            files_data = files_response.json()
            models_data = models_response.json()
            # Build dataset-to-model mapping (robust: match dataset part to uploaded file base name)
            dataset_to_models = {}
            model_to_features = {}
            # Build a set of uploaded file base names (no extension)
            uploaded_files_list = files_data.get('uploaded_files', [])
            uploaded_basenames = set()
            filename_map = {}  # base name -> full filename
            for f in uploaded_files_list:
                base = f['filename'].replace('.csv','').replace('.json','')
                uploaded_basenames.add(base)
                filename_map[base] = f['filename']

            for model in models_data:
                model_name = model.get('name') or model.get('model_id') or model.get('model_name')
                if not model_name:
                    continue
                # Try to extract dataset part: match from rightmost underscore, but check if it matches any uploaded base name
                parts = model_name.split('_')
                matched_dataset = None
                # Try all possible suffixes (from rightmost underscore to left)
                for i in range(1, len(parts)):
                    candidate = '_'.join(parts[i:])
                    if candidate in uploaded_basenames:
                        matched_dataset = candidate
                        break
                # If not found, try last part
                if not matched_dataset and len(parts) > 1 and parts[-1] in uploaded_basenames:
                    matched_dataset = parts[-1]
                # If still not found, try full model name (for legacy)
                if not matched_dataset and model_name in uploaded_basenames:
                    matched_dataset = model_name
                # If still not found, skip (model not mapped to any uploaded dataset)
                if not matched_dataset:
                    continue
                dataset_to_models.setdefault(matched_dataset, []).append(model_name)
                if 'features' in model:
                    model_to_features[model_name] = model['features']

            # Get available datasets (intersection with uploaded files)
            available_datasets = [filename_map[ds] for ds in dataset_to_models if ds in filename_map]
            if not available_datasets:
                st.info("� No datasets with trained models available. Train some models first in the Training section.")
            else:
                selected_dataset = st.selectbox("Select a dataset:", available_datasets)
                dataset_key = selected_dataset.replace('.csv','').replace('.json','')
                # --- Robust preview: fetch from backend preview endpoint ---

                preview_data = None
                preview_columns = None
                preview_error = None
                try:
                    preview_resp = requests.get(f'http://localhost:8000/dataset_preview?dataset={selected_dataset}', timeout=10)
                    if preview_resp.status_code == 200:
                        preview_json = preview_resp.json()
                        preview_data = preview_json.get('preview', [])
                        if preview_data:
                            preview_columns = list(preview_data[0].keys())
                        preview_error = preview_json.get('error')
                    else:
                        preview_error = f"Backend error: {preview_resp.text}"
                except Exception as e:
                    preview_error = f"Preview fetch error: {e}"

                # Always try to get columns from uploaded_files if not set
                if not preview_columns:
                    dataset_info = next((f for f in files_data.get('uploaded_files', []) if f['filename'] == selected_dataset), None)
                    if dataset_info:
                        preview_columns = dataset_info.get('columns', [])

                st.write(f"**Preview for {selected_dataset}:**")
                if preview_data and preview_columns:
                    preview_df = pd.DataFrame(preview_data, columns=preview_columns)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                elif preview_error:
                    st.info(f"No preview available: {preview_error}")
                else:
                    st.info("No preview available for this dataset.")

                # Show models trained on this dataset
                models_for_dataset = dataset_to_models.get(dataset_key, [])
                if not models_for_dataset:
                    st.warning("No models trained on this dataset.")
                else:
                    selected_models = st.multiselect("Select model(s) to use for prediction:", models_for_dataset, default=models_for_dataset[:1])
                    # Feature input UI (organized in max 3 columns)
                    st.markdown("**Enter feature values for prediction:**")
                    # Build list of features (skip target)
                    # Use preview_columns if available, else fallback to dataset_info columns
                    input_features = [col for col in (preview_columns if preview_columns else []) if col.lower() != 'target']
                    feature_inputs = {}
                    
                    # Organize features in max 3 columns
                    max_cols = 3
                    num_features = len(input_features)
                    if num_features > 0:
                        # Calculate number of rows needed
                        rows_needed = (num_features + max_cols - 1) // max_cols
                        
                        for row in range(rows_needed):
                            # Create columns for this row
                            cols_in_row = min(max_cols, num_features - row * max_cols)
                            cols = st.columns(cols_in_row)
                            
                            for col_idx in range(cols_in_row):
                                feature_idx = row * max_cols + col_idx
                                if feature_idx < num_features:
                                    feature_name = input_features[feature_idx]
                                    with cols[col_idx]:
                                        feature_inputs[feature_name] = st.text_input(
                                            f"{feature_name}", 
                                            key=f"predict_{feature_name}_{selected_dataset}"
                                        )
                    if st.button("🔮 Predict", use_container_width=True):
                        # Prepare feature dict for prediction
                        try:
                            features = {k: (float(v) if v.replace('.','',1).isdigit() else v) for k,v in feature_inputs.items() if v != ''}
                            if not features:
                                st.warning("Please enter values for at least one feature.")
                            else:
                                # Show predictions for each selected model
                                for model_name in selected_models:
                                    prediction_request = {
                                        "model_name": model_name,
                                        "features": features
                                    }
                                    prediction_response = requests.post(
                                        'http://localhost:8000/predict',
                                        json=prediction_request,
                                        timeout=30
                                    )
                                    if prediction_response.status_code == 200:
                                        prediction = prediction_response.json()
                                        st.success(f"Model `{model_name}` prediction: {prediction.get('prediction', 'N/A')}")
                                    else:
                                        st.error(f"Prediction failed for `{model_name}`: {prediction_response.text}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
        else:
            st.error("❌ Failed to get uploaded files or models from backend.")
    except Exception as e:
        st.error(f"❌ Error connecting to backend: {e}")
    
    # Debug section (can be removed in production)
    with st.sidebar.expander("🔧 Debug Tools"):
        if st.button("Clear Session State"):
            st.session_state['uploaded_files'] = {}
            st.session_state['file_configs'] = {}
            st.session_state['last_training_results'] = None
            st.success("Session state cleared")
            st.rerun()
        
        if st.button("Show Session State"):
            st.json({
                "uploaded_files": st.session_state.get('uploaded_files', {}),
                "file_configs": st.session_state.get('file_configs', {}),
                "last_training_results": st.session_state.get('last_training_results')
            })
        
        if st.button("Check Backend Uploaded Files"):
            try:
                response = requests.get('http://localhost:8000/uploaded_files', timeout=10)
                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(f"Backend error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
        
        if st.button("Check Trained Models"):
            try:
                response = requests.get('http://localhost:8000/models', timeout=10)
                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(f"Backend error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
                
        if st.button("Check Cluster Status"):
            cluster_status = get_cluster_status()
            st.json(cluster_status)

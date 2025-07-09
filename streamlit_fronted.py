import streamlit as st
import requests
import json
import os
import pandas as pd
import base64
import time

st.set_page_config(page_title="Plataforma distribuida de entrenamiento supervisado", layout="wide")

def ensure_session_state():
    """Initialize all session state variables to prevent KeyError crashes"""
    defaults = {
        'cluster': {'head': {'cpu': 2, 'ram': 4, 'running': False}},
        'uploaded_files': {},
        'file_configs': {},
        'last_training_results': None,
        'last_worker_count': None
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state to prevent crashes
ensure_session_state()

# --- API STATUS ---
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

# --- CLUSTER MANAGEMENT TAB ---
def cluster_tab():
    st.header("🖧 Clúster Ray")
    
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
        st.session_state['last_worker_count'] = current_worker_count
    
    if "error" not in cluster_status:
        
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
        st.markdown("#### 🖧 Nodos")
        
        # Prepare table data
        table_data = []
        
        # Add head node
        nodes = cluster_status.get("node_details", [])
        head_node = None
        if nodes:
            head_node = nodes[0]  # First node is typically the head
        
        # Use realistic CPU values instead of Ray's over-reported values
        head_cpu_raw = head_node.get("Resources", {}).get("CPU", 2.0) if head_node else 2.0
        head_cpu = min(head_cpu_raw, 8)  # Cap at 8 cores for more realistic display
        head_memory = head_node.get("Resources", {}).get("memory", 4e9) / 1e9 if head_node else 4.0
        head_status = "✅ Activo" if head_node and head_node.get("Alive") else "❌ Inactivo"
        
        table_data.append({
            "Nodo": "Head Node (ray-head)",
            "CPU": f"{head_cpu}",
            "RAM (GB)": f"{head_memory:.1f}",
            "Estado": head_status,
            "Tipo": "Líder"
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
                
                status_icon = "✅"
                status_text = "Activo"
                
                table_data.append({
                    "Nodo": f"Worker {worker['number']} ({worker['name']})",
                    "CPU": f"{worker_cpu}",
                    "RAM (GB)": f"{worker_memory:.1f}",
                    "Estado": f"{status_icon} {status_text}",
                    "Tipo": "Trabajador"
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
                        "Estado": "✅ Activo",
                        "Tipo": "Trabajador"
                    })
        
        # Display table with current worker count info
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No se pudo obtener información de los nodos del clúster")
        
        # Node selection for detailed information
        st.markdown("#### 🔍 Información Detallada de Nodos")
        
        # Create list of available nodes for selection
        available_nodes = ["Información General del Clúster"]
        node_details = {}
        
        # Add head node
        if head_node:
            node_name = "Head Node (ray-head)"
            available_nodes.append(node_name)
            node_details[node_name] = head_node
        
        # Add worker nodes
        if worker_details:
            for worker in worker_details:
                worker_name = f"Worker {worker['number']} ({worker['name']})"
                available_nodes.append(worker_name)
                # Create detailed info for worker
                worker_info = {
                    "NodeID": worker.get('name', 'N/A'),
                    "Alive": True,
                    "Resources": worker.get('resources', {}),
                    "WorkerNumber": worker['number']
                }
                node_details[worker_name] = worker_info
        else:
            # Fallback to Ray cluster nodes if API data not available
            for i, worker_node in enumerate(worker_nodes):
                if worker_node.get("Alive"):
                    worker_name = f"Worker {i+1} (ray-worker-{i+1})"
                    available_nodes.append(worker_name)
                    node_details[worker_name] = worker_node
        
        # Node selection dropdown
        selected_node = st.selectbox(
            "Seleccionar nodo para ver información detallada:",
            available_nodes,
            help="Selecciona un nodo específico para ver su información completa"
        )
        
        # Show detailed information based on selection
        if selected_node == "Información General del Clúster":
            st.markdown("**📊 Resumen General del Clúster:**")
            general_info = {
                "Total de Nodos": len(cluster_status.get("node_details", [])),
                "Nodos Activos": len([n for n in cluster_status.get("node_details", []) if n.get("Alive")]),
                "Workers Disponibles": len(worker_details) if worker_details else len(worker_nodes),
                "Estado del Clúster": "Activo" if cluster_status.get("node_details") else "Inactivo"
            }
            
            # Display as metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodos", general_info["Total de Nodos"])
            with col2:
                st.metric("Nodos Activos", general_info["Nodos Activos"])
            with col3:
                st.metric("Workers", general_info["Workers Disponibles"])
            with col4:
                st.metric("Estado", general_info["Estado del Clúster"])
            
            # Show full cluster status in a more organized way
            st.markdown("**🔧 Configuración Completa del Clúster:**")
            st.json(cluster_status)
        else:
            # Show specific node information
            if selected_node in node_details:
                node_info = node_details[selected_node]
                st.markdown(f"**📋 Información Detallada: {selected_node}**")
                
                # Show key metrics for the selected node
                if "Resources" in node_info:
                    resources = node_info["Resources"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cpu_count = resources.get("CPU", "N/A")
                        if isinstance(cpu_count, (int, float)):
                            cpu_count = min(cpu_count, 8) if "Head" in selected_node else min(cpu_count, 4)
                        st.metric("CPU Cores", f"{cpu_count}")
                    
                    with col2:
                        memory = resources.get("memory", 0)
                        if isinstance(memory, (int, float)):
                            memory_gb = memory / 1e9
                            st.metric("RAM (GB)", f"{memory_gb:.1f}")
                        else:
                            st.metric("RAM (GB)", "N/A")
                    
                    with col3:
                        status = "✅ Activo" if node_info.get("Alive", False) else "❌ Inactivo"
                        st.metric("Estado", status)
                
                # Show complete node information in table format
                st.markdown("**🔧 Información Completa del Nodo:**")
                
                # Convert node info to a more readable table format
                def format_node_info_table(node_data):
                    """Convert node information to a structured table format"""
                    table_data = []
                    
                    def add_row(key, value, category="General"):
                        if isinstance(value, dict):
                            # For nested dictionaries, add each key-value pair
                            for sub_key, sub_value in value.items():
                                table_data.append({
                                    "Categoría": category,
                                    "Propiedad": f"{key}.{sub_key}",
                                    "Valor": str(sub_value)
                                })
                        elif isinstance(value, list):
                            # For lists, join elements or show count
                            if len(value) > 0:
                                table_data.append({
                                    "Categoría": category,
                                    "Propiedad": key,
                                    "Valor": f"Lista con {len(value)} elementos: {', '.join(map(str, value[:3]))}" + ("..." if len(value) > 3 else "")
                                })
                            else:
                                table_data.append({
                                    "Categoría": category,
                                    "Propiedad": key,
                                    "Valor": "Lista vacía"
                                })
                        else:
                            table_data.append({
                                "Categoría": category,
                                "Propiedad": key,
                                "Valor": str(value)
                            })
                    
                    # Process each key in node_data
                    for key, value in node_data.items():
                        if key == "Resources":
                            add_row(key, value, "Recursos")
                        elif key in ["NodeID", "Alive", "WorkerNumber"]:
                            add_row(key, value, "Identificación")
                        elif key in ["NodeManagerAddress", "NodeManagerPort", "ObjectManagerPort"]:
                            add_row(key, value, "Red")
                        elif key in ["CPU", "memory", "node", "object_store_memory"]:
                            add_row(key, value, "Recursos")
                        else:
                            add_row(key, value, "Otros")
                    
                    return table_data
                
                # Create and display the table
                table_data = format_node_info_table(node_info)
                if table_data:
                    df_node_info = pd.DataFrame(table_data)
                    st.dataframe(df_node_info, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay información detallada disponible para este nodo")
            else:
                st.error("No se pudo obtener información para el nodo seleccionado")
    
    else:
        st.warning(f"Clúster no disponible: {cluster_status['error']}")
        st.info("Esto puede ocurrir si Ray no está completamente inicializado.")

# --- TRAINING TAB ---
def training_tab():
    st.header("🏋🏻‍♀️ Entrenamiento distribuido de Modelos de Machine Learning.")
    st.markdown("Seleccione los datasets deseados en formato csv y json para realizar el procesamiento y entrenamiento distribuido de modelos de ML en el clúster de Ray.")
    
    # Check for existing trained models (no longer shown in UI, but still fetched for prediction section)
    try:
        models_response = requests.get('http://localhost:8000/models', timeout=5)
        if models_response.status_code == 200:
            existing_models = models_response.json()
    except Exception:
        pass  # If check fails, continue normally
    
    # Show recent training results if available
    if st.session_state.get('last_training_results'):
        last_results = st.session_state['last_training_results']
        time_ago = int(time.time() - last_results['timestamp'])
        
        if time_ago < 3600:  # Show if less than 1 hour ago
            minutes_ago = time_ago // 60
            with st.expander(f"📈 Resultados de Entrenamiento Recientes ({minutes_ago} minutos atrás)"):
                result = last_results['results']
                
                if 'results' in result:
                    for dataset_name, dataset_result in result['results'].items():
                        if dataset_result.get('status') == 'success' and 'results' in dataset_result:
                            st.write(f"**{dataset_name}:**")
                            for model_name, model_result in dataset_result['results'].items():
                            
                                accuracy = model_result.get('accuracy')
                                if accuracy is None:
                                    accuracy = model_result.get('metrics', {}).get('accuracy')
                                
                                if accuracy is not None:
                                    try:
                                        # Ensure accuracy is a number before formatting
                                        accuracy_float = float(accuracy)
                                        st.write(f"  - {model_name}: {accuracy_float:.4f}")
                                    except (ValueError, TypeError):
                                        # If conversion fails, display as-is
                                        st.write(f"  - {model_name}: {accuracy}")
                                else:
                                    st.write(f"  - {model_name}: Entrenamiento completado")
    
    st.info("💡 Para realizar el entrenamiento primero suba el dataset deseado y luego seleccione la variable objetivo y los modelos a entrenar")

    uploaded_files = st.file_uploader(
        "Seleccione archivos csv o json para procesamiento distribuido:",
        type=['csv', 'json'],
        accept_multiple_files=True,
        help="Seleccione uno o más archivos csv/json para entrenar distribuido modelos de ML"
    )
    
    if uploaded_files:
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state['uploaded_files']]
        
        if not files_to_process and uploaded_files:
            if st.button("🔄 Forzar reprocesamiento de todos los archivos"):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state['uploaded_files']:
                        del st.session_state['uploaded_files'][uploaded_file.name]
                st.rerun()
        
        if files_to_process:
            st.info(f"🔄 {len(files_to_process)} archivo(s) nuevo(s) detectado(s). Procesando automáticamente...")
            uploaded_count = 0
            
            for i, uploaded_file in enumerate(files_to_process):
                filename = uploaded_file.name
                
                if i > 0:
                    time.sleep(0.5)
                
                try:
                    file_content = uploaded_file.getvalue()
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    
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
                        try:
                            error_details = response.json()
                            error_msg = error_details.get('detail', response.text)
                        except:
                            error_msg = response.text
                        
                        st.error(f"❌ Error procesando {filename} (Código {response.status_code})")
                                        
                        st.session_state['uploaded_files'][filename] = {"error": error_msg}
                        
                except Exception as e:
                    st.error(f"Error de conexión procesando {filename}: {e}")
                    st.session_state['uploaded_files'][filename] = {"error": str(e)}
            
            if uploaded_count > 0:
                st.info(f"{uploaded_count} archivo(s) distribuido(s) en el clúster Ray")
                st.success("🔄 Página actualizada. Los archivos están listos para configuración.")
            elif uploaded_count == 0 and files_to_process:
                st.error("❌ Ningún archivo pudo ser procesado. Verifique los errores arriba.")
                
                with st.expander("Información de diagnóstico"):
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
        else:
            st.success("✅ Todos los archivos han sido procesados y están listos para configuración")
    else:
        st.info("Seleccione el datasets deseado para comenzar el procesamiento")
    
    #Configuración de entrenamiento de archivos
    #Solo mostrar configuración si los archivos se han subido correctamente y tienen datos válidos
    successfully_uploaded_files = {k: v for k, v in st.session_state['uploaded_files'].items() 
                                    if v and 'rows' in v and 'columns' in v}

    if successfully_uploaded_files:
        try:
            response = requests.get('http://localhost:8000/uploaded_files', timeout=5)
            if response.status_code == 200:
                backend_response = response.json()
                backend_files = backend_response.get('files', [])
                
                if isinstance(backend_files, list):
                    missing_files = [f for f in successfully_uploaded_files.keys() if f not in backend_files]
                    if missing_files and len(backend_files) == 0:
                        st.warning(f"⚠️ Detectado que el clúster Ray se reinició. Los archivos necesitan ser vueltos a subir.")
                        for missing_file in missing_files:
                            if missing_file in st.session_state['uploaded_files']:
                                del st.session_state['uploaded_files'][missing_file]
                            if missing_file in st.session_state['file_configs']:
                                del st.session_state['file_configs'][missing_file]
                        
                        successfully_uploaded_files = {k: v for k, v in st.session_state['uploaded_files'].items() 
                                                        if v and 'rows' in v and 'columns' in v}
                        if not successfully_uploaded_files:
                            st.info("💡 Por favor, vuelve a subir los archivos.")
                    elif missing_files:
                
                        st.info(f"Algunos archivos ({missing_files}) no están disponibles en el backend. Esto puede ocurrir si se han agregado/eliminado workers o si el clúster se ha reconfigurado. Si persisten los problemas, vuelve a subir los archivos.")
        except Exception:
            pass  
    
    if successfully_uploaded_files:
        
        for filename, file_info in successfully_uploaded_files.items():
            st.markdown(f"#### Configurar el dataset {filename} para entrenamiento.")
            st.caption(f"{file_info['rows']} filas, {len(file_info['columns'])} columnas")

            if file_info.get('preview'):
                st.markdown("**Resumen preliminar de los datos:**")
                preview_df = pd.DataFrame(file_info['preview'])
                st.dataframe(preview_df, use_container_width=True)

            default_target = "target"
            if "target" in file_info['columns']:
                default_target = "target"
            elif "price" in [col.lower() for col in file_info['columns']]:
                default_target = next(col for col in file_info['columns'] if col.lower() == "price")
            elif "value" in [col.lower() for col in file_info['columns']]:
                default_target = next(col for col in file_info['columns'] if col.lower() == "value")
            elif any("y" == col.lower() for col in file_info['columns']):
                default_target = next(col for col in file_info['columns'] if col.lower() == "y")
            
            try:
                default_index = file_info['columns'].index(default_target)
            except ValueError:
                default_index = 0
            target_column = st.selectbox(
                "Columna Objetivo (Target)",
                file_info['columns'],
                index=default_index,
                key=f"target_{filename}",
                help=f"La columna a predecir. Usualmente llamada 'target', 'price', 'value', o 'y'"
            )

            task_type = "classification" 
            algorithms = ["Decision Tree Classifier", "Logistic Regression", "Random Forest Classifier", "K-Nearest Neighbors"]
            selected_algorithms = st.multiselect(
                "Seleccionar Modelos de Clasificación para Entrenar (puedes seleccionar múltiples)",
                algorithms,
                default=[algorithms[0]], 
                key=f"algos_{filename}",
                help="Puedes seleccionar múltiples modelos de clasificación para entrenar y comparar su rendimiento. Estos modelos predicen categorías (0/1, clases discretas)"
            )

            st.markdown("**Seleccione:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                test_size = st.slider("Tamaño del Test", 0.1, 0.5, 0.2, key=f"test_size_{filename}")
            with col2:
                random_state = st.number_input("Random State (semilla)", 1, 1000, 42, key=f"random_state_{filename}")

            st.session_state['file_configs'][filename] = {
                'task_type': task_type,
                'target_column': target_column,
                'algorithms': selected_algorithms,
                'test_size': test_size,
                'random_state': random_state
            }
        
            if selected_algorithms:
                st.success(f"✅ {len(selected_algorithms)} modelo(s) configurado(s) para {filename}")
            else:
                st.warning("⚠️ Por favor selecciona al menos un modelo")
            st.markdown("---") 
        
        st.subheader("Entrenar todos los Modelos seleccionados")
        
        total_models = 0
        valid_configs = 0
        
        for filename, config in st.session_state['file_configs'].items():
            if filename in successfully_uploaded_files and config.get('algorithms'):
                total_models += len(config['algorithms'])
                valid_configs += 1
        
        if total_models > 0:
            st.info(f"Listo para entrenar {total_models} modelo(s) en {valid_configs} dataset(s)")
            
            if st.button("🚀 Comenzar entrenamiento de Modelos", type="primary", use_container_width=True):
                with st.spinner(f"Entrenando {total_models} modelo(s) en {valid_configs} dataset(s)..."):
                    try:
                        def convert_algorithm_name(algo_name, task_type):
                            """Convert display name to API name"""
                            mapping = {
                                "Decision Tree Classifier": "decision_tree_classifier",
                                "Logistic Regression": "logistic_regression",
                                "Random Forest Classifier": "random_forest_classifier",
                                "K-Nearest Neighbors": "k_nearest_neighbors"
                            }
                            return mapping.get(algo_name, algo_name.lower().replace(" ", "_"))
                        
                        datasets_config = {}
                        
                        for filename, config in st.session_state['file_configs'].items():
                            if filename in successfully_uploaded_files and config.get('algorithms'):
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
                        
                        response = requests.post(
                            "http://localhost:8000/train_all_datasets",
                            json={"datasets": datasets_config},
                            timeout=1200  
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.session_state['last_training_results'] = {
                                'timestamp': time.time(),
                                'results': result
                            }
                            
                            total_successful = 0
                            if 'results' in result:
                                for dataset_result in result['results'].values():
                                    if dataset_result.get('status') == 'success' and 'results' in dataset_result:
                                        total_successful += len(dataset_result['results'])
                            
                            st.success(f"✅ ¡Entrenamiento completado! {total_successful} modelo(s) entrenado(s) exitosamente!")
                            
                            if 'results' in result:
                                for dataset_name, dataset_result in result['results'].items():
                                    with st.expander(f"📊 Resultados para {dataset_name}"):
                                        if dataset_result.get('status') == 'success':
                                            models_count = len(dataset_result.get('results', {}))
                                            st.success(f"✅ {models_count} modelo(s) entrenado(s) exitosamente")
                                    
                                            if 'results' in dataset_result:
                                                for model_name, model_result in dataset_result['results'].items():
                                                    st.markdown(f"**{model_name}:**")
                                                    st.markdown(f"**{model_name}** (Modelo de Clasificación)")
                                                    
                                                    metrics = model_result.get('metrics', {})
                                                    
                                                    accuracy = metrics.get('accuracy')
                                                    if accuracy is not None:
                                                        try:
                                                            accuracy_float = float(accuracy)
                                                            st.info(f"🎯 Accuracy: {accuracy_float:.4f}")
                                                        except (ValueError, TypeError):
                                                            st.info(f"🎯 Accuracy: {accuracy}")
                                                    else:
                                                        st.warning("🎯 Accuracy: No disponible")
                                                    
                                                    precision = metrics.get('precision')
                                                    if precision is not None:
                                                        try:
                                                            precision_float = float(precision)
                                                            st.info(f"📊 Precision: {precision_float:.4f}")
                                                        except (ValueError, TypeError):
                                                            st.info(f"📊 Precision: {precision}")
                                                    
                                                    recall = metrics.get('recall')
                                                    if recall is not None:
                                                        try:
                                                            recall_float = float(recall)
                                                            st.info(f"📈 Recall: {recall_float:.4f}")
                                                        except (ValueError, TypeError):
                                                            st.info(f"📈 Recall: {recall}")
                                                    
                                                    f1 = metrics.get('f1_score')
                                                    if f1 is not None:
                                                        try:
                                                            f1_float = float(f1)
                                                            st.info(f"🔄 F1-Score: {f1_float:.4f}")
                                                        except (ValueError, TypeError):
                                                            st.info(f"🔄 F1-Score: {f1}")

                                                    if metrics:
                                                        st.markdown("**Métricas detalladas:**")
                                                        st.json(metrics)
                                                        
                                                    st.markdown("**📊 Matriz de Confusión:**")
                                                    try:
                                                        confusion_response = requests.get(f'http://localhost:8000/visualization/{model_name}/confusion_matrix', timeout=60)
                                                        content_type = confusion_response.headers.get('content-type', '')
                                                        content_len = len(confusion_response.content)
                                                        if confusion_response.status_code == 200 and content_type.startswith('image') and content_len > 100:
                                                            st.image(confusion_response.content, caption=f"Matriz de Confusión - {model_name}")
                                                        else:
                                                            st.warning("Matriz de confusión no disponible.")
                                                    except Exception as e:
                                                        st.warning(f"Error en matriz de confusión: {e}")
                                        else:
                                            st.error(f"❌ Entrenamiento falló para {dataset_name}: {dataset_result.get('error', 'Error desconocido')}")
                        else:
                            st.error(f"❌ Entrenamiento por lotes falló: {response.text}")
                            
                    except Exception as e:
                        st.error(f"❌ Error durante el entrenamiento por lotes: {e}")
                        
                        cluster_status = get_cluster_status()
                        if "error" in cluster_status:
                            st.error("❌ El clúster Ray no está disponible. Verifica que los contenedores estén ejecutándose.")
                            st.info("Para resolver el problema, ejecuta: `docker-compose restart`")
                        else:
                            st.info("El clúster está funcionando. El error puede ser temporal. Intenta nuevamente en unos momentos.")
        else:
            st.warning("⚠️ Por favor configura y selecciona algoritmos para al menos un dataset antes del entrenamiento")
    else:
        if st.session_state['uploaded_files']:
            st.error("❌ Los archivos seleccionados no se han podido procesar correctamente.")
            
            backend_status = check_backend_connectivity()
            
            if backend_status["status"] == "connected":
                st.info("✅ El backend está funcionando. El problema puede estar en el formato de los archivos o en el procesamiento.")
            else:
                st.error(f"❌ Problema de conectividad: {backend_status['message']}")
        else:
            st.info("Primero selecciona y procesa el datasets en formato csv o json para continuar con la configuración de entrenamiento.")

def prediction_tab():
    st.header("🚀 Realiza Predicciones con los modelos entrenados.")
    try:
        files_response = requests.get('http://localhost:8000/uploaded_files', timeout=10)
        models_response = requests.get('http://localhost:8000/models', timeout=10)
        if files_response.status_code == 200 and models_response.status_code == 200:
            files_data = files_response.json()
            models_data = models_response.json()
            dataset_to_models = {}
            model_to_features = {}
            uploaded_files_list = files_data.get('uploaded_files', [])
            uploaded_basenames = set()
            filename_map = {} 
            for f in uploaded_files_list:
                base = f['filename'].replace('.csv','').replace('.json','')
                uploaded_basenames.add(base)
                filename_map[base] = f['filename']

            for model in models_data:
                model_name = model.get('name') or model.get('model_id') or model.get('model_name')
                if not model_name:
                    continue
                parts = model_name.split('_')
                matched_dataset = None
                for i in range(1, len(parts)):
                    candidate = '_'.join(parts[i:])
                    if candidate in uploaded_basenames:
                        matched_dataset = candidate
                        break

                if not matched_dataset and len(parts) > 1 and parts[-1] in uploaded_basenames:
                    matched_dataset = parts[-1]
                
                if not matched_dataset and model_name in uploaded_basenames:
                    matched_dataset = model_name
                
                if not matched_dataset:
                    continue
                dataset_to_models.setdefault(matched_dataset, []).append(model_name)
                if 'features' in model:
                    model_to_features[model_name] = model['features']

    
            available_datasets = [filename_map[ds] for ds in dataset_to_models if ds in filename_map]
            if not available_datasets:
                st.info("No hay datasets con modelos entrenados disponibles. Entrena algunos modelos primero en la sección de Entrenamiento.")
            else:
                selected_dataset = st.selectbox("Selecciona un dataset:", available_datasets)
                dataset_key = selected_dataset.replace('.csv','').replace('.json','')
                
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
                        preview_error = f"Error del backend: {preview_resp.text}"
                except Exception as e:
                    preview_error = f"Error obteniendo vista previa: {e}"

                if not preview_columns:
                    dataset_info = next((f for f in files_data.get('uploaded_files', []) if f['filename'] == selected_dataset), None)
                    if dataset_info:
                        preview_columns = dataset_info.get('columns', [])

                st.write(f"**Resumen preliminar para {selected_dataset}:**")
                if preview_data and preview_columns:
                    preview_df = pd.DataFrame(preview_data, columns=preview_columns)
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                elif preview_error:
                    st.info(f"Resumen preliminar no disponible: {preview_error}")
                else:
                    st.info("Resumen preliminar no disponible para este dataset.")

                models_for_dataset = dataset_to_models.get(dataset_key, [])
                if not models_for_dataset:
                    st.warning("No hay modelos entrenados en este dataset.")
                else:
                    selected_models = st.multiselect("Selecciona modelo(s) para usar en la predicción:", models_for_dataset, default=models_for_dataset[:1])
                    
                    st.markdown("**📝 Formulario de Predicción:**")
                    st.markdown("Completa los siguientes campos para realizar la predicción:")
                    
                    input_features = [col for col in (preview_columns if preview_columns else []) if col.lower() != 'target']
                    feature_inputs = {}
                    
                    if num_features := len(input_features):
                        with st.container():
                            st.markdown("---")
                            
                            for i, feature_name in enumerate(input_features):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.markdown(f"**{feature_name}:**")
                                    st.caption(f"Campo {i+1} de {num_features}")
                                
                                with col2:
                                    # Use different input types based on feature name patterns
                                    if any(keyword in feature_name.lower() for keyword in ['age', 'year', 'count', 'number']):
                                        feature_inputs[feature_name] = st.number_input(
                                            f"Valor para {feature_name}",
                                            key=f"predict_{feature_name}_{selected_dataset}",
                                            help=f"Ingresa un valor numérico para {feature_name}"
                                        )
                                    elif any(keyword in feature_name.lower() for keyword in ['price', 'cost', 'amount', 'salary', 'income']):
                                        feature_inputs[feature_name] = st.number_input(
                                            f"Valor para {feature_name}",
                                            min_value=0.0,
                                            key=f"predict_{feature_name}_{selected_dataset}",
                                            help=f"Ingresa un valor monetario para {feature_name}"
                                        )
                                    elif any(keyword in feature_name.lower() for keyword in ['rate', 'ratio', 'percentage', 'score']):
                                        feature_inputs[feature_name] = st.slider(
                                            f"Valor para {feature_name}",
                                            min_value=0.0,
                                            max_value=100.0,
                                            value=50.0,
                                            key=f"predict_{feature_name}_{selected_dataset}",
                                            help=f"Selecciona un valor entre 0 y 100 para {feature_name}"
                                        )
                                    else:
                                        feature_inputs[feature_name] = st.text_input(
                                            f"Valor para {feature_name}",
                                            key=f"predict_{feature_name}_{selected_dataset}",
                                            help=f"Ingresa el valor para {feature_name}"
                                        )
                                
                                # Add some spacing between fields
                                if i < len(input_features) - 1:
                                    st.markdown("")
                            
                            st.markdown("---")
                    
                    # Prediction button with enhanced styling
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("🚀 Realizar Predicción", type="primary", use_container_width=True):
                            # Prepare feature dict for prediction
                            try:
                                # Enhanced feature processing
                                features = {}
                                for k, v in feature_inputs.items():
                                    if v != '' and v is not None:
                                        # Try to convert to appropriate type
                                        if isinstance(v, (int, float)):
                                            features[k] = float(v)
                                        elif isinstance(v, str) and v.replace('.','',1).replace('-','',1).isdigit():
                                            features[k] = float(v)
                                        else:
                                            features[k] = v
                                
                                if not features:
                                    st.warning("⚠️ Por favor completa al menos un campo para realizar la predicción.")
                                else:
                                    st.success(f"✅ Realizando predicción con {len(features)} características...")
                                    
                                    # Show predictions for each selected model
                                    for model_name in selected_models:
                                        with st.container():
                                            st.markdown(f"### 🤖 Predicción del modelo: `{model_name}`")
                                            
                                            prediction_request = {
                                                "model_name": model_name,
                                                "features": features
                                            }
                                            
                                            with st.spinner(f"Procesando predicción con {model_name}..."):
                                                prediction_response = requests.post(
                                                    'http://localhost:8000/predict',
                                                    json=prediction_request,
                                                    timeout=30
                                                )
                                            
                                            if prediction_response.status_code == 200:
                                                prediction = prediction_response.json()
                                                pred_value = prediction.get('prediction', 'N/A')
                                                
                                                # Enhanced prediction display
                                                if isinstance(pred_value, (int, float)):
                                                    pred_class = int(round(pred_value))
                                                    
                                                    # Create a nice result display
                                                    col1, col2 = st.columns([1, 1])
                                                    with col1:
                                                        st.metric(
                                                            label="Predicción",
                                                            value=f"Clase {pred_class}",
                                                            help="Resultado de la clasificación"
                                                        )
                                                    with col2:
                                                        st.metric(
                                                            label="Confianza",
                                                            value=f"{abs(pred_value):.3f}",
                                                            help="Valor de confianza del modelo"
                                                        )
                                                    
                                                    # Color-coded result
                                                    if pred_class == 1:
                                                        st.success(f"🎯 **Resultado: POSITIVO** (Clase {pred_class})")
                                                    else:
                                                        st.info(f"🎯 **Resultado: NEGATIVO** (Clase {pred_class})")
                                                        
                                                else:
                                                    st.success(f"🎯 **Predicción:** {pred_value}")
                                                    
                                            else:
                                                st.error(f"❌ Error en la predicción para `{model_name}`: {prediction_response.text}")
                                            
                                            st.markdown("---")
                                            
                            except Exception as e:
                                st.error(f"❌ Error procesando la predicción: {e}")
        else:
            st.error("❌ Falló al obtener archivos subidos o modelos del backend.")
    except Exception as e:
        st.error(f"❌ Error conectando al backend: {e}")


st.title("Plataforma distribuida de entrenamiento supervisado")

# Single page layout - all sections in sequence
cluster_tab()

st.markdown("---")  # Separator between sections

training_tab()

st.markdown("---")  # Separator between sections

prediction_tab()

# Show backend status and clear memory button in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Estado de la API (backend)")
try:
    response = requests.get('http://localhost:8000/health', timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ API disponible")
    else:
        st.sidebar.warning("⚠️ API con problemas")
except Exception:
    st.sidebar.error("❌ API no disponible")

# Add clear memory button to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Herramientas")
if st.sidebar.button("🧹 Limpiar Memoria", help="Limpia toda la memoria y los modelos entrenados para evitar desbordamientos y empezar de cero.", use_container_width=True):
    with st.spinner("Limpiando memoria y recursos del clúster..."):
        try:
            response = requests.post("http://localhost:8000/clear_memory", timeout=60)
            if response.status_code == 200:
                result = response.json()
                
                # Clear frontend session state
                st.session_state['uploaded_files'] = {}
                st.session_state['file_configs'] = {}
                st.session_state['last_training_results'] = None
                
                # Show detailed cleanup results
                actors_cleared = result.get('actors_cleared', 0)
                datasets_cleared = result.get('datasets_cleared', 0)
                remaining_actors = result.get('remaining_actors', 0)
                remaining_files = result.get('remaining_files', 0)
                failed_actors = result.get('failed_actors', [])
                
                if actors_cleared > 0 or datasets_cleared > 0:
                    st.sidebar.success(f"✅ Memoria limpiada: {actors_cleared} modelos y {datasets_cleared} datasets eliminados")
                    
                    if remaining_actors > 0 or remaining_files > 0:
                        st.sidebar.warning(f"⚠️ Quedan: {remaining_actors} actores y {remaining_files} archivos")
                    
                    if failed_actors:
                        st.sidebar.warning(f"⚠️ No se pudieron eliminar {len(failed_actors)} actores")
                else:
                    st.sidebar.info("ℹ️ No había datos para limpiar")
                
                st.rerun()
            else:
                try:
                    error_msg = response.json().get('detail', response.text)
                except Exception:
                    error_msg = response.text
                st.sidebar.error(f"❌ Error al limpiar memoria: {error_msg}")
        except Exception as e:
            st.sidebar.error(f"❌ Error de conexión: {e}")
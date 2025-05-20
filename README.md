### **Proyecto Plataforma de Aprendizaje Supervisado Distribuido**  

#### **Descripción:**  
El proyecto consiste en desarrollar una plataforma capaz de entrenar y desplegar modelos de **aprendizaje supervisado** de manera distribuida, utilizando **Ray**, **Docker** y **Scikit-Learn**. La plataforma debe procesar conjuntos de datos etiquetados, entrenar múltiples modelos en paralelo y ponerlos en producción mediante una API funcional. El sistema debe reflejar los conceptos de computación distribuida aprendidos en el curso, garantizando escalabilidad, tolerancia a fallos y eficiencia.  

#### **Fases del Proyecto**  

1. **Entrenamiento Distribuido**  
   - Implementar un sistema que permita el entrenamiento simultáneo de múltiples modelos de *machine learning* sobre un mismo dataset.  
   - Soporte para cargar y procesar datos en un entorno distribuido.  

2. **Despliegue de Modelos (Serving)**  
   - Desarrollo de una **API** REST o programática para interactuar con los modelos entrenados.
   - Integración con contenedores Docker para garantizar portabilidad y reproducibilidad con autodescubrimiento.  

3. **Monitoreo y Visualización**  
   - Generación de gráficas que muestren:  
     - Métricas de rendimiento durante el entrenamiento (ej: precisión, pérdida).  
     - Estadísticas de inferencia en producción (ej: latencia, uso de recursos).  

#### **Criterios de Evaluación**  
- ✅ Diseño de un **sistema distribuido** que cumpla con las funcionalidades básicas y opcionales.  
- ✅ Implementación de **tolerancia a fallos** (ej: replicación de nodos, *autodescubrimiento*).  
- ✅ Capacidad de entrenar **múltiples datasets secuencialmente** en una misma ejecución.  
- ✅ Uso eficiente de **Ray** para gestión de tareas y recursos.  

#### **Funcionalidades Adicionales**  
1. Entrenamiento y *serving* de **varios datasets simultáneamente**.  
2. **Estadísticas avanzadas**: Comparativa de modelos, análisis de tendencias, etc.  
3. Eliminación del **punto único de fallo** en el líder del clúster.  
4. **Interfaz gráfica** (GUI) para gestión y visualización del sistema.  
5. **Seguridad**: Encriptación de comunicaciones y autenticación de nodos.  

---

- **Entrega**: 22 Junio 11:59:59 pm

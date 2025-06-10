# Usar una imagen base de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos del proyecto dentro del contenedor
COPY . /app

# Instalar las dependencias necesarias
RUN pip install -r requirements.txt

# Ejecutar el script de entrenamiento cuando el contenedor inicie
CMD ["python", "train.py"]

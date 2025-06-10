# Usar una imagen base de Python
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la API al contenedor
COPY . /app

# Instalar dependencias
RUN pip install -r requirements.txt

# Exponer el puerto 5000 para la API REST
EXPOSE 5000

# Ejecutar la API al iniciar el contenedor
CMD ["python", "app.py"]

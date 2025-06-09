FROM python:3.9
WORKDIR /app
COPY main.py .
RUN pip install fastapi uvicorn scikit-learn joblib
CMD ["python", "main.py"]

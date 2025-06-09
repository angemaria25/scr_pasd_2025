FROM python:3.9
WORKDIR /app
COPY app.py .
RUN pip install streamlit pandas
CMD ["streamlit", "run", "app.py", "--server.port=8501"]

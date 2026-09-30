# Build from the project root:
#   docker build -t energy-demand-forecasting-app -f app/Dockerfile .
# Run:
#   docker run -p 8501:8501 energy-demand-forecasting-app

FROM python:3.11-slim

WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/sample/ data/sample/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

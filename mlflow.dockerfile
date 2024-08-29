FROM python:3.10-slim

RUN pip install mlflow==2.15.1

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--artifacts-location", "/home/mlartifacts", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]
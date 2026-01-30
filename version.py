import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

MODEL_NAME = "fraud_detection_model"
MODEL_VERSION = "7"  # sua versão boa

# cria/atualiza alias Production
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Production",
    version=MODEL_VERSION
)

print("✅ Alias Production criado com sucesso!")

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model(
    "models:/fraud_detection_model@Production"
)

print(model)
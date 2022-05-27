from flask import Flask, request
import pandas as pd
import mlflow

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def home():
    data = request.get_json()["data"]
    df = pd.DataFrame.from_dict(data)

    mlflow.set_experiment("Churn Prediction")
    last_run = dict(mlflow.search_runs().sort_values(by="start_time", ascending=False).iloc[0])
    artifact_uri = last_run["artifact_uri"]
    model = mlflow.sklearn.load_model(artifact_uri + "/model_pipeline")

    predictions = model.predict(df)
    output = predictions[0]

    return "Churn Prediction {}".format(output)


if __name__ == "__main__":
    app.run()

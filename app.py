from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def home():
    data = request.get_json()["data"]
    df = pd.DataFrame.from_dict(data)

    model = joblib.load("src/models/GB_churn_imovel_web.pkl")
    predictions = model.predict(df)

    return predictions


if __name__ == "__main__":
    app.run()

from flask import Flask, request
from src.predict import Predicter

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def home():
    predicter = Predicter(request.get_json()["data"])
    return predicter.run()

if __name__ == "__main__":
    app.run()

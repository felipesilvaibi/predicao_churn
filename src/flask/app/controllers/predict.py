import pandas as pd
import mlflow

class Predicter:
    def __init__(self, data):
        self.df = pd.DataFrame.from_dict(data)

    def run(self):
        mlflow.set_experiment('Churn Prediction')

        last_run = mlflow.search_runs().sort_values(
            by="start_time", ascending=False).iloc[:1]
        artifact_uri = last_run["artifact_uri"][0]

        model = mlflow.sklearn.load_model(artifact_uri + "/model_pipeline")

        predictions = model.predict(self.df.drop(columns='id_sap'))
        self.df['churn'] = pd.Series(predictions).apply(lambda x: True if x == 1 else False)

        return self.df[['id_sap', 'churn']].to_dict(orient='records')

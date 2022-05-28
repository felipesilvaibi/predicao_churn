# - Importação de bibliotecas
from mlflow.models.signature import infer_signature
from sklearn.metrics import plot_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from datetime import datetime
import mlflow
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import datetime
import warnings
warnings.filterwarnings('ignore')

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from unidecode import unidecode

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# - Carregamento do dataset
df = pd.read_feather('./data/carteira_total.feather')

# - Data Preparation

# * Correção da nomenclatura das colunas
newColumnsName = []
for column_name in df:
  newColumnsName.append(
      re.sub('[^A-Za-z0-9]+', '_', unidecode(column_name).lower()))
df.columns = newColumnsName

# * Criação da variável target
df["upsale_downsale"].replace(
    {"Churn": "churn", "Upsell": "upsell", "Downsell": "downsell", "Ok": "ok"}, inplace=True)
df['churn'] = df['upsale_downsale']
df['churn'].replace(
    {"ok": "0", "upsell": "0", "downsell": "0", "churn": "1"}, inplace=True)

df.drop(columns=["upsale_downsale"], inplace=True)

# * Criação da variável "quantidades mês" (feature engeneering com a variável nativa "mês")
df_grouped = df[['mes', 'id_sap']].groupby(['id_sap']).count().reset_index()
df_grouped.rename(columns = {'mes':'quantidade_mes'}, inplace=True)

df = df.join(df_grouped.set_index('id_sap'), on='id_sap')

# * Criação da variável "status_pagamento" (feature engeneering utilizando fonte de dados externa)
xls = pd.ExcelFile('./data/quality_score.xlsx')
xls.sheet_names

i = 0
for data in xls.sheet_names:
    if i == 0:
        dfQuality = pd.read_excel(xls, data)
        dfQuality['data'] = data
        dfQuality.rename(columns={'Classificação Pagamento': 'status_pagamento', 'Quality Score Cobrança': 'status_pagamento',
                                  'PFIN': 'status_pagamento', 'PEFIN': 'status_pagamento'}, inplace=True)
    else:
        dfQualityAux = pd.read_excel(xls, data)
        dfQualityAux['data'] = data
        dfQualityAux.rename(columns={'Classificação Pagamento': 'status_pagamento', 'Quality Score Cobrança': 'status_pagamento',
                                     'PFIN': 'status_pagamento', 'PEFIN': 'status_pagamento'}, inplace=True)

        dfQuality = pd.concat([dfQuality, dfQualityAux])

    i+=1

dfQuality['status_pagamento'].replace({'4. Péssimo': 'Pessimo', '2. Regular ': 'Regular', '1. Bom': 'Bom', '3. Ruim': 'Ruim', '5. Novo': 'Novo',
                                       '2. Regular': 'Regular', '1. Bom ': 'Bom', 'lançamentos': np.nan, '5. novo': 'Novo', 0: np.nan}, inplace=True)
dfQuality.dropna(inplace=True)
regex = r'([0-9]{4})-([0-9]{2})-[0-9]{2}'
def fun_replace(data):
    return datetime.datetime.strptime(str(data.group(2)).lower(), '%m').strftime('%b').lower() + data.group(1)[-2:]
df['mes'] = df['mes'].astype(str).str.replace(regex, fun_replace, regex=True)
dfQuality.rename(columns={'ID SAP': 'id_sap', 'data': 'mes'}, inplace=True)
df = df.join(dfQuality.set_index(['id_sap', 'mes']), on=['id_sap', 'mes'])

# * Correção dos valores de colunas categóricas
df['oficina'].replace({'wi': 'WI'}, inplace=True)
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.lower()
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.strip()
df['frequencia_de_faturamento'].replace({'única vez': 'unica_vez'}, inplace=True)
df['equipe'].replace({'Relacionamento': 'RELACIONAMENTO', 'Jumbo': 'JUMBO', 'Resellers': 'RESELLERS', 'Regional DF': 'REGIONAL DF'}, inplace=True)

# * Seleção das colunas mais significativas
df = df[['pf_pj', 'contratado_ofertas_simples', 'utilizado_ofertas_simples',
         'leads_form', 'equipe', 'utilizado_destaque', 'valor_mensal',
         'quantidade_mes', 'status_pagamento', 'churn', 'regiao', 'oficina', 'tipo_de_plano', 'frequencia_de_faturamento']]

# - Modeling
# * Pipeline
df = df[df['status_pagamento'].notna()]

X = df.drop(['churn'], axis=1)
y = df['churn'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# * Criação do Pipeline
numeric_features = ['contratado_ofertas_simples', 'utilizado_ofertas_simples',
                    'leads_form', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes']
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

categorical_features = ['pf_pj', 'equipe', 'status_pagamento',
                        'regiao', 'oficina', 'tipo_de_plano', 'frequencia_de_faturamento']
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown='ignore')),
           ("imputer", SimpleImputer(strategy="constant"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("scaler", RobustScaler(with_centering=False)),
        ("classifier", MLPClassifier(hidden_layer_sizes=(6, 5),
                      random_state=1,
                      learning_rate_init=0.01))
    ]
)

# * Predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# - Monitoramento (ml flow)
# - Criação do experimento
mlflow.set_experiment('Churn Prediction')

# * Registro da signature e do pipeline
signature = infer_signature(X_test, y_pred)
mlflow.sklearn.log_model(pipeline, 'model_pipeline', signature=signature)

# * Registro dos Parâmetros do modelo
params = pipeline.named_steps["classifier"].get_params()
mlflow.log_params(params)

# * Registro da matriz de confusão
plot_confusion_matrix(pipeline, X_test, y_test, display_labels=[
    "positivo", "negativo"], values_format="d")

plt.savefig("mlruns/confusion_matrix_.png")
mlflow.log_artifact("mlruns/confusion_matrix_.png")

plt.close()

# * Registro de métricas
f1 = round(f1_score(y_test, y_pred, average='macro')*100, 2)
accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)

mlflow.log_metric("f1", f1)
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)

# * Finalização do experimento
mlflow.end_run()

# %% [markdown]
# ## Importação de bibliotecas

# %%
import mlflow
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from mlflow.models.signature import infer_signature


# %% [markdown]
# ## ML FLOW

# %% [markdown]
# ## Carregamento do dataset

# %%
df = pd.read_feather('./data/carteira_total.feather')


# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Correção da nomenclatura das colunas

# %%
newColumnsName = []
for column_name in df:
  newColumnsName.append(
      re.sub('[^A-Za-z0-9]+', '_', unidecode(column_name).lower()))
df.columns = newColumnsName


# %% [markdown]
# ### Criação da variável target

# %%
df["upsale_downsale"].replace(
    {"Churn": "churn", "Upsell": "upsell", "Downsell": "downsell", "Ok": "ok"}, inplace=True)
df['churn'] = df['upsale_downsale']
df['churn'].replace(
    {"ok": "0", "upsell": "0", "downsell": "0", "churn": "1"}, inplace=True)

df.drop(columns=["upsale_downsale"], inplace=True)


# %% [markdown]
# ### Criação da variável "quantidades mês" (feature engeneering com a variável nativa "mês")

# %%
df_grouped = df[['mes', 'id_sap']].groupby(['id_sap']).count().reset_index()

df_grouped.rename(columns = {'mes':'quantidade_mes'}, inplace=True)

# %%
df = df.join(df_grouped.set_index('id_sap'), on='id_sap')

# %% [markdown]
# ### Criação da variável "status_pagamento" (feature engeneering utilizando fonte de dados externa)

# %%
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

# %%
dfQuality['status_pagamento'].replace({'4. Péssimo': 'Pessimo', '2. Regular ': 'Regular', '1. Bom': 'Bom', '3. Ruim': 'Ruim', '5. Novo': 'Novo',
                                       '2. Regular': 'Regular', '1. Bom ': 'Bom', 'lançamentos': np.nan, '5. novo': 'Novo', 0: np.nan}, inplace=True)


# %%
dfQuality.dropna(inplace=True)


# %%
regex = r'([0-9]{4})-([0-9]{2})-[0-9]{2}'


def fun_replace(data):
    return datetime.datetime.strptime(str(data.group(2)).lower(), '%m').strftime('%b').lower() + data.group(1)[-2:]


df['mes'] = df['mes'].astype(str).str.replace(regex, fun_replace, regex=True)


# %%
dfQuality.rename(columns={'ID SAP': 'id_sap', 'data': 'mes'}, inplace=True)


# %%
df = df.join(dfQuality.set_index(['id_sap', 'mes']), on=['id_sap', 'mes'])


# %% [markdown]
# ### Correção dos valores de colunas categóricas

# %%
df['oficina'].replace({'wi': 'WI'}, inplace=True)

# %%
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.lower()


# %%
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.strip()


# %%
df['frequencia_de_faturamento'].replace(
    {'única vez': 'unica_vez'}, inplace=True)


# %%
df['equipe'].replace({'Relacionamento': 'RELACIONAMENTO', 'Jumbo': 'JUMBO',
                     'Resellers': 'RESELLERS', 'Regional DF': 'REGIONAL DF'}, inplace=True)


# %% [markdown]
# ### Drop de colunas com valores inutilizáveis

# %%
df.drop(['contratado_freemium', 'utilizado_freemium'], axis=1, inplace=True)


# %% [markdown]
# ### Tratamento de valores nulos

# %%
numericalColumns = df.select_dtypes(include=np.number)

df = df[df['status_pagamento'].notna()]
df[numericalColumns.columns] = numericalColumns.fillna(
    numericalColumns.median())

for col in numericalColumns:
    df[col][df[col] < 1] = 1


# %% [markdown]
# ### Seleção das colunas mais significativas

# %%
df = df[['pf_pj', 'contratado_ofertas_simples', 'utilizado_ofertas_simples',
         'leads_form', 'equipe', 'utilizado_destaque', 'valor_mensal',
         'quantidade_mes', 'status_pagamento', 'churn', 'regiao', 'oficina', 'tipo_de_plano', 'frequencia_de_faturamento']]


# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Tratamento de variáveis categóricas

# %%
dummy_df = pd.get_dummies(df[[
    'pf_pj', 'equipe', 'status_pagamento', 'regiao', 'oficina', 'tipo_de_plano', 'frequencia_de_faturamento']], drop_first=True)
df = df.join(dummy_df)
df.drop(columns=['pf_pj', 'equipe', 'status_pagamento', 'regiao',
        'oficina', 'tipo_de_plano', 'frequencia_de_faturamento'], inplace=True)


# %% [markdown]
# ### Data split

# %%
X = df.drop(['churn'], axis=1)
y = df['churn'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

X_train = pd.DataFrame(X_train, columns=X_train.columns)
y_train = pd.Series(y_train)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2


# %% [markdown]
# ### Tratamento de dados numéricos

# %%
def trataDados(df_dados, target):
    # Balanceamento dos dados
    sm = SMOTE(sampling_strategy='minority', random_state=7)
    df_dados, target = sm.fit_resample(df_dados, target.filter(df_dados.index))

    # Normalização dos dados
    numericalColumns = ['contratado_ofertas_simples', 'utilizado_ofertas_simples', 'leads_form',
                        'utilizado_destaque', 'valor_mensal', 'quantidade_mes']
    df_dados[numericalColumns] = RobustScaler(
    ).fit_transform(df_dados[numericalColumns])

    return df_dados, target


# %% [markdown]
# ### Pipeline

# %%
# df = df[df['status_pagamento'].notna()]

# %%
# numeric_features = ['contratado_ofertas_simples', 'utilizado_ofertas_simples', 'leads_form', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes']
# numeric_transformer = Pipeline(
#     steps=[("imputer", SimpleImputer(strategy="median"))]
# )

# categorical_features = ['pf_pj', 'equipe', 'status_pagamento', 'regiao', 'oficina', 'tipo_de_plano', 'frequencia_de_faturamento']
# categorical_transformer = OneHotEncoder()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#         ("cat", categorical_transformer, categorical_features),
#     ]
# )


# %%
# numeric_features = ['contratado_ofertas_simples', 'utilizado_ofertas_simples',
#                     'leads_form', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes']
# numeric_transformer = Pipeline(
#     steps=[('scaler', RobustScaler())]
# )

# posprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numeric_transformer, numeric_features),
#     ]
# )


# %%
# clf = Pipeline(
#     steps=[
#         # ("preprocessor", preprocessor),
#         ('sampling', SMOTE(sampling_strategy='minority', random_state=0)),
#         ("posprocessor", posprocessor),
#            ("classifier", MLPClassifier(hidden_layer_sizes=(6, 5),
#                                          random_state=1,
#                                          verbose=True,
#                                          learning_rate_init=0.01))]
# )

# X = df.drop(['churn'], axis=1)
# y = df['churn'].astype(int)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=1)

# clf.fit(X_train, y_train)
# predictDataSet = clf.predict(X_test)

# f1 = round(f1_score(y_test, y_pred, average='macro')*100, 2)
# accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
# precision = round(precision_score(y_test, predictDataSet)*100, 2)
# recall = round(recall_score(y_test, predictDataSet)*100, 2)

# print(f"F1 Score: {f1}%")
# print(f"Accuracy Score: {accuracy}%")
# print(f"Precision Score: {precision}%")
# print(f"Recall Score: {recall}%")

# plot_confusion_matrix(clf, X_test, y_test, display_labels=[
#     "positivo", "negativo"], values_format="d")
# plt.grid(False)
# plt.show()

# plot_roc_curve(clf, X_test, y_test)
# plt.show()


# %% [markdown]
# ### Execução do Modelo

# %%
X_train, y_train = trataDados(X_train, y_train)
X_val, y_val = trataDados(X_val, y_val)
X_test, y_test = trataDados(X_test, y_test)

# %%
def executeModel(model, X_train, y_train, X_test, y_test):
    ## Cria o experimento (mlflow)
    mlflow.set_experiment('Churn Prediction')
    
    ## Inicia o experimento (mlflow)
    mlflow.start_run()

    model.fit(X_train, y=y_train)

    y_pred = model.predict(X_test)

    f1 = round(f1_score(y_test, y_pred, average='macro')*100, 2)
    accuracy = round(accuracy_score(y_test, y_pred)*100, 2)
    precision = round(precision_score(y_test, y_pred)*100, 2)
    recall = round(recall_score(y_test, y_pred)*100, 2)

    ## Registro de métricas (mlflow)
    mlflow.log_metric("_f1", f1)
    mlflow.log_metric("_accuracy", accuracy)
    mlflow.log_metric("_precision", precision)
    mlflow.log_metric("_recall", recall)

    ## Registro da signature e do pipeline (mlflow)
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, 'model_pipeline', signature=signature)
    params = model.get_params()
    mlflow.log_params(params)

    plot_confusion_matrix(model, X_test, y_test, display_labels=[
        "positivo", "negativo"], values_format="d")

    ## Registro de Artefato confusion matrix (mlflow)
    plt.savefig("mlruns/confusion_matrix_.png")
    mlflow.log_artifact("mlruns/confusion_matrix_.png")
    plt.close()

    plot_roc_curve(model, X_test, y_test)
    
    ## Registro de Artefato curva roc (mlflow)
    plt.savefig("mlruns/roc_curve_.png")
    mlflow.log_artifact("mlruns/roc_curve_.png")
    plt.close()

    ## Finalização da execução (mlflow)
    mlflow.end_run()


# %%
executeModel(MLPClassifier(hidden_layer_sizes=(6, 5),
                           random_state=42,
                           learning_rate_init=0.01), X_train, y_train, X_val, y_val)




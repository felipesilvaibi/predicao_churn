# Importação de bibliotecas
import joblib
import datetime

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from unidecode import unidecode

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Carregamento do dataset
df = pd.read_feather('data/carteira_total.feather')

# Padronização do nome das colunas
newColumnsName = []
for column_name in df:
    newColumnsName.append(re.sub('[^A-Za-z0-9]+', '_', unidecode(column_name).lower()))
df.columns = newColumnsName

# Criação da coluna churn, e definição do seu valor com base nos valores da coluna upsale_downsale
df["upsale_downsale"].replace({"Churn": "churn", "Upsell": "upsell", "Downsell": "downsell", "Ok": "ok"}, inplace=True)
df['churn'] = df['upsale_downsale']
df['churn'].replace({"ok": "0", "upsell": "0", "downsell": "0", "churn": "1"}, inplace=True)
df['churn'] = df['churn'].astype(int)

df.drop(columns = ["upsale_downsale"], inplace = True)

# Adição da coluna quantidade_mes respectiva a quantidade de meses os ID SAP é cliente
df_grouped = df[['mes', 'id_sap']].groupby(['id_sap']).count().reset_index()
df_grouped.rename(columns = {'mes':'quantidade_mes'}, inplace=True)
df = df.join(df_grouped.set_index('id_sap'), on='id_sap')

# Definição do quality score
xls = pd.ExcelFile('data/quality_score.xlsx')
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
dfQuality.rename(columns={'ID SAP': 'id_sap', 'data' : 'mes'}, inplace=True)
df = df.join(dfQuality.set_index(['id_sap', 'mes']), on=['id_sap', 'mes'])

# Tratando valores de colunas categóricas
df['oficina'].replace({'wi': 'WI'}, inplace=True)
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.lower()
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.strip()
df['frequencia_de_faturamento'].replace({'única vez': 'unica_vez'}, inplace=True)
df['equipe'].replace({'Relacionamento': 'RELACIONAMENTO', 'Jumbo': 'JUMBO',
                    'Resellers': 'RESELLERS', 'Regional DF': 'REGIONAL DF'}, inplace=True)
df.drop(['contratado_freemium', 'utilizado_freemium'], axis=1, inplace=True)
df = df[['pf_pj','leads_form', 'total_contratado', 'total_de_listings', 'equipe', 'utilizado_super_destaques', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes', 'status_pagamento', 'churn']].copy()

# Separação em treino e teste
X = df.drop(['churn'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

X_train = pd.DataFrame(X_train, columns=X_train.columns)
y_train = pd.Series(y_train)

# Criação das funções do pipeline
def fnDropStatusPagamentoNa(df):
    df.dropna(subset=['status_pagamento'], inplace=True)

def fnGetDummies(df):
    dfCategoricas = pd.get_dummies(df.select_dtypes(include=['object']), columns=['pf_pj', 'equipe', 'status_pagamento'])
    df = df.join(dfCategoricas)
    df.drop(columns=['pf_pj', 'equipe', 'status_pagamento'], inplace=True)

def gnFillNumericalNaWithMedian(df):
    dfNumerical = df.select_dtypes(include=np.number).copy()
    dfNumerical = dfNumerical.fillna(dfNumerical.median()).copy()
    df.drop(columns=df.select_dtypes(include=np.number).columns, inplace=True)
    df = df.join(dfNumerical)

def fnReplaceNegativeWithZero(df):
    dfNumerical = df.select_dtypes(include=np.number)
    for col in dfNumerical.columns:
        df[col][dfNumerical[col] < 0] = 0

def fnApplyRobustScaler(df):
    dfNumerical = df.select_dtypes(include=np.number)
    dfNumerical[['leads_form', 'total_contratado', 'total_de_listings', 'utilizado_super_destaques', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes']] = RobustScaler(
    ).fit_transform(dfNumerical[['leads_form', 'total_contratado', 'total_de_listings', 'utilizado_super_destaques', 'utilizado_destaque', 'valor_mensal', 'quantidade_mes']])

# Criação do pipeline
# pipeline = Pipeline(steps=[
#     ('Drop status pagamento nulo', fnDropStatusPagamentoNa(X_train)),
#     ('Dumifica categóricas', fnGetDummies(X_train)),
#     ('Substitui numéricos nulos pela mediana da série', gnFillNumericalNaWithMedian(X_train)),
#     ('Substitui os número negativos por 0', fnReplaceNegativeWithZero(X_train)),
#     # ('Aplica o Robust Scaler', fnApplyRobustScaler(X_train)),
#     ('Gradient Boosting', GradientBoostingClassifier(learning_rate=0.2, max_depth=5, max_features='log2',
#                                                      min_samples_leaf=0.1,
#                                                      min_samples_split=0.28181818181818186,
#                                                      n_estimators=500, subsample=0.95))
# ])

fnDropStatusPagamentoNa(X_train)
fnGetDummies(X_train)
gnFillNumericalNaWithMedian(X_train)
fnReplaceNegativeWithZero(X_train)

clf= GradientBoostingClassifier(learning_rate=0.2, max_depth=5, max_features='log2',
                           min_samples_leaf=0.1,
                           min_samples_split=0.28181818181818186,
                           n_estimators=500, subsample=0.95)

# fit do modelo
clf.fit(X_train, y_train)
joblib.dump(clf, 'models/model_pipeline.pkl')

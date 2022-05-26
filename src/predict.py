# Importação de bibliotecas
import joblib
import datetime

import re
import numpy as np
import pandas as pd

from unidecode import unidecode

# Carregamento do dataset
df = pd.read_feather('data/carteira_total.feather')

# Padronização do nome das colunas
newColumnsName = []
for column_name in df:
    newColumnsName.append(
        re.sub('[^A-Za-z0-9]+', '_', unidecode(column_name).lower()))
df.columns = newColumnsName

# Criação da coluna churn, e definição do seu valor com base nos valores da coluna upsale_downsale
df["upsale_downsale"].replace(
    {"Churn": "churn", "Upsell": "upsell", "Downsell": "downsell", "Ok": "ok"}, inplace=True)
df['churn'] = df['upsale_downsale']
df['churn'].replace(
    {"ok": "0", "upsell": "0", "downsell": "0", "churn": "1"}, inplace=True)
df['churn'] = df['churn'].astype(int)

df.drop(columns=["upsale_downsale"], inplace=True)

# Adição da coluna quantidade_mes respectiva a quantidade de meses os ID SAP é cliente
df_grouped = df[['mes', 'id_sap']].groupby(['id_sap']).count().reset_index()
df_grouped.rename(columns={'mes': 'quantidade_mes'}, inplace=True)
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

    i += 1

dfQuality['status_pagamento'].replace({'4. Péssimo': 'Pessimo', '2. Regular ': 'Regular', '1. Bom': 'Bom', '3. Ruim': 'Ruim', '5. Novo': 'Novo',
                                       '2. Regular': 'Regular', '1. Bom ': 'Bom', 'lançamentos': np.nan, '5. novo': 'Novo', 0: np.nan}, inplace=True)
dfQuality.dropna(inplace=True)

regex = r'([0-9]{4})-([0-9]{2})-[0-9]{2}'


def fun_replace(data):
    return datetime.datetime.strptime(str(data.group(2)).lower(), '%m').strftime('%b').lower() + data.group(1)[-2:]


df['mes'] = df['mes'].astype(str).str.replace(regex, fun_replace, regex=True)
dfQuality.rename(columns={'ID SAP': 'id_sap', 'data': 'mes'}, inplace=True)
df = df.join(dfQuality.set_index(['id_sap', 'mes']), on=['id_sap', 'mes'])

# Tratando valores de colunas categóricas
df['oficina'].replace({'wi': 'WI'}, inplace=True)
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.lower()
df['frequencia_de_faturamento'] = df['frequencia_de_faturamento'].str.strip()
df['frequencia_de_faturamento'].replace(
    {'única vez': 'unica_vez'}, inplace=True)
df['equipe'].replace({'Relacionamento': 'RELACIONAMENTO', 'Jumbo': 'JUMBO',
                      'Resellers': 'RESELLERS', 'Regional DF': 'REGIONAL DF'}, inplace=True)
df.drop(['contratado_freemium', 'utilizado_freemium'], axis=1, inplace=True)
df = df[['pf_pj', 'leads_form', 'total_contratado', 'total_de_listings', 'equipe', 'utilizado_super_destaques',
         'utilizado_destaque', 'valor_mensal', 'quantidade_mes', 'status_pagamento', 'churn']].copy()

# Carregamento e execução do modelo
model = joblib.load("src/models/model_pipeline.pkl")
predictions = model.predict(df)
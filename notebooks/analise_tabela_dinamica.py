import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pandas.api.types import is_string_dtype
import re
import unidecode

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

def trataColunasDataFrame(df):
    newColumnsName = []
    for column_name in df:
        newColumnsName.append(re.sub('[^A-Za-z0-9]+', '_', unidecode(column_name).lower()))
    df.columns = newColumnsName

    df["upsale_downsale"].replace({"Churn": "churn", "Upsell": "upsell", "Downsell": "downsell", "Ok":"ok"}, inplace=True)
    df['churn'] = df['upsale_downsale']
    df['churn'].replace({"ok": '0', "upsell": '0', "downsell": '0', "churn": '1'}, inplace=True)

@st.cache
def load_data():
    df = pd.read_csv('../data/carteira_total.csv')
    trataColunasDataFrame(df)
    return df

st.sidebar.title('DataSets')

df = load_data()

op = st.sidebar.selectbox(
    'Selecione a opção!', (
        'Select...',
        'Teste 1'
    )
)

def getDimensoesPorQue():
    return[]

def getDimensoesQuem():
    return[]

def getDimensoesOQue():
    return []

def getDimensoesComo():
    return[]

def getDimensoesQuando():
    return []

def getDimensoesOnde():
    return[]

def getMedidas():
    return []

def formataAgrupadorUS(descricaoAgrupador):
    agrupadores = {
        'total': 'sum',
        'média': 'mean',
        'quantidade': 'count',
        'mínimo': 'min',
        'máximo': 'max'
    }
    return agrupadores[descricaoAgrupador]

def formataAgrupadorPlural(descricaoAgrupador):
    agrupadores = {
        'total': 'totais',
        'média': 'médios',
        'quantidade': 'quantidades',
        'mínimo': 'mínimos',
        'máximo': 'máximos'
    }
    return agrupadores[descricaoAgrupador]

#Faz o pivotable da tabela informada, de acordo com os eixos, funções de agregação e filtros informados
def functionPivotTable(dataframe, axis_x_vars, axis_y_vars, medidas, agg_func, filters=''):
    if filters == False:
        return False

    if filters != '':
        nan = np.nan
        df_pivot = dataframe.query(filters).copy()
    else:
        df_pivot = dataframe.copy()

    #Somente informar valores numéricos no parâmetros "values"
    df_pivot = round(df_pivot.pivot_table(
        index=axis_y_vars,
        columns=axis_x_vars,
        values=medidas,
        aggfunc=formataAgrupadorUS(agg_func),
        fill_value=0
    ),2)

    return df_pivot

#Cria um filtro para ser utilizado no pivot table
def getCompleteWhere(registrosfiltro, registrosScore, op_medida):
    where = []

    scoreClass = {'Muito Alto_min': 4.5, 'Muito Alto_max': 5, 
                  'Alto_min': 4, 'Alto_max': 4.5,
                  'Médio_min': 3, 'Médio_max': 4,
                  'Baixo_min': 1.6, 'Baixo_max': 3,
                  'Muito Baixo_min': 0, 'Muito Baixo_max': 1.6}

    if len(registrosScore) > 0:
        for score in registrosScore:
            df_RFM = criaRFMColuna(score['dimensao'], op_medida)
            df_RFM = df_RFM[(df_RFM['RFM_Score'] > scoreClass[score['class_rfm'] + '_min']) & (df_RFM['RFM_Score'] <= scoreClass[score['class_rfm'] + '_max'])]
            if is_string_dtype(df_RFM[score['dimensao']]):
                valoresColuna = []
                for valorColuna in df_RFM[score['dimensao']]:
                    valoresColuna.append('"' + valorColuna + '"')
            else:
                valoresColuna = df_RFM[score['dimensao']].astype(str)
            where.append('(' + score['dimensao'] + ' == (' + ','.join(valoresColuna) + '))')

    if len(registrosfiltro) > 0:
        for filtro in registrosfiltro:
            if filtro['operador'] in ('in', 'not in'):
                if is_string_dtype(df[filtro['nome']]):
                    listaTratada = []
                    listaNaoTratada = list(filtro['valor'].split(','))
                    for item in listaNaoTratada:
                        listaTratada.append('"' + item.strip() + '"')
                    filtro['valor'] = ','.join(listaTratada)
                filtro['valor'] = '(' + filtro['valor'].strip() + ')'
                filtro['operador'] = '==' if filtro['operador'] == 'in' else '!='
            elif is_string_dtype(df[filtro['nome']]):
                filtro['valor'] = '"' + filtro['valor'].strip() + '"'
            where.append('(' + filtro['nome'] + ' ' + filtro['operador'] + ' ' + str(filtro['valor']) + ')')
    return ' & '.join(where)

def criaRFMColuna(column_name, op_medida):
    #Análise de recência
    df_recency = df.groupby(by=column_name,
                        as_index=False)['data'].max()
    df_recency.columns = [column_name, 'data_ultima_transacao']
    recent_date = df_recency['data_ultima_transacao'].max()
    df_recency['Recency'] = df_recency['data_ultima_transacao'].apply(
        lambda x: (recent_date - x).days)
    df_recency.head()

    #Análise de Frequência
    frequency_df = df.drop_duplicates().groupby(
        by=[column_name], as_index=False)['data'].count()
    frequency_df.columns = [column_name, 'Frequency']
    frequency_df.head()

    monetary_df = df.groupby(by=column_name, as_index=False)[op_medida].sum()
    monetary_df.columns = [column_name, 'Monetary']
    monetary_df.head()

    #Análise de Valor Monetário
    rf_df = df_recency.merge(frequency_df, on=column_name)
    rfm_df = rf_df.merge(monetary_df, on=column_name).drop(
        columns='data_ultima_transacao')
    rfm_df.head()

    #Rankeamento por RFM
    rfm_df['R_rank'] = rfm_df['Recency'].rank(ascending=False)
    rfm_df['F_rank'] = rfm_df['Frequency'].rank(ascending=True)
    rfm_df['M_rank'] = rfm_df['Monetary'].rank(ascending=True)
    
    #Rankeamento de RFM normalizado
    rfm_df['R_rank_norm'] = (rfm_df['R_rank']/rfm_df['R_rank'].max())*100
    rfm_df['F_rank_norm'] = (rfm_df['F_rank']/rfm_df['F_rank'].max())*100
    rfm_df['M_rank_norm'] = (rfm_df['F_rank']/rfm_df['M_rank'].max())*100
    rfm_df.drop(columns=['R_rank', 'F_rank', 'M_rank'], inplace=True)
    
    #Definição de Score
    rfm_df['RFM_Score'] = 0.15 * rfm_df['R_rank_norm'] + 0.28 * rfm_df['F_rank_norm'] + 0.57 * rfm_df['M_rank_norm']
    rfm_df['RFM_Score'] *= 0.05
    rfm_df = rfm_df.round(2)

    return rfm_df
    
def doFindOutliers(column, tipoLimite=False):
    outliers = []
    Q1 = column.quantile(.25)
    Q3 = column.quantile(.75)
    IQR = Q3 - Q1

    algoritmoQuantilLimite = (1.5 * IQR)

    limiteInferior = Q1 - (algoritmoQuantilLimite)
    limiteSuperior = Q3 + (algoritmoQuantilLimite)

    if tipoLimite == 'Superior':
        for value in column:
            if value > limiteSuperior:
                outliers.append(value)        
    elif tipoLimite == 'Inferior':
        for value in column:
            if value < limiteInferior:
                outliers.append(value)        
    else:
        for value in column:
            if value > limiteSuperior or value < limiteInferior:
                outliers.append(value)

    return np.array(outliers)

if op == 'Contabilidade - Despesas':
    with st.container():
        st.header('Filtros')    
        col1, col2, col3 = st.columns(3)

        with col1:
            op_colunas = st.multiselect('Dimensões Coluna', df.columns)
            op_agregador = st.selectbox('Agrupadores', ['total', 'média', 'quantidade', 'mínimo', 'máximo'])

        with col2:
            op_linhas = st.multiselect('Dimensões Linha', df.columns)

        with col3:
            op_medida = st.selectbox('Medidas', getDimensoesMedida())
    if len(op_colunas) > 0 and len(op_linhas) > 0 and len(op_medida) > 0:
        with st.container():
            st.header('Segmentos')
            registrosFiltro = []
            col4, col5, col6 = st.columns(3)
            while True:
                id = len(registrosFiltro) + 1
                nome = col4.selectbox('Nome', ['Selecione...', *op_colunas, *op_linhas, *op_medida], key=f'nome_{id}')
                operador = col5.selectbox('Operador', ['Selecione...', '==', '!=', '>', '<', 'in', 'not in'], key=f'operador_{id}')
                valor = col6.text_input('Valor', key=f'valor_{id}')
                
                if nome == 'Selecione...' or operador == 'Selecione...' or valor == '':
                    break
                registrosFiltro.append({'nome': nome, 'operador': operador, 'valor': valor})
        with st.container():
            st.header('Clusterização (Score RFM)')
            registrosScore = []
            col7, col8 = st.columns(2)
            while True:
                id = len(registrosScore) + 1
                dimensao = col7.selectbox('Dimensão', ['Selecione...', *op_colunas, *op_linhas], key=f'dimensao_{id}')
                class_rfm = col8.selectbox('Categoria', ['Selecione...', 'Muito Alto', 'Alto', 'Médio', 'Baixo', 'Muito Baixo'], key=f'class_rfm_{id}')
                
                if dimensao == 'Selecione...' or class_rfm == 'Selecione...':
                    break
                registrosScore.append({'dimensao': dimensao, 'class_rfm': class_rfm})
        with st.container():
            st.title('Análise')

            completeWhere = getCompleteWhere(registrosFiltro, registrosScore, op_medida)

            if len(op_medida) == 1:
                op_medida = op_medida[0]

            if op_agregador:
                df_pivot = functionPivotTable(df, axis_x_vars=op_colunas, axis_y_vars=op_linhas, medidas=op_medida, filters=completeWhere, agg_func=op_agregador)

            if df_pivot.empty:
                st.warning('Nenhum registro encontrado para o filtro informado')
            else:
                if isinstance(df_pivot.columns, pd.MultiIndex):
                    df_pivot.columns = [' - '.join(col) for col in df_pivot.columns.values]
                if isinstance(df_pivot.index, pd.MultiIndex):
                    df_pivot.index = [''.join(str(idx)) for idx in df_pivot.index.values]
                
                st.dataframe(df_pivot)
                if any(op_linha in getDimensoesQuando() for op_linha in op_linhas) & (len(df_pivot.index) > 1):
                    fig = px.line(df_pivot, x=df_pivot.index, y=df_pivot.columns, labels={'index': '(' + ','.join(op_linhas) + ')', 'value': 'valor (' + op_agregador + ')', 'variable': ' - '.join(op_colunas)})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.bar(df_pivot, x=df_pivot.index, y=df_pivot.columns, labels={'index': '(' + ','.join(op_linhas) + ')', 'value': 'valor (' + op_agregador + ')', 'variable': ' - '.join(op_colunas)}, barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

                column10, column11 = st.columns(2)

                with column10:
                    for y in df_pivot.index:
                        df_sorted = df_pivot.loc[y].sort_values(ascending=False).to_frame().dropna()
                        df_sorted = df_sorted[df_sorted[y] != 0].copy()

                        outliersSuperior = doFindOutliers(df_sorted[y], 'Superior')

                        if len(outliersSuperior) > 0:
                            if op_agregador == 'quantidade':
                                st.warning('Os registros da dimensão "' + ' - '.join(op_colunas) + '" que geraram despesas com  ' + formataAgrupadorPlural(op_agregador)  + ' discrepantes (muito acima da média) para a categoria "' + str(y) + 
                                           '" da dimensão "' + ' - '.join(op_linhas) + '" (de acordo com os segmentos/clusters utilizados), somam o total de ' + str(round(sum(outliersSuperior), 2)) + 
                                           ' (equivalente aos ' + str(round(len(outliersSuperior) * 100 / len(df_sorted[y]), 2)) + 
                                        '% das fontes de despesa com valores movimentados). A soma das ' + formataAgrupadorPlural(op_agregador)  + ' de outros registros resulta em ' + str(round(sum(df_sorted[df_sorted[y] < min(outliersSuperior)][y]), 2)))
                            else:
                                st.warning('Os registros da dimensão "' + ' - '.join(op_colunas) + '" que geraram despesas com valores ' + formataAgrupadorPlural(op_agregador)  + ' discrepantes (muito acima da média) para a categoria "' + str(y) + 
                                           '" da dimensão "' + ' - '.join(op_linhas) + '" (de acordo com os segmentos/clusters utilizados), somam um valor total de R$' + str(round(sum(outliersSuperior), 2)) + 
                                           ' (equivalente aos ' + str(round(len(outliersSuperior) * 100 / len(df_sorted[y]), 2)) + 
                                           '% das fontes de despesa com valores movimentados). A soma dos valores ' + formataAgrupadorPlural(op_agregador)  + ' de outros registros resulta em R$' + str(round(sum(df_sorted[df_sorted[y] < min(outliersSuperior)][y]), 2)))
                            if st.button('Visualizar Relação', key='dataframe_maior_' + str(y)):
                                st.dataframe(df_sorted[df_sorted[y] >= min(outliersSuperior)])
                        else:
                            st.warning('Não são existentes fontes de despesa que geraram valores altos (outliers) para a categoria "' + str(y) + '" da dimensão "' + ' - '.join(op_linhas) + '"')
# Análise de Predição de Churn da Empresa Imovel Web
Predição de churn de uma empresa de venda e aluguel de imóveis

Organização das Pastas
----
- analysis: Todo e qualquer documento utilizado nas análises (.CSV, .XLSX, JSON, ...) Toda a qualquer análise (sendo arquivo .ipynb ou .py) 
- data: Todo e qualquer documento utilizado nas análises ou testes (.CSV, .XLSX, JSON, ...)
- mlruns: Meta dados para versionamento do modelo (mlflow)
- src: Arquivos de treino do modelo e predição de dados

Passos para a execução da predição:
----
1. Executar o comando `python src/train.py`
    - O modelo será treinando e armazenado em uma nova versão, na pasta mlruns

2. Executar o comando `flask run`
    - Para subir o servidor local do flask

3. Em alguma ferramenta de simulação de requisição http (Thunder, Postman, ...)
    - Query String: http://127.0.0.1:5000/predict
    - Método: Post
    - Body: Informar o conteúdo do arquivo data/churn_predict.json
    - Executar e verificar o retorno na seguinte estrutura:
            [
            {
                "id_sap": 1,
                "churn": "não"
            },
            {
                "id_sap": 1,
                "churn": "não"
            }
            ]

Painel de acesso do versionamento do modelo:
----

1. Executar o comando `mlflow ui`
2. Acessar no navegador o link http://127.0.0.1:5000/

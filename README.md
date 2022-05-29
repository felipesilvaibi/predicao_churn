# Análise de Predição de Churn
Predição de churn da empresa de venda e aluguel de imóveis "Imovel Web"

Estrutura das Pastas
----
```
.
├── analysis               # Arquivos de análises (.CSV, .XLSX, JSON, ...) Toda a qualquer análise (sendo arquivo .ipynb ou .py) 
├── data                   # Arquivos utilizados nas análises (.CSV, .XLSX, JSON, ...)
├── mlruns                 # Meta dados de versionamento do modelo (mlflow)
├── docs                   # Arquivos de documentação da API
├── src                    # Arquivos da aplicação
│   ├── controllers        # Arquivos de treino do modelo e predição de dados
│   ├── models             # Arquivos de modelo para requests e responses da API
│   ├── routes             # Rota para acesso ao arquivo de predição
│   └── server             # Servidor Flask
├── .env 
├── .gitignore
├── main.py
└── README.md

```    

Comandos:
----
- Treinamento do modelo: `python src/controllers/train.py`

- Iniciar o servidor Flask: `flask run`
    - Rotas da API:
        - Documentação Swagger da API: `/docs`
        - Predição de churn de clientes informados: `/predict`

- Iniciar o servidor do ML Flow: `mlflow ui`

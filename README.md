# Análise de Predição de Churn
Predição de churn da empresa de venda e aluguel de imóveis

Observação: Somente realizado o upload dos arquivos de data para fins didáticos (ambos não possuem informações que infringem a lgpd, não necessitando
a anomizização dos dados
) 

Estrutura das Pastas
----
```
.
├── analysis                            # Arquivos de análises (.CSV, .XLSX, JSON, ...) Toda a qualquer análise (sendo arquivo .ipynb ou .py) 
├── data                                # Arquivos utilizados nas análises (.CSV, .XLSX, JSON, ...)
├── mlruns                              # Meta dados de versionamento do modelo (mlflow)
├── src                                 # Arquivos da aplicação
│   └── flask                           # Servidor Flask
│       └── app                         # Aplicação 
│           ├── controllers             # Arquivos de treino do modelo e predição de dados
│           ├── models                  # Modelo para a request e response da API
│           ├── routes                  # Rota para acesso ao arquivo de predição
│           └── requirements.txt        # Dependências do Projeto
│       ├── env                         # Ambiente Virtual (somente disponível no repositório de desenvolvimento, e no docker)
│       ├── .dockerignore               # Arquivos ignorados pelo docker
│       ├── app.ini                     # Configurações da aplicação
│       ├── Dockerfile                  # Arquivo de configurações docker
│       └── run.py                      # Starter da aplicação
│   └── ngix                            # Servidor Nginx
│       ├── Dockerfile                  # Arquivo de configurações docker 
│       └── app.ini                     # Configurações do servidor
├── .env.example
├── .gitignore
├── .docker-compose.yml                 # Arquivo de execução dos containers docker em parapelo
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

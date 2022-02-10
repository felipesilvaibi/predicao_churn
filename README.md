# Análise de Predição de Churn da Empresa Imovel Web
Predição de churn de uma empresa de venda e aluguel de imóveis

Organização das Pastas
----
1. data: Todo e qualquer documento utilizado nas análises (.CSV, .XLSX, JSON, ...)
2. docs: Todo e qualquer documento do projeto (protótipos, assigments, manuais, artigos, ...)
3. notebooks: Toda a qualquer análise (sendo arquivo .ipynb ou .py) 

Observação: Não alterar os demais arquivos ou os arquivos de configuração de ambiente 

Configuração de Ambiente
----
Observação: Ao seguir os passos abaixo, substituir todas as informações pessoais (nome e email "felipe") por informações pessoas próprias (o email
deverá ser o mesmo utilizado no github) 

1. Baixar o virtualbox, criar uma nova máquina virtual ubuntu e vincular a imagem (sistema operacional) mais recente 
do ubuntu a máquina criada (https://www.youtube.com/watch?v=7FCYFy0J4NQ&ab_channel=AGclubedainform%C3%A1tica)

2. Realizar o download do VSCode
Observação: Instalar as extensões "Python intelissence", "Jupyter Notebook Renderes", "Remote - SSH", "Material Icon Theme"

3. Seguir os passos do documento "Configurao_do_Servidor_na_VM_Ubuntu.pdf"
- Observação: Utilizar o terminal do VSCode (Terminal / New Terminal)

4. (No diretório da máquina virtual "Ubuntu" (através do VSCode aberto no windows) seguir os passos do arquivo "Criao_de_Ambiente_Virtual_no_VSCode.pdf"

5. Seguir os passos do arquivo "Integrando_com_o_Git_Hub.pdf"

6. Após importar o repositório do github, mover a pasta "env" para dentro do repositório clonado

7. Acessar o repositório, verificar se o terminal está conectado a máquina virtual e executar o seguinte comando "pip install -r requirements.txt"

8. Quando todos os pacotes utilizados forem instalados, ja é possível trabalhar com os arquivos da pasta notebooks

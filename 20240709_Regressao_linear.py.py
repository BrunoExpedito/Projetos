#!/usr/bin/env python
# coding: utf-8

# # Explorando a Demanda de Passageiros Aéreos com Regressão Linear Múltipla!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configurar a semente aleatória para reprodutibilidade
np.random.seed(42)

# Definir os destinos
destinos = ['São Paulo', 'Rio de Janeiro']

# Criar uma lista para armazenar os dados
data = []

# Gerar dados fictícios para São Paulo e Rio de Janeiro
for destino in destinos:
    for _ in range(100):  # Criar 100 observações para cada destino
        data.append([
            destino,
            pd.Timestamp('2020-01-01') + pd.DateOffset(months=np.random.randint(0, 36)),  # Data aleatória nos próximos 3 anos
            np.random.randint(100, 1000),  # Número de passageiros entre 100 e 1000
            np.random.uniform(50, 500),  # Preço das passagens entre 50 e 500
            np.random.uniform(1000, 2000),  # PIB entre 1000 e 2000
            np.random.uniform(1, 10),  # Inflação entre 1 e 10
            np.random.randint(0, 2),  # Indicador de feriado (0 ou 1)
            np.random.randint(5, 50)  # Frequência dos voos entre 5 e 50
        ])

# Converter a lista de dados para um DataFrame do pandas
dados = pd.DataFrame(data, columns=[
    'Destino', 'Data', 'Passageiros', 'PrecoPassagem', 'PIB', 'Inflacao', 'Feriado', 'FrequenciaVoos'
])

# Visualizar as primeiras linhas do DataFrame
print(dados.head())

# Função para criar, treinar e avaliar o modelo para um destino específico
def avaliar_modelo_para_destino(destino, dados):
    # Filtrar os dados para o destino específico
    dados_destino = dados[dados['Destino'] == destino]
    
    # Definir as variáveis independentes (X) e a variável dependente (y)
    X = dados_destino[['PrecoPassagem', 'PIB', 'Inflacao', 'Feriado', 'FrequenciaVoos']]
    y = dados_destino['Passageiros']
    
    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Criar o modelo de regressão linear
    modelo = LinearRegression()
    
    # Treinar o modelo com os dados de treino
    modelo.fit(X_train, y_train)
    
    # Fazer previsões com os dados de teste
    y_pred = modelo.predict(X_test)
    
    # Calcular o erro quadrático médio (MSE) e o coeficiente de determinação (R²)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Exibir os resultados
    print(f'Modelo para {destino}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    print(f'Coeficientes:')
    
    # Criar um DataFrame com os coeficientes do modelo
    coeficientes = pd.DataFrame(modelo.coef_, X.columns, columns=['Coeficiente'])
    print(coeficientes)
    print('-'*30)

    # Plotar os resultados reais vs. preditos
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title(f'Valores Reais vs. Valores Preditos para {destino}')
    plt.show()

# Avaliar o modelo para São Paulo e Rio de Janeiro
for destino in destinos:
    avaliar_modelo_para_destino(destino, dados)


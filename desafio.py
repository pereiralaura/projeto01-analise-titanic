# Desafio: que tipos de pessoas têm maior probabilidade de sobreviver? Use os dados dos passageiros do Titanic (nome, idade, preço da passagem, etc) para tentar prever quem sobreviverá e quem morrerá.

# Bibliotecas & importações
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Criar o caminho do diretório
dir_raiz = os.getcwd()
dir_train_csv = os.path.join(dir_raiz, 'data', 'train.csv')
dir_test_csv = os.path.join(dir_raiz, 'data', 'test.csv')

# Carregar o conjunto de dados
data_train = pd.read_csv(dir_train_csv) #conjunto de dados treinamento
data_test = pd.read_csv(dir_test_csv) #conjunto de dados teste

# Criar random forest model
# Identificar características e alvo nos dados de treinamento
y = data_train["Survived"]

# Codificar colunas categóricas usando one-hot encoding
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(data_train[features])
X_test = pd.get_dummies(data_test[features])

# Treinar modelo de random forest
# Criar o modelo de Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# Treinar o modelo com os dados de treinamento
rf_model.fit(X, y)

# Fazer Previsões nos Dados de Teste
# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Adicionar as previsões aos dados de teste e salvar em um novo arquivo
data_test = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': y_pred})
data_test.to_csv('test_data_with_predictions.csv', index=False)
print("Your submission was successfully saved!")

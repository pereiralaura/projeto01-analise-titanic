# Desafio: que tipos de pessoas têm maior probabilidade de sobreviver? Use os dados dos passageiros do Titanic (nome, idade, preço da passagem, etc) para tentar prever quem sobreviverá e quem morrerá.

#bibliotecas & importações
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os #import data on the file
from sklearn.ensemble import RandomForestClassifier

diretorio_raiz = os.getcwd()
dir_train_csv = os.path.join(diretorio_raiz, 'data', 'train.csv')
dir_test_csv = os.path.join(diretorio_raiz, 'data', 'test.csv')

# Lendo o csv e salvando dentro de uma variável (conjunto de treinamento/verdade básica)
train_data = pd.read_csv(dir_train_csv)

#Lendo o csv e salvando dentro de uma variável (conjunto de treinamento/verdade básica)
test_data = pd.read_csv(dir_test_csv)

# #teste para verificar uma hipótese de um exemplo
# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)

# men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)

#Testando random forest model

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
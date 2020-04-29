import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random
from sklearn import svm
from imblearn.under_sampling import TomekLinks

# Diccionario para codificar los nombres de las clases
categorical_encoder_binary = {'RESIDENTIAL': 0,
    'INDUSTRIAL': 1,
    'PUBLIC': 1,
    'OFFICE': 1,
    'OTHER': 1,
    'RETAIL': 1,
    'AGRICULTURE': 1
}

categorical_encoder_class = {'RESIDENTIAL': 0,
    'INDUSTRIAL': 1,
    'PUBLIC': 2,
    'OFFICE': 3,
    'OTHER': 4,
    'RETAIL': 5,
    'AGRICULTURE': 6
}

# Diccionario para codificar la variable categorica CADASTRALQUALITYID a un vector one-hot
categorical_encoder_catastral = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    'B': [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    'C': [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '1': [0, 0, 0, 1, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '2': [0, 0, 0, 0, 1, 0, 0, 0, 0 ,0 ,0 ,0 ,0],
    '3': [0, 0, 0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0 ,0],
    '4': [0, 0, 0, 0, 0, 0, 1, 0, 0 ,0 ,0 ,0 ,0],
    '5': [0, 0, 0, 0, 0, 0, 0, 1, 0 ,0 ,0 ,0 ,0],
    '6': [0, 0, 0, 0, 0, 0, 0, 0, 1 ,0 ,0 ,0 ,0],
    '7': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1 ,0 ,0 ,0],
    '8': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,1 ,0 ,0],
    '9': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,1 ,0],
    '""': [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,1]
}

# Variable que contendrá las muestras
data = []
true_data = []

with open(r'Data\Modelar_UH2020.txt') as read_file:
    # La primera linea del documento es el nombre de las variables, no nos interesa
    read_file.readline()
    # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
    for line in read_file.readlines():
        # Eliminamos el salto de línea final
        line = line.replace('\n', '')
        # Separamos por el elemento delimitador
        line = line.split('|')
        # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
        line[52] = 2020 - int(line[52])
        if line[53] is '':
            line[53] = 0
        # Codificamos CADASTRALQUALITYID y arreglamos la muestra
        data.append(line[1:54] + categorical_encoder_catastral[line[54]] + [categorical_encoder_binary[line[55]]])
        true_data.append(line[1:54] + categorical_encoder_catastral[line[54]] + [categorical_encoder_class[line[55]]])


# Finalmente convertimos las muestras preprocesadas a una matriz
data = np.array(data).astype('float32')
true_data = np.array(true_data).astype('float32')


# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.2


# X -> Datos ya tratados sin predicción
# Y -> Predicción
last_position = len(data[0]) - 1
X, Y = (data[:, :last_position], data[:, last_position])

model = xgb.XGBClassifier(
    objective = 'binary:hinge',
)

sss = StratifiedShuffleSplit(
    n_splits = 1,       # Solo una partición
    test_size = 0.2,    # Repartición 80/20 
)

# La función es un iterador (debemos iterar, aunque sea solo una vez)
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

data_bis = []
data_bis_pred = model.predict(X)

for i in range(len(data)):
    if data_bis_pred[i] == 1.0:
        data_bis.append(true_data[i])

data_bis = np.array(data_bis).astype('float32')


# Segundo clasificador

last_position = len(data_bis[0]) - 1
X, Y = (data_bis[:, :last_position], data_bis[:, last_position])

model = xgb.XGBClassifier(
    objective = 'multi:softmax',
    num_class = 7,
)

sss = StratifiedShuffleSplit(
    n_splits = 1,       # Solo una partición
    test_size = 0.2,    # Repartición 80/20 
)

# La función es un iterador (debemos iterar, aunque sea solo una vez)
for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
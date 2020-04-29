import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import random

### INICIO PARTE PRETRATAMIENTO DE DATOS ###
# Diccionario para codificar los nombres de las clases
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
        line[55] = categorical_encoder_class[line[55]]
        # Codificamos CADASTRALQUALITYID y arreglamos la muestra
        data.append(line[1:54] + categorical_encoder_catastral[line[54]] + [line[55]])


# Finalmente convertimos las muestras preprocesadas a una matriz
data = np.array(data).astype('float32')

# Variable que contendrá las muestras a predecir
data_predict = []

# Mismo procesamiento de datos que para el conjunto inicial
with open(r'Data\Estimar_UH2020.txt') as read_file:
    # La primera linea del documento es el nombre de las variables, no nos interesa
    read_file.readline()
    # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valos catastral)
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
        data_predict.append(line[:54] + categorical_encoder_catastral[line[54]])

# Finalmente convertimos las muestras preprocesadas a una matriz (no númerica, nos interesa el id esta vez)
data_predict = np.array(data_predict)
### FIN PARTE PRETRATAMIENTO DE DATOS ###

# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])
# Añadimos a la lista de cada clase las muestras de esta
for sample in data:
    data_per_class[int(sample[len(sample) - 1])].append(sample)


# Variable que contendrá los datos procesados
data_proc = []


# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations =  20

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = False

# Si True, guarda los mejores modelos para poder replicar el clasificador
persistent_mode = False

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.20

# Variable en el rango (0.0 - 1.0) que indica el procentaje de mejores modelos a utilizar
best_model_avg = 0.5

# Variables que miden las métricas globlales del ENTRENAMIENTO
accuracy_avg = 0
precision_avg = 0
recall_avg = 0
f1_avg = 0

# Lista que contendra diccionarios con las metricas de cada modelo, predicciones y conjunto de datos utilizados
concensus = []

# Los diccionarios anteriores seguiran el siguiente formato:
'''
model_info = {
    "accuracy":
    "precision":
    "recall":
    "f1":
    "predictions":
    "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    "model": <- Solo si 'persistent_mode' es True
}
'''

for ite in range(iterations):
    data_proc = []
    # Muestras de la clase RESIDENTIAL
    random.shuffle(data_per_class[0])
    data_proc += data_per_class[0][:6000]

    # Muestras de las otras clases
    for i in range(6):
        data_proc += data_per_class[i + 1]
        
    # Volvemos a convertir los datos una vez procesados a una matriz
    data_proc = np.array(data_proc)

    # Obtenemos una separación del conjunto de train y test equilibrado (respecto a porcentaje de cada clase)
    pos = len(data_proc[0]) - 1
    X, Y = (data_proc[:, :pos], data_proc[:, pos])

    sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_avg)

    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]


    # Mostramos el porcentaje de entrenamiento
    print('Entrenamiento completo al {}%'.format(ite/iterations * 100))
    
    # Modelo XGB
    model = xgb.XGBClassifier(
        # General parameters
        # Tree Booster parameters
        eta = 0.15,
        max_depth = 10,
        n_estimators = 240,
        tree_method = 'exact',
        # Learning task parameters
        objective = 'multi:softmax',
        num_class =  7,
        eval_metric = 'merror',
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas del modelo entrenado
    if debug_mode:
        print('XGBoost ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}\n'.format(f1_score(y_test, y_pred, average = 'macro')))
    
    # Actualización de las métricas de ENTRENAMIENTO
    accuracy_avg += accuracy_score(y_test, y_pred)
    precision_avg += precision_score(y_test, y_pred, average = 'macro')
    recall_avg += recall_score(y_test, y_pred, average = 'macro')
    f1_avg += f1_score(y_test, y_pred, average = 'macro')

    # Diccionario con la información del modelo
    model_info = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average = 'macro'),
        "recall": recall_score(y_test, y_pred, average = 'macro'),
        "f1": f1_score(y_test, y_pred, average = 'macro'),
        "predictions": model.predict(data_predict[:, 1:].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
    if persistent_mode:
        model_info['model'] = model
    concensus.append(model_info)

        
    # Modelo RandomForest
    model = RandomForestClassifier(
        criterion = 'entropy',
        n_jobs = -1,
        max_features = None,
        n_estimators = 400,
        max_depth = 50,
        min_samples_split = 3,
        min_samples_leaf = 1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas del modelo entrenado
    if debug_mode:
        print('RF ({})'.format(ite))
        print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
        print('Precision (macro): {}'.format(precision_score(y_test, y_pred, average = 'macro')))
        print('Recall (macro): {}'.format(recall_score(y_test, y_pred, average = 'macro')))
        print('F1 (macro): {}\n'.format(f1_score(y_test, y_pred, average = 'macro')))
    
    # Actualización de las métricas de ENTRENAMIENTO
    accuracy_avg += accuracy_score(y_test, y_pred)
    precision_avg += precision_score(y_test, y_pred, average = 'macro')
    recall_avg += recall_score(y_test, y_pred, average = 'macro')
    f1_avg += f1_score(y_test, y_pred, average = 'macro')

    # Diccionario con la información del modelo
    model_info = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average = 'macro'),
        "recall": recall_score(y_test, y_pred, average = 'macro'),
        "f1": f1_score(y_test, y_pred, average = 'macro'),
        "predictions": model.predict(data_predict[:, 1:].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
    if persistent_mode:
        model_info['model'] = model
    concensus.append(model_info)


print('\nEntrenamiento completo\n')
print('MÉTRICAS DEL ENTRENAMIENTO (global)')
print('Accuracy: {}'.format(accuracy_avg / (iterations * 2)))
print('Precision (macro): {}'.format(precision_avg / (iterations * 2)))
print('Recall (macro): {}'.format(recall_avg / (iterations * 2)))
print('F1(macro): {}'.format(f1_avg / (iterations * 2)))

# 1. Ordenamos 'concensous' según una métrica
concensus = sorted(concensus, key = lambda i: i['f1'], reverse = True)

# 2. Obtenemos los 'x' mejores modelos
n = int(iterations * 2 * best_model_avg)

# 3. Calculamos la métrica general para los 'x' modelos y predecimos
accuracy_avg = 0
precision_avg = 0
recall_avg = 0
f1_avg = 0

for i in range(n):
    # Métricas
    accuracy_avg += concensus[i]['accuracy']
    precision_avg += concensus[i]['precision']
    recall_avg += concensus[i]['recall']
    f1_avg += concensus[i]['f1']

    # Predicciones
    predictions_aux = concensus[i]['predictions']
    for i in range(len(data_predict)):
        if (data_predict[i, 0] not in predictions):
            predictions[data_predict[i, 0]] = [int(predictions_aux[i])]
        else:
            predictions[data_predict[i, 0]].append(int(predictions_aux[i]))

print('\nMÉTRICAS DEL MODELO (concenso)')
print('Accuracy: {}'.format(accuracy_avg / n))
print('Precision (macro): {}'.format(precision_avg / n))
print('Recall (macro): {}'.format(recall_avg / n))
print('F1 (macro): {}'.format(f1_avg / n))

# 4. Guardamos la partición que mejor resultado ha dado
# TODO: concensus[0]['data'] -> fichero csv
np.savetxt(r'Resultados/Res_FINAL_(X_train).csv', concensus[0]['data']['X_train'], delimiter = '|')
np.savetxt(r'Resultados/Res_FINAL_(X_test).csv', concensus[0]['data']['X_test'], delimiter = '|')
np.savetxt(r'Resultados/Res_FINAL_(y_train).csv', concensus[0]['data']['y_train'], delimiter = '|')
np.savetxt(r'Resultados/Res_FINAL_(y_test).csv', concensus[0]['data']['y_test'], delimiter = '|')


# Diccionario para decodificar el nombre de las clases
categorical_decoder_class = {0: 'RESIDENTIAL',
    1: 'INDUSTRIAL',
    2: 'PUBLIC',
    3: 'OFFICE',
    4: 'OTHER',
    5: 'RETAIL',
    6: 'AGRICULTURE'}

def most_frequent(lst): 
    return max(set(lst), key = lst.count) 


with open(r'Resultados/Res_FINAL', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in data_predict:
        write_file.write('{}|{}\n'.format(sample[0], categorical_decoder_class[most_frequent(predictions[sample[0]])]))


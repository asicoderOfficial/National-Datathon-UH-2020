# 1. Imports
# Librerías externas
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import random

# Métodos realizados por Astralaria, en Astralarialib
from datasets_get import get_modelar_data, getX, getY, get_mod_data_original, get_estimar_data, get_modelar_data_ids, get_categorical_decoder_class, get_estimar_ids
from feature_engineering import coordinates_fe
from random_undersampling import random_undersample_residential
from visualization import pca_general, tsne, hist_decomposition


# 2. Obtención de los datasets
# Sin ID, para las visualizaciones
# Dataset modelar sin IDs, sin preprocesamiento
mod_or = get_mod_data_original()
# Dataset modelar original sin IDs y sin la variable clase
X_mod_or = getX(mod_or)
# Dataset modelar original sin IDs y solo con la variable clase
Y_mod_or = getY(mod_or)

# Con ID, para el FE
# Dataset modelar con IDs, preprocesado y con Random Undersampling de RESIDENTIAL
modelar_df = random_undersample_residential(get_modelar_data_ids())
# Dataset estimar, preprocesado
X_estimar = get_estimar_data()
# Dataset modelar sin la variable clase
X_modelar = getX(modelar_df)
# Dataset modelar, solo la variable clase
Y_modelar = getY(modelar_df)


# 3. Visualizaciones
# Histograma descompensación
#hist_decomposition()

# PCA 2D y PCA 3D
#pca_general(X_mod_or, Y_mod_or)

# t-SNE
#tsne(X_mod_or, Y_mod_or, [50], 2)


# 4. Feature engineering
# Obtenemos los nuevos datasets con las 7 features de los K-1 vecinos más próximos
X_modelar, X_estimar, est_IDS = coordinates_fe(X_modelar, Y_modelar, X_estimar)
# Ajustamos para numpy arrays
Y_modelar = Y_modelar.values

Y_modelar = Y_modelar[1:, :]
X_modelar = X_modelar[1:, :]


# 5. Modelo final Random Forest + XGBClassifier
# Variable que contendrá las muestras separadas por clase
data_per_class = []

# Añadimos una lista vacía por clase
for _ in range(7):         
    data_per_class.append([])

# Añadimos a la lista de cada clase las muestras de esta
for i in range(len(X_modelar)):
    data_per_class[int(Y_modelar[i])].append(X_modelar[i, 1:].tolist() + Y_modelar[i].tolist())


# Variable que contendrá los datos procesados
data_proc = []


# Variable que contendra las predicciones globales de cada muestra
predictions = {}

# Número de iteraciones total por módelo
iterations = 100

# Si True, muestra información de cada modelo local tras entrenarlo
debug_mode = True

# Variable en el rango (0.0 - 1.0) que indica el procentaje de muestras de validación
test_avg = 0.20

# Variable en el rango (0.0 - 1.0) que indica el procentaje de mejores modelos a utilizar
best_model_avg = 0.4

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


    pos = len(X_estimar[0]) - 1
    
    # Mostramos el porcentaje de entrenamiento
    print('Entrenamiento completo al {}%'.format(ite/iterations * 100))
    
    # Modelo XGB
    model = xgb.XGBClassifier(
        eta = 0.15,
        max_depth = 10,
        n_estimators = 240,
        tree_method = 'exact',
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
        "predictions": model.predict(X_estimar[:, :pos].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
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
        "predictions": model.predict(X_estimar[:, :pos].astype('float32')),
        "data": {
            'X_train': X_train, 
            'X_test': X_test, 
            'y_train': y_train, 
            'y_test': y_test,
        },
    }
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
    pos = len(X_estimar[0]) - 1
    predictions_aux = model.predict(X_estimar[:, :pos].astype('float32'))
    ids = get_estimar_ids()
    for i in range(len(ids)):
        if (ids[i] not in predictions):
            predictions[ids[i]] = [int(predictions_aux[i])]
        else:
            predictions[ids[i]].append(int(predictions_aux[i])) 

print('\nMÉTRICAS DEL MODELO (concenso)')
print('Accuracy: {}'.format(accuracy_avg / n))
print('Precision (macro): {}'.format(precision_avg / n))
print('Recall (macro): {}'.format(recall_avg / n))
print('F1 (macro): {}'.format(f1_avg / n))


# 6. Predicción final
# Diccionario para decodificar el nombre de las clases
categorical_decoder_class = get_categorical_decoder_class()

#Método que calcula la moda.
def most_frequent(lst): 
    return max(set(lst), key = lst.count) 

with open(r'UPV_Astralaria.txt', 'w') as write_file:
    write_file.write('ID|CLASE\n')
    for sample in get_estimar_ids():
        write_file.write('{}|{}\n'.format(sample, categorical_decoder_class[most_frequent(predictions[sample])]))
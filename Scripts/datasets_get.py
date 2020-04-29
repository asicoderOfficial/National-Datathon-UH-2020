 # coding=utf-8
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import random

scriptpath = os.path.dirname(__file__)

CLASS = 'CLASS'

#Lista de las categorías
categories_list = ['RESIDENTIAL', 'INDUSTRIAL', 'PUBLIC', 'OFFICE', 'OTHER', 'RETAIL', 'AGRICULTURE']

# Diccionario para codificar los nombres de las clases
categorical_encoder_class = {'RESIDENTIAL': 0,
    'INDUSTRIAL': 1,
    'PUBLIC': 2,
    'OFFICE': 3,
    'OTHER': 4,
    'RETAIL': 5,
    'AGRICULTURE': 6
}

# Diccionario para codificar las variables no numéricas
categorical_encoder_catastral = {'A': -10,
    'B': -20,
    'C': -30,
    '""': 50
}

# Diccionario para decodificar el nombre de las clases
categorical_decoder_class = {0: 'RESIDENTIAL',
    1: 'INDUSTRIAL',
    2: 'PUBLIC',
    3: 'OFFICE',
    4: 'OTHER',
    5: 'RETAIL',
    6: 'AGRICULTURE'}

# Diccionario para codificar la variable categorica CADASTRALQUALITYID a un vector one-hot
categorical_encoder_catastral_onehot = {
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

def get_categorical_encoder_class():
    return categorical_encoder_class

def get_categorical_encoder_catastral():
    return categorical_encoder_catastral

def get_categorical_decoder_class():
    return categorical_decoder_class

def get_categories_list():
    return categories_list

def get_categorical_encoder_catastral_onehot():
    return categorical_encoder_catastral_onehot


"""
Método que devuelve el dataset modelar preprocesado,
como se menciona en el apartado 4.1 de Astralaria.pdf.
""" 
def get_modelar_data(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras
    data_list = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Modelar_UH2020.txt') as read_file:
    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        read_file.readline()
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
            line[52] = 2020 - int(line[52])
            if line[53] is '':
                line[53] = missing_value
            line[55] = categorical_encoder_class[line[55]]
            # Codificamos CADASTRALQUALITYID y arreglamos la muestra
            if one_hot:
                data_list.append(line[1:54] + categorical_encoder_catastral_onehot[line[54]] + [line[55]])
            else:
                if line[54] in categorical_encoder_catastral:
                    line[54] = categorical_encoder_catastral[line[54]]
                data_list.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_list = np.array(data_list).astype('float32')
    # Convertimos dicha matriz a un dataframe de pandas
    #df = pd.DataFrame(data = data_list).rename(columns={np.int64(66):'CLASS'})
    df = pd.DataFrame(data=data_list).rename(columns={66:'CLASS'})
    return df


"""
Método que devuelve el dataset modelar sin preprocesar.
"""
def get_mod_data_original():
    # Variable que contendrá las muestras
    data = []

    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        read_file.readline()
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            if line[54] in categorical_encoder_catastral:
                line[54] = categorical_encoder_catastral[line[54]]
                if line[54] is 50:
                    line[53] = -1
            line[55] = categorical_encoder_class[line[55]]
            # No nos interesa el identificador de la muestra, lo descartamos
            data.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data = np.array(data).astype('float32')
    df = pd.DataFrame(data=data).rename(columns={54:'CLASS'})
    return df


"""
Método que devuelve el dataset modelar con los IDs,
y preprocesado como se menciona en el apartado 4.1 de Astralaria.pdf.
"""
def get_modelar_data_ids(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras
    data_list = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Modelar_UH2020.txt') as read_file:
    with open(r'Data/Modelar_UH2020.txt') as read_file:
        # La primera linea del documento es el nombre de las variables, no nos interesa
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valor catastral y la clase)
        read_file.readline()
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
            #line[52] = 2020 - int(line[52])
            if line[53] is '':
                line[53] = missing_value
            line[55] = categorical_encoder_class[line[55]]
            # Codificamos CADASTRALQUALITYID y arreglamos la muestra
            if one_hot:
                data_list.append(line[:54] + categorical_encoder_catastral_onehot[line[54]] + [line[55]])
            else:
                if line[54] in categorical_encoder_catastral:
                    line[54] = categorical_encoder_catastral[line[54]]
                data_list.append(line[1:])

    # Finalmente convertimos las muestras preprocesadas a una matriz
    a = np.array(data_list)
    ids = a[:, 0]
    a = np.delete(a, 0, axis=1)
    a = a.astype('float32')
    dfids = pd.DataFrame(data=ids)
    dfa = pd.DataFrame(data=a).rename(columns={66:'CLASS'})
    dfa[0] = dfids
    return dfa

"""
Método que devuelve los IDs del conjunto estimar.
"""
def get_estimar_ids():
    res = []
    with open(r'Data/Estimar_UH2020.txt') as read_file:
        read_file.readline()
        for sample in read_file.readlines():
            sample = sample.split('|')
            res.append(sample[0])

    return res


"""
Método que devuelve el dataset sin la variable clase.
"""
def getX(modelar_df):
    return modelar_df.loc[:, modelar_df.columns!=CLASS]


"""
Método que devuelve el dataset que solo contiene la
variable clase.
"""
def getY(modelar_df):
    return modelar_df.loc[:, modelar_df.columns == CLASS]


"""
Método que devuelve el dataset estimar preprocesado como se
menciona en el apartado 4.1 de Astralaria.pdf.
"""
def get_estimar_data(missing_value = 0, one_hot = True):
    # Variable que contendrá las muestras a predecir
    data_predict = []
    # with open(r'/home/asicoder/Documentos/Projects/Python/UniversityHack-2020/Data/Estimar_UH2020.txt') as read_file:
    with open(r'Data/Estimar_UH2020.txt') as read_file:
        # Leemos línea por línea adaptando las muestras al formato deseado (codificar el valos catastral)
        read_file.readline()
        for line in read_file.readlines():
            # Eliminamos el salto de línea final
            line = line.replace('\n', '')
            # Separamos por el elemento delimitador
            line = line.split('|')
            # Cambiamos CONTRUCTIONYEAR a la antiguedad del terreno
            line[52] = 2020 - int(line[52])
            if line[53] is '':
                line[53] = missing_value
            if one_hot:
                data_predict.append(line[:54] + categorical_encoder_catastral_onehot[line[54]])
            else:
                data_predict.append(line)

    # Finalmente convertimos las muestras preprocesadas a una matriz
    data_predict = np.array(data_predict)    
    ids = data_predict[:, 0]
    data_predict = np.delete(data_predict, 0, axis=1)
    data_predict = data_predict.astype('float32')
    dfids = pd.DataFrame(data=ids)
    dfa = pd.DataFrame(data=data_predict)
    dfa[0] = dfids
    return dfa

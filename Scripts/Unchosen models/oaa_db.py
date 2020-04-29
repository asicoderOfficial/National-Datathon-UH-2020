 # coding=utf-8
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import recall_score,accuracy_score,confusion_matrix, f1_score, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier  
from datasets_get import get_modelar_data
from not_random_test_generator import dividir_dataset
from sampling import smote_enn, smote_tomek, near_miss, condensed_nearest_neighbour, edited_nearest_neighbour, random_over_sampler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import random
import math
import datetime
import featuretools as ft 


INDEX = 0
CLASS = 'CLASS'

modelar_df = get_modelar_data()

#print(modelar_df.shape)
#Obtener train 80% y test 20% aleatoriamente.
X_modelar = modelar_df.loc[:, modelar_df.columns!=CLASS].values

y_modelar = modelar_df.loc[:, CLASS].values
#print('modelar')
#print(X_modelar.shape, y_modelar.shape)
#X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(X_modelar,y_modelar,test_size = 0.2,shuffle=True)
#print('1')
#Obtener train 80% y test 20% NO aleatoriamente.
classes = np.unique(y_modelar)

modelar_train_80_20, modelar_test_80_20 = dividir_dataset(modelar_df)

#print(modelar_train_80_20.shape, modelar_test_80_20.shape)
#print('2')
X_train_80_20  = modelar_train_80_20.loc[:, modelar_train_80_20.columns != CLASS].values
#print(X_train_80_20.shape)
#print('3')

y_train_80_20  = modelar_train_80_20.loc[:, CLASS].values


#print(y_train_80_20.shape)
#print('4')
X_test_80_20   = modelar_test_80_20.loc[:, modelar_test_80_20.columns != CLASS].values
#print(X_test_80_20.shape)
#print('5')
y_test_80_20   = modelar_test_80_20.loc[:, CLASS].values
#print(y_test_80_20.shape)
#print('6')

#Los datos ya están preprocesados con one-hot vectors.
#Modelo XGBClassifier
#xgbClass = xgb.XGBClassifier(learning_rate=0.01, max_depth=2, n_estimators=60, objective='multi:softmax', num_class=7)
xgbClass = xgb.XGBClassifier()

def classifier1(f):
    print('Entra en Clasificador 1')
    #OAA 1º probamos con este
    ovsr1 = OneVsRestClassifier(xgbClass,n_jobs=-1).fit(X_train_80_20,y_train_80_20)
    print('Sale de clasificador 1')
    prediction = ovsr1.predict(X_test_80_20)

    scores = []
    scores.append((f1_score(y_test_80_20,prediction,average='macro'), precision_score(y_test_80_20,prediction,average='micro'), recall_score(y_test_80_20,prediction,average='micro'), accuracy_score(y_test_80_20,prediction)))
    results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])

    f.write('-PRIMER CLASIFICADOR-\n')
    f.write(str(results))
    f.write('\n\n')
    return ovsr1, ovsr1.predict_proba(X_test_80_20)


def data_balancing(method, f):
    #method: 0 SMOTE y ENN, 1 SMOTE y Tomek, 2 SMOTE Y CMTNN
    print('Entra en data balancing')
    f.write('-BALANCEO DE DATOS, técnica: ')
    if method == 0:
        #ROS y ENN
        f.write('ROS + ENN-\n')
        X_s_enn, y_s_enn = random_over_sampler(X_train_80_20, y_train_80_20)
        X_s_enn, y_s_enn = edited_nearest_neighbour(X_s_enn, y_s_enn)
        f.write(str('Número de componentes X: ' + str(X_s_enn.shape[0]) + '\n'))
        f.write(str('Número de componentes y: ' + str(y_s_enn.shape[0]) + '\n'))
        print('Sale de data balancing')
        return X_s_enn, y_s_enn
    else:
        #SMOTE y Tomek
        f.write('SMOTE + Tomek Links\n')
        X_s_tomek, y_s_tomek = smote_tomek(X_train_80_20, y_train_80_20)
        f.write(str('Número de componentes X: ' + str(X_s_tomek.shape[0]) + '\n'))
        f.write(str('Número de componentes y: ' + str(y_s_tomek.shape[0]) + '\n'))
        print('Sale de data balancing')
        return X_s_tomek, y_s_tomek


def classifier2(f, X_balanced, y_balanced):
    #Entrenamiento 2º clasificador para cada técnica balanceo de datos.
    print('entra 2º clasificador')
    ovsr2 = OneVsRestClassifier(xgbClass, n_jobs=-1).fit(X_balanced, y_balanced)
    print('sale 2º clasificador')
    prediction = ovsr2.predict(X_test_80_20)

    scores = []
    scores.append((f1_score(y_test_80_20,prediction,average='macro'), precision_score(y_test_80_20,prediction,average='micro'), recall_score(y_test_80_20,prediction,average='micro'), accuracy_score(y_test_80_20,prediction)))
    results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])

    f.write('-SEGUNDO CLASIFICADOR-\n')
    f.write(str(results))
    f.write('\n\n')
    return ovsr2, ovsr2.predict_proba(X_test_80_20)
    

#Paso final del algoritmo, decisión con thresolds.
#Empleamos las predicciones del primer clasificador (prediction)
#y del 2º con cada técnica (predENN, predTomek)
def get_pred_final(f, ovsr1, ovsr2, pred1, pred2, thresold1, thresold2):
    f.write('PREDICCIÓN FINAL PARA THRESOLD 1: {} Y THRESOLD 2: {}\n'.format(thresold1, thresold2))
    obs = X_test_80_20.shape[0]
    probClasses = ovsr1.classes_
    res = []
    a = 0
    b = 0
    c = 0
    for i in range(obs):
        arr1 = []
        for j in range(7):
            print(pred1[i, j])
            if pred1[i, j] > thresold1:
                arr1.append(probClasses[j])
        #Si solo hay una clase en la lista, es la que predecimos.
        if len(arr1) == 1:
            res.append(arr1[0])
            a+=1
        else:
            #Creamos 2ª lista para comprobar el 2º clasificador.
            arr2 = []
            for j in range(7):
                if pred2[i,j] > thresold2:
                    arr2.append(probClasses[j])
            #Repetimos, si solo hay una clase en la lista, es la que predecimos.
            if len(arr2) == 1:
                res.append(arr2[0])
                b+=1
            #Paso final, no hay más clasificadores.
            #Cogemos clase que ha obtenido mayor probabilidad EN EL 1º.
            #Es posible el empate?
            else:
                res.append(probClasses[np.argmax(pred1[i,:])])
                c += 1
    f.write(str(a) + ' ' + str(b) + ' ' + str(c) + '\n')
    return res


def informe():
    #Creamos text file para el informe con el día y la hora 
    filename = str('Resultados/OAA-DB' + str(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
    print('COMIENZA INFORME')
    #thresolds = [(10,50), (20, 50), (30,50), (40, 50), (50, 50), (60, 50), (70, 50), (10,40)]
    thresolds = [(70, 50), (80, 50), (20, 60), (20, 80), (30, 70), (50, 50), (60, 60), (70, 70), (90, 90), (30, 50)]
    with open(filename, 'w+') as f:
        ovsr1, pred1 = classifier1(f)
        for i in range(2):
            X_balanced, y_balanced = data_balancing(i, f)
            for par in thresolds:
                ovsr2, pred2 = classifier2(f, X_balanced, y_balanced)
                res = get_pred_final(f, ovsr1, ovsr2, pred1, pred2, par[0], par[1])
                scores = []
                scores.append((f1_score(y_test_80_20,res,average='macro'), precision_score(y_test_80_20,res,average='micro'), recall_score(y_test_80_20,res,average='micro'), accuracy_score(y_test_80_20,res)))
                results = pd.DataFrame(scores,columns=['f1','precision','recall','accuracy'])
                print(str(results))
                f.write(str(results))
                f.write('\n----------------------------------------------------------------------------------------\n')
                f.write(classification_report(y_test_80_20, res))
                print(str(classification_report(y_test_80_20, res)))
                f.write('----------------------------------------------------------------------------------------\n\n')
informe()

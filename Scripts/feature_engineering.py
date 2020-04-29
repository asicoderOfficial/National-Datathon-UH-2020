import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from datasets_get import get_modelar_data, get_estimar_data, getX, getY, get_modelar_data_ids, get_categories_list
from scipy.spatial import cKDTree
from scipy.special import softmax
import featuretools as ft


"""
Método que añade las 7 features según los K-1 vecinos más próximos.
Se explica en el apartado 4.3 del archivo Astralaria.pdf.
"""
def coordinates_fe(X_modelar, y_modelar, X_estimar, K=4):
    est_IDs = X_estimar[0]
    X_est_mod = pd.concat([X_modelar, X_estimar], sort=False)
    coords = X_est_mod[[1, 2]].rename(columns={1:'X', 2:'Y'})

    spatialTree = cKDTree(np.c_[coords.X.ravel(),coords.Y.ravel()])

    X_est_mod.drop([0],inplace=True,axis=1)
    X_estimar.drop([0],inplace=True,axis=1)
    X_modelar.drop([0],inplace=True,axis=1)

    classifier = xgb.XGBClassifier()
    ovsr = OneVsRestClassifier(classifier,n_jobs=-1).fit(X_modelar,y_modelar)
    pred_estimar = ovsr.predict_proba(X_estimar)

    offset = X_modelar.shape[0]
    classes = get_categories_list()
    col_names = []

    for i in range(7):
        col_names.append('coords_' + classes[i])

    cont = [] 

    for i in range(X_est_mod.shape[0]):
        
        indices = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        
        neigh_dist, neigh_indices = spatialTree.query([[coords.iloc[i,0],coords.iloc[i,1]]],k=K)
        
        for j in range(1,K):
            # -Si la variable se encuentra en X_modelar,
            # para cada vecino sumamos 1 a la variable contexto de la clase de la finca 
            # -Si la variable se encuentra en X_estimar, 
            # sumamos la probabilidad de que sea un vecino.
            if neigh_indices[0][j] < offset : 
                indices[int(y_modelar.loc[neigh_indices[0][j], 'CLASS'])] += 1
            else:
                
                indices = np.add(indices, pred_estimar[neigh_indices[0][j]-offset,:])

        cont.append(indices)

    indexes_est = []
    for i in range(X_estimar.shape[0]):
        indexes_est.append(i)

    context  = pd.DataFrame(data=cont,columns=col_names)
    context_modelar = context.loc[:offset-1]
    
    context_estimar = context.loc[offset:]
    context_estimar.index = range(5618)

    #Visualizamos las primeras 20 features añadidas a cada dataset.
    print('Nuevas 7 features dataset modelar, primeras 20.')
    print(context_modelar.head(20))
    print('Nuevas 7 features dataset estimar, primeras 20.')
    print(context_estimar.head(20))
    

    for column in col_names:
        X_modelar[column] = context_modelar[column]
        X_estimar[column] = context_estimar[column] 

    return X_modelar.values, X_estimar.values, est_IDs


"""
Método que realiza la reducción de la dimensionalidad por densidad RGB.
Para cada muestra, primero obtiene la suma de todos los deciles de cada color.
Seguidamente las compara, y sustituye todas las features RGB en una por color.
Se asigna un valor a cada una de 0, 1 o 2, según qué suma de cada color ha sido
la menor, media o mayor respectivamente, para cada muestra.
Se menciona en el apartado 4.2 de Astralaria.pdf.
"""
def density_RGB_scale(df):
    colorRed = []
    colorGreen = []
    colorBlue = []
    for j in range(df.shape[0]):
        sumR = 0
        sumG = 0
        sumB = 0
        for i in range(3,14):
            sumR += df.loc[j,i]
        for i in range(14,25):
            sumG += df.loc[j,i]
        for i in range(25,36):
            sumB += df.loc[j,i]
        sums = [sumR, sumG, sumB]
        min_index = sums.index(min(sums))
        max_index = sums.index(max(sums))
        if min_index == 0:
            colorRed.append(0)
        elif min_index == 1:
            colorGreen.append(0)
        elif min_index == 2:
            colorBlue.append(0)
        if max_index == 0:
            colorRed.append(2)
        elif max_index == 1:
            colorGreen.append(2)
        elif max_index == 2:
            colorBlue.append(2)
        if len(colorRed) < len(colorGreen):
            colorRed.append(1)
        elif len(colorGreen) < len(colorRed):
            colorGreen.append(1)
        elif len(colorBlue) < len(colorGreen):
            colorBlue.append(1)
    for i in range(3,36):
        del df[i]
    df['RED'] = colorRed
    df['GREEN'] = colorGreen
    df['BLUE'] = colorBlue

    return df  


"""
Método que condensa los valores NIR en una feature, que puede
contener los valores 0 o 1, según si la suma de los deciles de
una muestra es menor o mayor que la media de la suma general,
respectivamente.
Se menciona en el apartado 4.2 de Astralaria.pdf.
"""
def density_NIR_conditional_mean(df):
    colorNIR = []
    sums = []
    total_sum = 0
    for j in range(df.shape[0]):
        sumNIR = 0
        for i in range(36, 47):
            sumNIR += df.loc[j,i]
        total_sum += sumNIR
        sums.append(sumNIR)
    mean = total_sum / df.shape[0]
    for value in sums:
        if value <= mean:
            colorNIR.append(0)
        else:
            colorNIR.append(1)
    for i in range(36,47):
        del df[i]
    df['NIR_MEAN_COND'] = colorNIR

    return df


"""
Método que implementa la técninca de Deep Feature Synthesis.
Emplea los valores por defecto.
Se menciona en el apartado 4.3 de Astralaria.pdf.
"""
def dfs_fe(df):
    columns_ids = list(df.columns.values)
    print(columns_ids)
    for value in columns_ids:
        if isinstance(value, int):
            df.rename(columns={value : str(value)}, inplace=True)
    es = ft.EntitySet(id='main')
    es.entity_from_dataframe(entity_id='data', dataframe=df, make_index=True, index='index')
    feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='data')
    print(feature_matrix)
    print(feature_defs)
    return df.values


"""
Método que elimina las num variables menos significativas
según el GridSearch realizado para XGBClassifier.
"""
def reduce_dimension_modelar(modelar_df, num=30):
    if num > 55:
        print('num no mayor a 55')
    else:
        importance_df = pd.read_csv('Importancia de parametros.csv')
        indexes_list = list(importance_df['Index'])
        indexes_list[::-1]
        indexes_quited = []
        i = 0
        j = 0
        while j < num:
            if not 53 <= indexes_list[i] <= 65 and indexes_list[i] != 1:
                indexes_quited.append(indexes_list[i])
                del modelar_df[indexes_list[i]]
                j += 1
            i+=1
        return modelar_df


"""
Método que elimina los deciles 2,3,4,6,7 y 8 de cada color.
Mencionado en el apartado 4.2 de Astralaria.pdf.
"""
def reduce_colors(df):
    indices_start = [4, 5, 6, 8, 9, 10]
    for i in range(len(indices_start)):
        df.drop([indices_start[i], indices_start[i]+11, indices_start[i]+22, indices_start[i]+33],inplace=True,axis=1)
    return df


"""
Método que sustituye las variables geométricas
por su media para cada muestra.
"""
def reduce_geometry_average(df):
    avgs = []
    for i in range(df.shape[0]):
        avgs.append((df.loc[i, 48] + df.loc[i, 49] + df.loc[i, 50] + df.loc[i, 51]) / 4)
    del df[48]
    del df[49]
    del df[50]
    del df[51]
    df['GEOM_AVG'] = avgs
    
    return df
 # coding=utf-8
import pandas as pd

"""
def dividir_dataset(df, train_percentage=0.8, test_percentage=0.2, randomized=False, residential_perc=3000, industrial_perc=1400, public_perc=900, retail_perc=700, office_perc=400, other_perc=600, agriculture_perc=100):
    np.random.seed(7)
    if train_percentage + test_percentage != 1:
        print("El dataset debe emplearse entero, los porcentajes de train y test deben sumar 1.")
    elif randomized == True:
        train,test = train_test_split(df, test_size=0.2)
        return train, test
    else:
        percentages_list = [residential_perc, industrial_perc, public_perc, retail_perc, office_perc, other_perc, agriculture_perc]
        df_list = []
        
        for i in range(7):
            current_df = df.loc[df['CLASS'] == i]
            remove_num = percentages_list[i]
            df_list.append(current_df.sample(n=remove_num))
        for i in range(1,7):
            df_list[0] = df_list[0].append(df_list[i], ignore_index=True)
        df = pd.concat([df, df_list[0]]).drop_duplicates(keep=False)
        dfres = df.loc[df['CLASS'] == 0].sample(n=6000)
        dfothers = df.loc[df['CLASS'] != 0]
        df_train = dfres.append(dfothers, ignore_index=True)
        #Devuelve 3 valores: X_modelar, Y_modelar, 
        for i in range(7):
            print('Class ' + str(i) + ' train: ' + str(df_train.loc[df_train['CLASS'] == i].shape[0]))
            print('Class ' + str(i) + ' test: ' + str(df_list[0].loc[df_list[0]['CLASS'] == i].shape[0]))

        print(df_train)
        print(df_list[0])
        return df_train.sample(frac=1), df_list[0].sample(frac=1)
"""

"""
Método que ejecuta el Random Undersampling de RESIDENTIAL.
Mencionado en el apartado 5.3 de Astralaria.pdf.
"""
def random_undersample_residential(df, residential_samples=6000):
    n_drop = df.loc[df['CLASS'] == 0].shape[0] - 6000
    df_res_drop = df.loc[df['CLASS'] == 0].sample(n=n_drop)
    df = pd.concat([df, df_res_drop]).drop_duplicates(keep=False).sample(frac=1)
    df.index = range(df.shape[0])
    return df

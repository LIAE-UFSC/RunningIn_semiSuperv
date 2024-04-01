# Importa bibliotecas

import pandas as pd


def filtro_A(y_pred, df_X):
    df_pred = pd.DataFrame(y_pred, columns=['Resultado'])
    df_Xy = pd.merge(df_X, df_pred, left_index=True, right_index=True)
    Xy = df_Xy.loc[(df_Xy.N_ensaio == 0), :]
    # Unidades
    Xy_a2 = Xy.loc[(df_Xy.Unidade == 'a2'), :]
    Xy_a3 = Xy.loc[(df_Xy.Unidade == 'a3'), :]
    Xy_a4 = Xy.loc[(df_Xy.Unidade == 'a4'), :]
    Xy_a5 = Xy.loc[(df_Xy.Unidade == 'a5'), :]
    # Original
    y = Xy['Resultado']
    X = Xy.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
    # X dividido por unidades
    X_a2 = Xy_a2.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
    X_a3 = Xy_a3.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
    X_a4 = Xy_a4.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
    X_a5 = Xy_a5.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
    # y dividido
    y_a2 = Xy_a2['Resultado']
    y_a3 = Xy_a3['Resultado']
    y_a4 = Xy_a4['Resultado']
    y_a5 = Xy_a5['Resultado']

    return X, y, X_a2, X_a3, X_a4, X_a5, y_a2, y_a3, y_a4, y_a5, Xy

def retira_parte_incerta(array_true,array_pred):
    array_true_sem_incerteza = tuple(x for x in array_true if x != -1)
    array_pred = tuple(array_pred[i] for i in range(len(array_true)) if array_true != -1)

    return array_true_sem_incerteza, array_pred
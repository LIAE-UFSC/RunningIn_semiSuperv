# Importa bibliotecas

import pandas as pd


def filtro(y_pred, df_X, modelo):
    df_pred = pd.DataFrame(y_pred, columns=['Resultado'])
    df_Xy = pd.merge(df_X, df_pred, left_index=True, right_index=True)
    Xy = df_Xy.loc[(df_Xy.N_ensaio == 0), :]
    if modelo == 'a':
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
    elif modelo == 'b':
        # Unidades
        Xy_b5 = Xy.loc[(df_Xy.Unidade == 'b5'), :]
        Xy_b7 = Xy.loc[(df_Xy.Unidade == 'b7'), :]
        Xy_b8 = Xy.loc[(df_Xy.Unidade == 'b8'), :]
        Xy_b10 = Xy.loc[(df_Xy.Unidade == 'b10'), :]
        Xy_b11 = Xy.loc[(df_Xy.Unidade == 'b11'), :]
        Xy_b12 = Xy.loc[(df_Xy.Unidade == 'b12'), :]
        Xy_b15 = Xy.loc[(df_Xy.Unidade == 'b15'), :]
        # Original
        y = Xy['Resultado']
        X = Xy.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        # X divided by units
        X_b5 = Xy_b5.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b7 = Xy_b7.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b8 = Xy_b8.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b10 = Xy_b10.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b11 = Xy_b11.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b12 = Xy_b12.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        X_b15 = Xy_b15.drop(['Resultado', 'Unidade', 'N_ensaio'], axis=1)
        # y divided
        y_b5 = Xy_b5['Resultado']
        y_b7 = Xy_b7['Resultado']
        y_b8 = Xy_b8['Resultado']
        y_b10 = Xy_b10['Resultado']
        y_b11 = Xy_b11['Resultado']
        y_b12 = Xy_b12['Resultado']
        y_b15 = Xy_b15['Resultado']

        return X, y, X_b5, X_b7, X_b8, X_b10, X_b11, X_b12, X_b15, y_b5, y_b7, y_b8, y_b10, y_b11, y_b12, y_b15, Xy

def retira_parte_incerta(array_true,array_pred):
    array_true_sem_incerteza = tuple(x for x in array_true if x != -1)
    array_pred = tuple(array_pred[i] for i in range(len(array_true)) if array_true != -1)

    return array_true_sem_incerteza, array_pred
# Importa bibliotecas
from buildDataset import en_dict2data, preproc_dataset
import pandas as pd
import numpy as np



def retira_y(dataframe):
    return tuple(dataframe.pop('Amaciado'))

def prep_data(N_instantes,D_espacamento,M_janela,df_analise,lista_grands,tMin,tMax):
        return preproc_dataset(df_analise, lista_grands, M_janela, N_instantes, D_espacamento, tempoMin = tMin, tempoMax = tMax, dropTempo = False, scale=False)


def label_df(df):
    df.loc[:, 'Amaciado'] = -1  # Sem valores
    df.loc[(df.Tempo <= 5) & (df.N_ensaio == 0), 'Amaciado'] = 0
    df.loc[(df.N_ensaio > 0) & (df.N_ensaio > 0), 'Amaciado'] = 1
    return df

def separa_Xy(df):
    y = df.loc[:,'Amaciado']
    X = df.drop(['Amaciado'], axis= 1 )
    return X,y


def retira_valores(df):
    X = df.drop(['Unidade','N_ensaio','Tempo'], axis= 1 )
    return X

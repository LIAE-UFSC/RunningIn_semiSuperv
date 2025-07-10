from delayedsw import DelayedSlidingWindow
import pandas as pd
import numpy as np

def splitXY(data):
    # TODO: write docstring
    
    y = data.loc[:,'Amaciado']
    X = data.drop(['Amaciado'], axis= 1 )
    return X,y

def filterTime(data, tMin, tMax):
    # TODO: write docstring

    return data[(data['Tempo'] >= tMin) & (data['Tempo'] <= tMax)]

def labelData(data):
    # TODO: write docstring

    data.loc[:, 'Amaciado'] = -1  # Sem valores
    data.loc[(data.Tempo <= 5) & (data.N_ensaio == 0), 'Amaciado'] = 0
    data.loc[(data.N_ensaio > 0) & (data.N_ensaio > 0), 'Amaciado'] = 1
    return data

def preprocessData(data,window_size,delay,moving_average,feat,tMin=0,tMax=np.inf):
    # TODO: write docstring
    
    # TODO: Implement the preprocessing steps
    pass


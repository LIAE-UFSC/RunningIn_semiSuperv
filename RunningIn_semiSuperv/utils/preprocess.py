from delayedsw import DelayedSlidingWindow, MovingAverageTransformer
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

def preprocessData(data, window_size, delay, feat, moving_average = 1, tMin=0,tMax=np.inf):
    # TODO: write docstring

    group_columns = ['Unidade','N_ensaio']
    order_columns = ['Tempo']
    
    # Filter rows based on time
    data = filterTime(data, tMin, tMax)

    # Label the data
    data = labelData(data)

    # Group by 'Unidade' and 'N_ensaio' and order by 'Tempo'
    data = data.sort_values(by=group_columns + order_columns)

    # Split X and y
    X, y = splitXY(data)

    # Moving average preprocessing
    if moving_average > 1:
        moving_average_transformer = MovingAverageTransformer(window=moving_average)
        X = moving_average_transformer.fit_transform(X)

    # Sliding window preprocessing
    sliding_window = DelayedSlidingWindow(window_size=window_size, delay_space=delay, columns_to_transform=feat,
                                            split_by=group_columns, order_by=order_columns, inclupassde_order=True, include_split=True)
    X = sliding_window.fit_transform(X)

    # Join X and y and drop NaN values
    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    data = data.reset_index(drop=True)

    # TODO: refactor code in order to avoid splitting X and y again after preprocessing

    # Split X and y again after preprocessing
    X, y = splitXY(data)

    return X, y


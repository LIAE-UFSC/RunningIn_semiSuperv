import pandas as pd
import numpy as np
from slidingWindow import DSSW_ensaio
from sklearn.preprocessing import StandardScaler

def separa_xy_runin(data:pd.DataFrame):
    """Separa observações e respostas de um conjunto de dados de amaciamento.

        Parâmetros
        ----------
        dataset : DataFrame
        Dataframe com os dados do conjunto

        Retorna
        -------
        X: array-like of shape (n_samples, n_features)
            Vetores de observação (grandezas c/ sliding window)

        Y: list[int]
            Resposta (estado de amaciamento do compressor)
        """

    y = data['Amaciado']
    X = data.drop(['Unidade','N_ensaio','Amaciado'],axis=1)

    return X,y

def en_dict2data(ensaios_total:list[dict]):
    """Converte um dicionário de ensaios em um dataFrame.

        Parâmetros
        ----------
        ensaios_total : list[dict]
            lista de dicionário com os seguintes elementos
                'unidade': nome da unidade
                'ensaios': dados importados de ensaios, em ordem cronológica

        Retorna
        -------
        dataset_total: pd.DataFrame
            Dados formatados de todas as unidades, adicionando as colunas 'Unidade' e 'N_ensaio'
        """

    dataset_total = pd.DataFrame()

    for un in ensaios_total:
        dataUnidade = pd.DataFrame()
        for k,en in enumerate(un['ensaios']):
            en['N_ensaio'] = k
            dataUnidade = pd.concat([dataUnidade,en])
        dataUnidade['Unidade'] = un['unidade']
        dataset_total = pd.concat([dataset_total,dataUnidade])

    return dataset_total

def preproc_dataset(dataset_raw:pd.DataFrame, grandezas: list[str], M_janela:int, N_features: int, D_espacamento:int, tempoMax: int, tempoMin: int = 1, dropTempo:bool = True, scale:bool = False):
    """Processa as time series de cada ensaio utilizando filtro de médias móveis e delayed space sliding window

    Parâmetros
    ----------
    dataset_raw : DataFrame
        Dataframe com os dados brutos do conjunto

    grandeza: list[str]
        Lista com o nome das grandezas desejadas no formato "Nicolas". Nomes disponíveis:
                   'CorrenteRMS'               'CorrenteVariancia'            'CorrenteCurtose'
            'VibracaoCalotaInferiorRMS' 'VibracaoCalotaInferiorCurtose' 'VibracaoCalotaInferiorVariancia'
            'VibracaoCalotaSuperiorRMS' 'VibracaoCalotaSuperiorCurtose' 'VibracaoCalotaSuperiorVariancia'
                      'Vazao'

    M_janela: int
        Valor M, largura da janela do filtro de médias móveis

    N_features: int
        Valor N, número de elementos do vetor de predição (features) utilizados na formatação

    D_espacamento: int
        Valor D, espaçamento temporal entre features

    tempoMax: int
        Instante máximo para análise

    tempoMin: int
        Instante mínimo para análise (default = 1)

    dropTempo: bool
        Manter ou não coluna do tempo no dataset gerado (default = True)

    scale: bool (default = False)
        Escalonar ou não os dados com sklearn.preprocessing.StandardScaler()

    Retorna
    -------
    dataset_proc: pd.DataFrame
        Dados processados do conjunto

    dataset_scaled: pd.DataFrame
        Dados processados e escalonados do conjunto

    scaler: StandardScaler()
        Escalonador ajustado com os dados processados do conjunto
    """

    dataset_proc = pd.DataFrame()

    # Nome das unidades e ensaios
    unique_ensaio = dataset_raw[['Unidade','N_ensaio']].drop_duplicates().reset_index(drop=True)

    # Aplica o tratamento para todas as grandezas desejadas
    for gr in grandezas:

        unidade_grandeza = pd.DataFrame()

        # Concatena verticalmente os dados tratados de todos os ensaios
        for ind in unique_ensaio.index:
            data_select = unique_ensaio.iloc[[ind]].merge(dataset_raw) # Seleciona os dados do ensaio
            unidade_grandeza = pd.concat([unidade_grandeza, DSSW_ensaio(data_select, gr, M_janela, N_features, D_espacamento, tempoMax, dropTempo=dropTempo)]) # Aplica processamento

        # Adiciona as colunas de diferentes grandezas ao dataset
        try:
            dataset_proc = pd.concat([dataset_proc,unidade_grandeza], axis=1)
        except:
            print("Grandeza ", gr, "não encontrada")
        
    # Retira grandezas duplicadas na concatenação
    dataset_proc = dataset_proc.loc[:,~dataset_proc.columns.duplicated()].reset_index(drop=True)

    if scale:
        un_en_train = dataset_proc[["Unidade","N_ensaio","Amaciado"]]
        valTrain = dataset_proc.drop(["Unidade","N_ensaio","Amaciado"], axis=1)
        scaler = StandardScaler()
        valTrain.loc[:,:] = scaler.fit_transform(valTrain)
        dataset_scaled = pd.concat([valTrain,un_en_train],axis=1)
        return dataset_proc, dataset_scaled, scaler
    else:
        return dataset_proc
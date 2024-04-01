import pandas as pd
import numpy as np

def delayedSpace_slidingWindow(ensaio_raw: pd.DataFrame, M_janela: int, N_features: int, D_espacamento: int):
    """Cria os features por rolling window, com N features espaçados em D amostras

    Parâmetros
    ----------
    ensaio_raw: pd.DataFrame
        Dados brutos de um ensaio

    M_janela: int
        Valor M, largura da janela do filtro de médias móveis

    N_features: int
        Valor N, número de elementos do vetor de predição (features) utilizados na formatação

    D_espacamento: int
        Valor D, espaçamento temporal entre features

    Retorna
    -------
    ensaio_espacado: pd.DataFrame
        Dados formatados do ensaio
    """
    
    for n in range(1,N_features):
        # Cria as N colunas com shift de D amostras
        ensaio_raw[f'{M_janela}(K-{D_espacamento*n})'] = ensaio_raw[f'{M_janela}'].shift(D_espacamento*n)

    ensaio_espacado = ensaio_raw.copy()

    return ensaio_espacado

def DSSW_ensaio(ensaio: pd.DataFrame, grandeza: str, M_janela: int, N_features: int, D_espacamento: int, tempoMax: float, tempoMin:float = 1, dropTempo = True):
    """Cria o dataset de um ensaio por rolling window a partir dos dados brutos

    Parâmetros
    ----------
    ensaio_raw : pd.DataFrame
        Dados brutos de um ensaio

    grandeza: str
        Nome da grandeza desejada no formato "Nicolas". Nomes disponíveis:
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

    Retorna
    -------
    dados_espacados: pd.DataFrame
        Dados formatados do ensaio
    """

    # Seleciona o tempo e a grandeza desejada
    dados_espacados = ensaio[['Tempo','N_ensaio','Unidade','Amaciado',grandeza]].copy()

    # Seleciona as linhas dentro dos limites de tempo desejados
    dados_espacados = dados_espacados[(dados_espacados.Tempo<=tempoMax) & (dados_espacados.Tempo>=tempoMin)]

    # Retira linhas NaN
    dados_espacados = dados_espacados.dropna(how='any')
    # Atualiza o indice do dataframe do pandas
    dados_espacados.reset_index(drop=True, inplace=True)

    # Aplica filtro de médias móveis p/ cada valor M e cria as colunas de delayed space sliding window
    
    dados_espacados[f'{grandeza}_MA_{M_janela}'] = dados_espacados[grandeza].rolling(M_janela, min_periods=1).mean()
    media = delayedSpace_slidingWindow(dados_espacados,f'{grandeza}_MA_{M_janela}',N_features, D_espacamento).drop(grandeza, axis=1)
    dados_espacados = media.copy()

    if dropTempo:
        dados_espacados = dados_espacados.drop("Tempo",axis=1)

    # Retira linhas NaN
    dados_espacados = dados_espacados.dropna(how='any')

    return dados_espacados

def DSSW_unidade(unidade_raw: pd.DataFrame, grandezas: list[str], M_janela:int, N_features: int, D_espacamento:int, tempoMax: float, tempoMin: float = 1, dropTempo:bool = True):
    """Cria o dataset de uma unidade com N_features features, para várias grandezas, espaçamentos D e filtros de médias móveis de largura M.

    Parâmetros
    ----------
    unidade_raw : DataFrame
        Dataframe com os dados brutos da unidade

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

    tempoMax: float
        Instante máximo para análise

    tempoMin: float
        Instante mínimo para análise (default = 1)

    dropTempo: bool
        Manter ou não coluna do tempo no dataset gerado (default = True)

    Retorna
    -------
    dataset_total: pd.DataFrame
        Dados formatados da unidade
    """

    unidade_tratado = pd.DataFrame()

    # Aplica o tratamento para todas as grandezas desejadas
    for gr in grandezas:

        unidade_grandeza = pd.DataFrame()

        # Concatena verticalmente os dados tratados de todos os ensaios
        for en in np.unique(unidade_raw['N_ensaio']):
            data_select = unidade_raw.loc[unidade_raw['N_ensaio']==en]
            unidade_grandeza = pd.concat([unidade_grandeza, DSSW_ensaio(data_select, gr, M_janela, N_features, D_espacamento, tempoMax, tempoMin, dropTempo)])

        # Adiciona as colunas de diferentes grandezas ao dataset
        try:
            unidade_tratado = pd.concat([unidade_tratado,unidade_grandeza], axis=1)
        except:
            print("Grandeza ", gr, "não encontrada")
        
    # Retira grandezas duplicadas na concatenação
    unidade_tratado = unidade_tratado.loc[:,~unidade_tratado.columns.duplicated()].reset_index(drop=True)

    if dropTempo:
        unidade_tratado = unidade_tratado.drop("Tempo",axis=1)

    return unidade_tratado
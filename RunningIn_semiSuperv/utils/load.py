import pandas as pd
from pathlib import PurePath
import os

def folderdict2data(test_all:list[dict]):
    """Converte um dicionário de ensaios em um dataFrame.

        Parâmetros
        ----------
        test_all : list[dict]
            lista de dicionário com os seguintes elementos
                'unidade': nome da unidade
                'ensaios': dados importados de ensaios, em ordem cronológica

        Retorna
        -------
        dataset_total: pd.DataFrame
            Dados formatados de todas as unidades, adicionando as colunas 'Unidade' e 'N_ensaio'
        """

    dataset_total = pd.DataFrame()

    for un in test_all:
        dataUnidade = pd.DataFrame()
        for k,en in enumerate(un['ensaios']):
            en['N_ensaio'] = k
            dataUnidade = pd.concat([dataUnidade,en])
        dataUnidade['Unidade'] = un['unidade']
        dataset_total = pd.concat([dataset_total,dataUnidade])

    return dataset_total

def load_base_data(model):
    # Importa dados
    dataFolder = PurePath(os.getcwd(),'LabeledData')
    test_all = [
        {
            'unidade': 'a2',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'A2_N_2019_07_09.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A2_A_2019_08_08.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A2_A_2019_08_28.csv'))]
        },
        {
            'unidade': 'a3',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'A3_N_2019_12_04.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A3_A_2019_12_09.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A3_A_2019_12_11.csv'))]
        },
        {
            'unidade': 'a4',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'A4_N_2019_12_16.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A4_A_2019_12_19.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A4_A_2020_01_06.csv'))]
        },
        {
            'unidade': 'a5',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'A5_N_2020_01_22.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A5_A_2020_01_27.csv')),
                        pd.read_csv(PurePath(dataFolder, 'A5_A_2020_01_28.csv'))]
        },
        {
            'unidade': 'b5',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B5_N_2020_10_16.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B5_A_2020_10_22.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B5_A_2020_10_27.csv'))]
        },
        {
            'unidade': 'b7',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B7_N_2021_02_05.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B7_A_2021_02_08.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B7_A_2021_02_15.csv'))]
        },
        {
            'unidade': 'b8',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B8_N_2021_02_18.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B8_A_2021_02_22.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B8_A_2021_02_26.csv'))]
        },
        {
            'unidade': 'b10',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B10_N_2021_03_22.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B10_A_2021_03_25.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B10_A_2021_03_30.csv'))]
        },
        {
            'unidade': 'b11',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B11_N_2021_04_05.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B11_A_2021_04_08.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B11_A_2021_04_22.csv'))]
        },
        {
            'unidade': 'b12',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B12_N_2021_04_27.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B12_A_2021_04_30.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B12_A_2021_05_04.csv'))]
        },
        {
            'unidade': 'b15',
            'ensaios': [pd.read_csv(PurePath(dataFolder, 'B15_N_2021_05_31.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B15_A_2021_06_09.csv')),
                        pd.read_csv(PurePath(dataFolder, 'B15_A_2021_06_15.csv'))]
        },
    ]

    dataUn = folderdict2data(test_all)

    del test_all

    # Separa conjuntos de treino e teste
    if model == "a":
        data = dataUn[dataUn["Unidade"] < "b"]
    elif model == "b":
        data = dataUn[dataUn["Unidade"] >= "b"]

    return data
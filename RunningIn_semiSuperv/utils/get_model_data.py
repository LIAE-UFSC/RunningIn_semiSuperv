# Importa bibliotecas
import os
from buildDataset import en_dict2data, preproc_dataset
import pandas as pd


def analise_modelos(modelo, score: str = "MCC", unTest: str = "a5", dropTempo: bool = True, param=None):
    grands = [
        ['CorrenteRMS'], ['CorrenteVariancia'], ['CorrenteCurtose'],
        ['VibracaoCalotaInferiorRMS'], ['VibracaoCalotaInferiorCurtose'],
        ['VibracaoCalotaInferiorVariancia'], ['VibracaoCalotaSuperiorRMS'],
        ['VibracaoCalotaSuperiorCurtose'], ['VibracaoCalotaSuperiorVariancia'],
        ['Vazao']
    ]  # Grandezas para análise

    # Importa dados
    dataFolder = os.getcwd() + '\\LabeledData'
    ensaios_total = [
        {
            'unidade': 'a2',
            'ensaios': [pd.read_csv(dataFolder + '\\A2_N_2019_07_09.csv'),
                        pd.read_csv(dataFolder + '\\A2_A_2019_08_08.csv'),
                        pd.read_csv(dataFolder + '\\A2_A_2019_08_28.csv')]
        },
        {
            'unidade': 'a3',
            'ensaios': [pd.read_csv(dataFolder + '\\A3_N_2019_12_04.csv'),
                        pd.read_csv(dataFolder + '\\A3_A_2019_12_09.csv'),
                        pd.read_csv(dataFolder + '\\A3_A_2019_12_11.csv')]
        },
        {
            'unidade': 'a4',
            'ensaios': [pd.read_csv(dataFolder + '\\A4_N_2019_12_16.csv'),
                        pd.read_csv(dataFolder + '\\A4_A_2019_12_19.csv'),
                        pd.read_csv(dataFolder + '\\A4_A_2020_01_06.csv')]
        },
        {
            'unidade': 'a5',
            'ensaios': [pd.read_csv(dataFolder + '\\A5_N_2020_01_22.csv'),
                        pd.read_csv(dataFolder + '\\A5_A_2020_01_27.csv'),
                        pd.read_csv(dataFolder + '\\A5_A_2020_01_28.csv')]
        },
        {
            'unidade': 'b5',
            'ensaios': [pd.read_csv(dataFolder + '\\B5_N_2020_10_16.csv'),
                        pd.read_csv(dataFolder + '\\B5_A_2020_10_22.csv'),
                        pd.read_csv(dataFolder + '\\B5_A_2020_10_27.csv')]
        },
        {
            'unidade': 'b7',
            'ensaios': [pd.read_csv(dataFolder + '\\B7_N_2021_02_05.csv'),
                        pd.read_csv(dataFolder + '\\B7_A_2021_02_08.csv'),
                        pd.read_csv(dataFolder + '\\B7_A_2021_02_15.csv')]
        },
        {
            'unidade': 'b8',
            'ensaios': [pd.read_csv(dataFolder + '\\B8_N_2021_02_18.csv'),
                        pd.read_csv(dataFolder + '\\B8_A_2021_02_22.csv'),
                        pd.read_csv(dataFolder + '\\B8_A_2021_02_26.csv')]
        },
        {
            'unidade': 'b10',
            'ensaios': [pd.read_csv(dataFolder + '\\B10_N_2021_03_22.csv'),
                        pd.read_csv(dataFolder + '\\B10_A_2021_03_25.csv'),
                        pd.read_csv(dataFolder + '\\B10_A_2021_03_30.csv')]
        },
        {
            'unidade': 'b11',
            'ensaios': [pd.read_csv(dataFolder + '\\B11_N_2021_04_05.csv'),
                        pd.read_csv(dataFolder + '\\B11_A_2021_04_08.csv'),
                        pd.read_csv(dataFolder + '\\B11_A_2021_04_22.csv')]
        },
        {
            'unidade': 'b12',
            'ensaios': [pd.read_csv(dataFolder + '\\B12_N_2021_04_27.csv'),
                        pd.read_csv(dataFolder + '\\B12_A_2021_04_30.csv'),
                        pd.read_csv(dataFolder + '\\B12_A_2021_05_04.csv')]
        },
        {
            'unidade': 'b15',
            'ensaios': [pd.read_csv(dataFolder + '\\B15_N_2021_05_31.csv'),
                        pd.read_csv(dataFolder + '\\B15_A_2021_06_09.csv'),
                        pd.read_csv(dataFolder + '\\B15_A_2021_06_15.csv')]
        },
    ]

    dataUn = en_dict2data(ensaios_total)

    del ensaios_total

    # Separa conjuntos de treino e teste
    if modelo == "a":
        data = dataUn[dataUn["Unidade"] < "b"]
    elif modelo == "b":
        data = dataUn[dataUn["Unidade"] >= "b"]

    return data
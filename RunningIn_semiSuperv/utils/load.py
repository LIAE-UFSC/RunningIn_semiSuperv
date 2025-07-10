import pandas as pd
from pathlib import PurePath
import os

def joinTests(test_all:list[dict]):
    # TODO: write docstring

    data_total = pd.DataFrame()

    for un in test_all: # Iterate through each unit
        data_unit = pd.DataFrame()
        for k,en in enumerate(un['tests']): # Iterate through each test
            en['N_ensaio'] = k # Add the test number
            data_unit = pd.concat([data_unit,en])
        data_unit['Unidade'] = un['unit'] # Add the unit name
        data_total = pd.concat([data_total,data_unit])

    return data_total

def loadBaseData(model):
    # TODO: write docstring

    dataFolder = PurePath(os.getcwd(),'LabeledData')

    # Generate a list of dictionaries with the test data
    test_all = [
        {
            'unit': 'a2',
            'tests': [pd.read_csv(PurePath(dataFolder, 'A2_N_2019_07_09.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A2_A_2019_08_08.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A2_A_2019_08_28.csv'))]
        },
        {
            'unit': 'a3',
            'tests': [pd.read_csv(PurePath(dataFolder, 'A3_N_2019_12_04.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A3_A_2019_12_09.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A3_A_2019_12_11.csv'))]
        },
        {
            'unit': 'a4',
            'tests': [pd.read_csv(PurePath(dataFolder, 'A4_N_2019_12_16.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A4_A_2019_12_19.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A4_A_2020_01_06.csv'))]
        },
        {
            'unit': 'a5',
            'tests': [pd.read_csv(PurePath(dataFolder, 'A5_N_2020_01_22.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A5_A_2020_01_27.csv')),
                      pd.read_csv(PurePath(dataFolder, 'A5_A_2020_01_28.csv'))]
        },
        {
            'unit': 'b5',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B5_N_2020_10_16.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B5_A_2020_10_22.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B5_A_2020_10_27.csv'))]
        },
        {
            'unit': 'b7',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B7_N_2021_02_05.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B7_A_2021_02_08.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B7_A_2021_02_15.csv'))]
        },
        {
            'unit': 'b8',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B8_N_2021_02_18.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B8_A_2021_02_22.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B8_A_2021_02_26.csv'))]
        },
        {
            'unit': 'b10',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B10_N_2021_03_22.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B10_A_2021_03_25.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B10_A_2021_03_30.csv'))]
        },
        {
            'unit': 'b11',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B11_N_2021_04_05.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B11_A_2021_04_08.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B11_A_2021_04_22.csv'))]
        },
        {
            'unit': 'b12',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B12_N_2021_04_27.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B12_A_2021_04_30.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B12_A_2021_05_04.csv'))]
        },
        {
            'unit': 'b15',
            'tests': [pd.read_csv(PurePath(dataFolder, 'B15_N_2021_05_31.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B15_A_2021_06_09.csv')),
                      pd.read_csv(PurePath(dataFolder, 'B15_A_2021_06_15.csv'))]
        },
    ]

    # Convert the list of dictionaries to a DataFrame
    data_all = folderdict2data(test_all)

    del test_all

    # Select data based on model
    if model == "a":
        data = data_all[data_all["Unidade"] < "b"]
    elif model == "b":
        data = data_all[data_all["Unidade"] >= "b"]

    return data
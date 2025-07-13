import pandas as pd
from pathlib import PurePath
import os

class RunInDataLoader:
    _base_folders = [
            {
                'unit': 'a2',
                'tests': ['A2_N_2019_07_09.csv',
                         'A2_A_2019_08_08.csv',
                         'A2_A_2019_08_28.csv']
            },
            {
                'unit': 'a3',
                'tests': ['A3_N_2019_12_04.csv',
                         'A3_A_2019_12_09.csv',
                         'A3_A_2019_12_11.csv']
            },
            {
                'unit': 'a4',
                'tests': ['A4_N_2019_12_16.csv',
                         'A4_A_2019_12_19.csv',
                         'A4_A_2020_01_06.csv']
            },
            {
                'unit': 'a5',
                'tests': ['A5_N_2020_01_22.csv',
                         'A5_A_2020_01_27.csv',
                         'A5_A_2020_01_28.csv']
            },
            {
                'unit': 'b5',
                'tests': ['B5_N_2020_10_16.csv',
                         'B5_A_2020_10_22.csv',
                         'B5_A_2020_10_27.csv']
            },
            {
                'unit': 'b7',
                'tests': ['B7_N_2021_02_05.csv',
                         'B7_A_2021_02_08.csv',
                         'B7_A_2021_02_15.csv']
            },
            {
                'unit': 'b8',
                'tests': ['B8_N_2021_02_18.csv',
                         'B8_A_2021_02_22.csv',
                         'B8_A_2021_02_26.csv']
            },
            {
                'unit': 'b10',
                'tests': ['B10_N_2021_03_22.csv',
                         'B10_A_2021_03_25.csv',
                         'B10_A_2021_03_30.csv']
            },
            {
                'unit': 'b11',
                'tests': ['B11_N_2021_04_05.csv',
                         'B11_A_2021_04_08.csv',
                         'B11_A_2021_04_22.csv']
            },
            {
                'unit': 'b12',
                'tests': ['B12_N_2021_04_27.csv',
                         'B12_A_2021_04_30.csv',
                         'B12_A_2021_05_04.csv']
            },
            {
                'unit': 'b15',
                'tests': ['B15_N_2021_05_31.csv',
                         'B15_A_2021_06_09.csv',
                         'B15_A_2021_06_15.csv']
            },
        ]
    
    def __init__(self, dict_folder=None, model=None, features=None):
        # TODO: write docstring

        if dict_folder is None:
            baseFolder = PurePath(os.getcwd(),'LabeledData')
            if model is None:
                raise ValueError("Model must be specified if dict_folder is not provided.")
            elif model == "all":
                test_all = self._base_folders
                self.dict_folder = [{"unit": unit["unit"], "tests": [PurePath(baseFolder, test) for test in unit['tests']]}
                                    for unit in test_all]
            else:
                test_all = [unit for unit in self._base_folders if unit['unit'][0] == model]
                self.dict_folder = [{"unit": unit['unit'], "tests": [PurePath(baseFolder, test) for test in unit['tests']]}
                                    for unit in test_all]
        else:
            self.dict_folder = dict_folder

        self.features = features
        self.data = pd.DataFrame()

    def load_data(self):
        # TODO: write docstring

        if self.data.empty:
            data_total = [{"unit": unit['unit'], "tests": [pd.read_csv(test) for test in unit['tests']]} for unit in self.dict_folder]
            data_total = self._joinTests(data_total)
            if self.features is not None:
                # Filter the data to only include the specified features
                data_total = data_total[self.features + ['Tempo', 'Unidade', 'N_ensaio']]

            self.data = data_total

        return self.data
    
    def clear_data(self):
        self.data = pd.DataFrame()

    def reload_data(self):
        self.clear_data()
        return self.load_data()

    def _joinTests(self,test_all:list[dict]):
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
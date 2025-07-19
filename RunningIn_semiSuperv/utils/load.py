import pandas as pd
from pathlib import PurePath
import os
from typing import Dict, List, Optional, Union, Any

class RunInDataLoader:
    """
    A data loader class for loading and managing running-in test data from CSV files.
    
    This class provides functionality to load CSV files containing test data from 
    different units and tests, combining them into a single DataFrame with proper
    unit and test identification. It supports filtering by model type and specific 
    features.
    
    Attributes:
        _base_folders (list): Class-level configuration defining the mapping between
            units and their corresponding test files.
        dict_folder (list): Instance-level list of dictionaries containing unit names
            and their associated test file paths.
        features (list, optional): List of feature column names to filter the data.
        data (pd.DataFrame): Cached loaded data.
    
    Example:
        >>> # Load all data
        >>> loader = RunInDataLoader(model="all")
        >>> data = loader.load_data()
        
        >>> # Load only 'a' series units
        >>> loader = RunInDataLoader(model="a")
        >>> data = loader.load_data()
        
        >>> # Load with specific features
        >>> loader = RunInDataLoader(model="all", features=['PressaoDescarga', 'CorrenteRMS'])
        >>> data = loader.load_data()
        
        >>> # Load custom data files
        >>> custom_dict = [{'unit': 'test', 'tests': ['/path/to/test.csv']}]
        >>> loader = RunInDataLoader(dict_folder=custom_dict)
        >>> data = loader.load_data()
    """
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
    
    def __init__(self, 
                 dict_folder: Optional[List[Dict[str, Union[str, List[str]]]]] = None, 
                 model: Optional[str] = None, 
                 features: Optional[List[str]] = None) -> None:
        """
        Initialize the RunInDataLoader.
        
        Args:
            dict_folder (list, optional): Custom list of dictionaries containing unit
                information and test file paths. Each dictionary should have 'unit' 
                and 'tests' keys. If None, uses predefined _base_folders with model 
                parameter. Defaults to None.
            model (str, optional): Model type to filter units. Valid options:
                - "all": Load all available units
                - "a": Load only units starting with 'a' (a2, a3, a4, a5)
                - "b": Load only units starting with 'b' (b5, b7, b8, b10, b11, b12, b15)
                Required if dict_folder is None. Defaults to None.
            features (list, optional): List of feature column names to include in
                the loaded data. If None, all columns are included. The columns
                'Tempo', 'Unidade', and 'N_ensaio' are always included regardless
                of this parameter. Defaults to None.
                
        Raises:
            ValueError: If both dict_folder and model are None.
            
        Example:
            >>> # Initialize with model filtering
            >>> loader = RunInDataLoader(model="a", features=['PressaoDescarga'])
            
            >>> # Initialize with custom folder structure
            >>> custom_folders = [
            ...     {'unit': 'test1', 'tests': ['/path/to/test1.csv']}
            ... ]
            >>> loader = RunInDataLoader(dict_folder=custom_folders)
        """

        if dict_folder is None:
            base_folder = PurePath(os.getcwd(), 'LabeledData')
            if model is None:
                raise ValueError("Model must be specified if dict_folder is not provided.")
            elif model == "all":
                test_all = self._base_folders
                self.dict_folder = [{"unit": unit["unit"], "tests": [PurePath(base_folder, test) for test in unit['tests']]}
                                    for unit in test_all]
            else:
                test_all = [unit for unit in self._base_folders if unit['unit'][0] == model]
                self.dict_folder = [{"unit": unit['unit'], "tests": [PurePath(base_folder, test) for test in unit['tests']]}
                                    for unit in test_all]
        else:
            self.dict_folder = dict_folder

        self.features = features
        self.data = pd.DataFrame()

    def load_data(self) -> pd.DataFrame:
        """
        Load and return the test data from CSV files.
        
        This method loads CSV files specified in dict_folder, combines them into a 
        single DataFrame, and adds unit identification and test numbering. The data
        is cached after the first load to improve performance on subsequent calls.
        
        The method automatically adds two columns:
        - 'Unidade': Unit identifier for each row
        - 'N_ensaio': Test number (0-indexed) for each test within a unit
        
        If features were specified during initialization, the returned DataFrame
        will only contain those features plus 'Tempo', 'Unidade', and 'N_ensaio'.
        
        Returns:
            pd.DataFrame: Combined DataFrame containing all test data with added
                unit and test identification columns. Columns include:
                - All original CSV columns (or filtered features if specified)
                - 'Unidade': Unit identifier (str)
                - 'N_ensaio': Test number within unit (int, 0-indexed)
                
        Raises:
            FileNotFoundError: If any of the specified CSV files cannot be found.
            pd.errors.EmptyDataError: If any CSV file is empty.
            pd.errors.ParserError: If any CSV file cannot be parsed.
            
        Example:
            >>> loader = RunInDataLoader(model="a")
            >>> data = loader.load_data()
            >>> print(data.columns)
            Index(['Tempo', 'PressaoDescarga', ..., 'Unidade', 'N_ensaio'], dtype='object')
            >>> print(data['Unidade'].unique())
            ['a2' 'a3' 'a4' 'a5']
        """

        if self.data.empty:
            data_total = [{"unit": unit['unit'], "tests": [pd.read_csv(test) for test in unit['tests']]} for unit in self.dict_folder]
            data_total = self._joinTests(data_total)
            if self.features is not None:
                # Filter the data to only include the specified features
                data_total = data_total[self.features + ['Tempo', 'Unidade', 'N_ensaio']]

            self.data = data_total

        return self.data
    
    def clear_data(self) -> None:
        """
        Clear the cached data from memory.
        
        This method resets the internal data cache to an empty DataFrame, forcing
        the next call to load_data() to reload the data from CSV files. Useful
        for memory management or when you want to ensure fresh data loading.
        
        Example:
            >>> loader = RunInDataLoader(model="all")
            >>> data = loader.load_data()  # Data is loaded and cached
            >>> loader.clear_data()        # Cache is cleared
            >>> data = loader.load_data()  # Data is reloaded from files
        """
        self.data = pd.DataFrame()

    def reload_data(self) -> pd.DataFrame:
        """
        Clear cached data and reload it from CSV files.
        
        This method is a convenience function that combines clear_data() and 
        load_data() operations. It ensures that the data is freshly loaded
        from the CSV files, bypassing any cached data.
        
        Returns:
            pd.DataFrame: Freshly loaded DataFrame containing all test data with
                unit and test identification columns.
                
        Example:
            >>> loader = RunInDataLoader(model="all")
            >>> data1 = loader.load_data()   # Initial load
            >>> data2 = loader.reload_data() # Clear cache and reload
            >>> # data1 and data2 contain the same data but are different objects
        """
        self.clear_data()
        return self.load_data()

    def _joinTests(self, test_all: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Join multiple test DataFrames with proper unit and test identification.
        
        This private method processes a list of dictionaries containing unit
        information and their associated test DataFrames, combining them into
        a single DataFrame with added identification columns.
        
        Args:
            test_all (list[dict]): List of dictionaries where each dictionary
                contains:
                - 'unit' (str): Unit identifier
                - 'tests' (list[pd.DataFrame]): List of DataFrames for that unit
                
        Returns:
            pd.DataFrame: Combined DataFrame with all tests from all units,
                including added columns:
                - 'N_ensaio': Test number within each unit (0-indexed)
                - 'Unidade': Unit identifier for each row
                
        Note:
            This is a private method used internally by load_data(). The method
            modifies the input DataFrames by adding the 'N_ensaio' column.
            
        Example:
            >>> test_data = [
            ...     {'unit': 'a2', 'tests': [df1, df2]},
            ...     {'unit': 'a3', 'tests': [df3]}
            ... ]
            >>> combined_df = loader._joinTests(test_data)
            >>> # Result has 'N_ensaio' (0,1 for a2; 0 for a3) and 'Unidade' columns
        """

        data_total = pd.DataFrame()

        for unit in test_all:  # Iterate through each unit
            data_unit = pd.DataFrame()
            for k, test in enumerate(unit['tests']):  # Iterate through each test
                test['N_ensaio'] = k  # Add the test number
                data_unit = pd.concat([data_unit, test])
            data_unit['Unidade'] = unit['unit']  # Add the unit name
            data_total = pd.concat([data_total, data_unit])

        return data_total
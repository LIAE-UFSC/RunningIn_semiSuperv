import unittest
import pandas as pd
import os
import tempfile
import shutil
from pathlib import PurePath
from unittest.mock import patch
import numpy as np

from RunningIn_semiSuperv.utils.load import RunInDataLoader


class TestRunInDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.labeled_data_dir = os.path.join(self.test_dir, 'LabeledData')
        os.makedirs(self.labeled_data_dir)
        
        # Create sample CSV files for testing
        self.sample_data = pd.DataFrame({
            'Tempo': [1.0, 2.0, 3.0],
            'PressaoDescarga': [1.1, 2.1, 3.1],
            'PressaoSuccao': [1.2, 2.2, 3.2],
            'CorrenteRMS': [1.3, 2.3, 3.3],
            'Amaciado': [0, 0, 1]
        })
        
        # Create test CSV files
        self.test_files = [
            'A2_N_2019_07_09.csv',
            'A2_A_2019_08_08.csv',
            'A3_N_2019_12_04.csv'
        ]
        
        for file in self.test_files:
            file_path = os.path.join(self.labeled_data_dir, file)
            self.sample_data.to_csv(file_path, index=False)
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)
    
    @patch('os.getcwd')
    def test_init_with_model_all(self, mock_getcwd):
        """Test initialization with model='all'."""
        mock_getcwd.return_value = self.test_dir
        
        loader = RunInDataLoader(model="all")
        
        # Should have all units
        self.assertEqual(len(loader.dict_folder), 11)  # All units in _base_folders
        self.assertIsInstance(loader.dict_folder, list)
        self.assertEqual(loader.features, None)
        self.assertTrue(loader.data.empty)
    
    @patch('os.getcwd')
    def test_init_with_model_a(self, mock_getcwd):
        """Test initialization with model='a' (should get units starting with 'a')."""
        mock_getcwd.return_value = self.test_dir
        
        loader = RunInDataLoader(model="a")
        
        # Should only have units starting with 'a'
        self.assertEqual(len(loader.dict_folder), 4)  # a2, a3, a4, a5
        for unit in loader.dict_folder:
            self.assertTrue(unit['unit'].startswith('a'))
    
    @patch('os.getcwd')
    def test_init_with_model_b(self, mock_getcwd):
        """Test initialization with model='b' (should get units starting with 'b')."""
        mock_getcwd.return_value = self.test_dir
        
        loader = RunInDataLoader(model="b")
        
        # Should only have units starting with 'b'
        self.assertEqual(len(loader.dict_folder), 7)  # b5, b7, b8, b10, b11, b12, b15
        for unit in loader.dict_folder:
            self.assertTrue(unit['unit'].startswith('b'))
    
    def test_init_with_dict_folder(self):
        """Test initialization with custom dict_folder."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        
        self.assertEqual(loader.dict_folder, custom_dict)
        self.assertEqual(loader.features, None)
        self.assertTrue(loader.data.empty)
    
    def test_init_with_features(self):
        """Test initialization with specific features."""
        features = ['PressaoDescarga', 'CorrenteRMS']
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict, features=features)
        
        self.assertEqual(loader.features, features)
    
    def test_init_no_model_no_dict_folder_raises_error(self):
        """Test that ValueError is raised when neither model nor dict_folder is provided."""
        with self.assertRaises(ValueError) as context:
            RunInDataLoader()
        
        self.assertEqual(str(context.exception), "Model must be specified if dict_folder is not provided.")
    
    def test_load_data_basic(self):
        """Test basic data loading functionality."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        data = loader.load_data()
        
        # Check that data is loaded
        self.assertFalse(data.empty)
        self.assertIn('Unidade', data.columns)
        self.assertIn('N_ensaio', data.columns)
        self.assertIn('Tempo', data.columns)
        
        # Check that unit and test number are added correctly
        self.assertTrue(all(data['Unidade'] == 'test_unit'))
        self.assertTrue(all(data['N_ensaio'] == 0))
    
    def test_load_data_multiple_tests(self):
        """Test loading data with multiple tests per unit."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [
                    PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv'),
                    PurePath(self.labeled_data_dir, 'A2_A_2019_08_08.csv')
                ]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        data = loader.load_data()
        
        # Should have data from both tests
        self.assertEqual(len(data), 6)  # 3 rows per file * 2 files
        
        # Check test numbers
        test_numbers = data['N_ensaio'].unique()
        self.assertEqual(len(test_numbers), 2)
        self.assertIn(0, test_numbers)
        self.assertIn(1, test_numbers)
    
    def test_load_data_with_features(self):
        """Test loading data with specific features filter."""
        features = ['PressaoDescarga', 'CorrenteRMS']
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict, features=features)
        data = loader.load_data()
        
        # Should only have specified features plus required columns
        expected_columns = features + ['Tempo', 'Unidade', 'N_ensaio']
        self.assertEqual(set(data.columns), set(expected_columns))
    
    def test_load_data_caching(self):
        """Test that data is cached and not reloaded on subsequent calls."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        
        # First load
        data1 = loader.load_data()
        
        # Second load should return the same cached data
        data2 = loader.load_data()
        
        # Should be the same object (cached)
        self.assertIs(data1, data2)
    
    def test_clear_data(self):
        """Test clearing loaded data."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        
        # Load data
        loader.load_data()
        self.assertFalse(loader.data.empty)
        
        # Clear data
        loader.clear_data()
        self.assertTrue(loader.data.empty)
    
    def test_reload_data(self):
        """Test reloading data (clear and load again)."""
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'A2_N_2019_07_09.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        
        # Load data first time
        data1 = loader.load_data()
        
        # Reload data
        data2 = loader.reload_data()
        
        # Should not be the same object (reloaded)
        self.assertIsNot(data1, data2)
        # But should have the same content
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_join_tests_single_unit_single_test(self):
        """Test _joinTests method with single unit and single test."""
        loader = RunInDataLoader(dict_folder=[])
        
        test_data = [{
            'unit': 'test_unit',
            'tests': [self.sample_data.copy()]
        }]
        
        result = loader._joinTests(test_data)
        
        self.assertEqual(len(result), 3)
        self.assertIn('Unidade', result.columns)
        self.assertIn('N_ensaio', result.columns)
        self.assertTrue(all(result['Unidade'] == 'test_unit'))
        self.assertTrue(all(result['N_ensaio'] == 0))
    
    def test_join_tests_single_unit_multiple_tests(self):
        """Test _joinTests method with single unit and multiple tests."""
        loader = RunInDataLoader(dict_folder=[])
        
        test_data = [{
            'unit': 'test_unit',
            'tests': [self.sample_data.copy(), self.sample_data.copy()]
        }]
        
        result = loader._joinTests(test_data)
        
        self.assertEqual(len(result), 6)  # 3 rows * 2 tests
        test_numbers = result['N_ensaio'].unique()
        self.assertEqual(len(test_numbers), 2)
        self.assertIn(0, test_numbers)
        self.assertIn(1, test_numbers)
    
    def test_join_tests_multiple_units(self):
        """Test _joinTests method with multiple units."""
        loader = RunInDataLoader(dict_folder=[])
        
        test_data = [
            {
                'unit': 'unit1',
                'tests': [self.sample_data.copy()]
            },
            {
                'unit': 'unit2',
                'tests': [self.sample_data.copy()]
            }
        ]
        
        result = loader._joinTests(test_data)
        
        self.assertEqual(len(result), 6)  # 3 rows * 2 units
        units = result['Unidade'].unique()
        self.assertEqual(len(units), 2)
        self.assertIn('unit1', units)
        self.assertIn('unit2', units)
    
    def test_base_folders_structure(self):
        """Test that _base_folders has the expected structure."""
        # Check that all expected units are present
        expected_units = ['a2', 'a3', 'a4', 'a5', 'b5', 'b7', 'b8', 'b10', 'b11', 'b12', 'b15']
        actual_units = [unit['unit'] for unit in RunInDataLoader._base_folders]
        
        self.assertEqual(set(actual_units), set(expected_units))
        
        # Check that each unit has the expected structure
        for unit in RunInDataLoader._base_folders:
            self.assertIn('unit', unit)
            self.assertIn('tests', unit)
            self.assertIsInstance(unit['tests'], list)
            self.assertTrue(len(unit['tests']) > 0)
    
    @patch('pandas.read_csv')
    def test_error_handling_file_not_found(self, mock_read_csv):
        """Test error handling when CSV file is not found."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        custom_dict = [
            {
                'unit': 'test_unit',
                'tests': [PurePath(self.labeled_data_dir, 'nonexistent.csv')]
            }
        ]
        
        loader = RunInDataLoader(dict_folder=custom_dict)
        
        with self.assertRaises(FileNotFoundError):
            loader.load_data()
    
    def test_mock_csv_creation_loading_and_cleanup(self):
        """Test creating a mock CSV file, loading it with RunInDataLoader, validating content, and cleanup."""
        # Create a more comprehensive mock dataset
        mock_data = pd.DataFrame({
            'Tempo': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'PressaoDescarga': [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4],
            'PressaoSuccao': [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
            'CorrenteRMS': [0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024],
            'CorrenteCurtose': [4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4],
            'CorrenteAssimetria': [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
            'VibracaoCalotaInferiorRMS': [0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015],
            'VibracaoCalotaSuperiorRMS': [0.0008, 0.0009, 0.0010, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017],
            'Vazao': [0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34],
            'Amaciado': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        })
        
        # Create a temporary directory for this specific test
        temp_dir = tempfile.mkdtemp()
        temp_labeled_dir = os.path.join(temp_dir, 'LabeledData')
        os.makedirs(temp_labeled_dir)
        
        try:
            # Create the mock CSV file
            mock_filename = 'MOCK_TEST_2024_01_01.csv'
            mock_filepath = os.path.join(temp_labeled_dir, mock_filename)
            
            # Save mock data to CSV
            mock_data.to_csv(mock_filepath, index=False)
            
            # Verify the file was created
            self.assertTrue(os.path.exists(mock_filepath), "Mock CSV file should be created")
            
            # Create a custom dict_folder for the test
            custom_dict = [
                {
                    'unit': 'mock_unit',
                    'tests': [PurePath(mock_filepath)]
                }
            ]
            
            # Initialize RunInDataLoader with the mock data
            loader = RunInDataLoader(dict_folder=custom_dict)
            
            # Load the data
            loaded_data = loader.load_data()
            
            # Validate the loaded content
            self.assertFalse(loaded_data.empty, "Loaded data should not be empty")
            
            # Check that all original columns are present
            original_columns = set(mock_data.columns)
            loaded_columns = set(loaded_data.columns)
            self.assertTrue(original_columns.issubset(loaded_columns), 
                          "All original columns should be present in loaded data")
            
            # Check that additional columns were added
            self.assertIn('Unidade', loaded_data.columns, "Unidade column should be added")
            self.assertIn('N_ensaio', loaded_data.columns, "N_ensaio column should be added")
            
            # Validate data integrity
            self.assertEqual(len(loaded_data), len(mock_data), 
                           "Loaded data should have same number of rows as original")
            
            # Check unit assignment
            self.assertTrue(all(loaded_data['Unidade'] == 'mock_unit'), 
                          "All rows should have correct unit assignment")
            
            # Check test number assignment
            self.assertTrue(all(loaded_data['N_ensaio'] == 0), 
                          "All rows should have test number 0 (first test)")
            
            # Validate specific data values (check a few key columns)
            np.testing.assert_array_almost_equal(
                np.array(loaded_data['Tempo']), 
                np.array(mock_data['Tempo']),
                decimal=6,
                err_msg="Tempo values should match original data"
            )
            
            np.testing.assert_array_almost_equal(
                np.array(loaded_data['PressaoDescarga']), 
                np.array(mock_data['PressaoDescarga']),
                decimal=6,
                err_msg="PressaoDescarga values should match original data"
            )
            
            np.testing.assert_array_equal(
                np.array(loaded_data['Amaciado']), 
                np.array(mock_data['Amaciado']),
                err_msg="Amaciado values should match original data"
            )
            
            # Test feature filtering
            selected_features = ['PressaoDescarga', 'CorrenteRMS', 'Vazao']
            loader_with_features = RunInDataLoader(
                dict_folder=custom_dict, 
                features=selected_features
            )
            
            filtered_data = loader_with_features.load_data()
            expected_columns = selected_features + ['Tempo', 'Unidade', 'N_ensaio']
            self.assertEqual(set(filtered_data.columns), set(expected_columns),
                           "Filtered data should only contain selected features plus required columns")
            
            # Test data reload functionality
            reloaded_data = loader.reload_data()
            pd.testing.assert_frame_equal(loaded_data, reloaded_data,
                                        check_dtype=False)
            
            # Verify file still exists before cleanup
            self.assertTrue(os.path.exists(mock_filepath), "Mock CSV file should still exist")
            
        finally:
            # Clean up: remove the temporary directory and all its contents
            shutil.rmtree(temp_dir)
            
            # Verify cleanup was successful
            self.assertFalse(os.path.exists(mock_filepath), "Mock CSV file should be deleted after cleanup")
            self.assertFalse(os.path.exists(temp_dir), "Temporary directory should be deleted after cleanup")


if __name__ == '__main__':
    unittest.main()
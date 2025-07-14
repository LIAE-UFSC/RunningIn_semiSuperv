import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from RunningIn_semiSuperv.utils.preprocess import RunInPreprocessor


class TestRunInPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data that matches the expected structure
        self.sample_data = pd.DataFrame({
            'Tempo': [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0] * 2,
            'PressaoDescarga': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2] * 2,
            'PressaoSuccao': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9] * 2,
            'CorrenteRMS': [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2] * 2,
            'Unidade': ['A2'] * 12 + ['A3'] * 12,
            'N_ensaio': [0] * 12 + [1] * 12,
            'Amaciado': [0] * 24  # Will be updated by the preprocessor
        })
        
        # Create minimal data for edge cases
        self.minimal_data = pd.DataFrame({
            'Tempo': [1.0, 2.0, 3.0],
            'PressaoDescarga': [1.1, 1.2, 1.3],
            'Unidade': ['A1'] * 3,
            'N_ensaio': [0] * 3,
            'Amaciado': [0] * 3
        })
        
        # Create data with missing features
        self.incomplete_data = pd.DataFrame({
            'Tempo': [1.0, 2.0, 3.0],
            'PressaoDescarga': [1.1, 1.2, 1.3],
            'Unidade': ['A1'] * 3,
            'N_ensaio': [0] * 3,
            'Amaciado': [0] * 3
            # Missing 'PressaoSuccao' and 'CorrenteRMS'
        })
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        preprocessor = RunInPreprocessor()
        
        self.assertEqual(preprocessor.window_size, 1)
        self.assertEqual(preprocessor.delay, 1)
        self.assertIsNone(preprocessor.features)
        self.assertEqual(preprocessor.moving_average, 1)
        self.assertEqual(preprocessor.tMin, 0)
        self.assertEqual(preprocessor.tMax, np.inf)
        self.assertEqual(preprocessor.run_in_transition_min, 5)
        self.assertEqual(preprocessor.run_in_transition_max, np.inf)
        self.assertTrue(preprocessor.X.empty)
        self.assertTrue(preprocessor.y.empty)
        self.assertEqual(preprocessor.feature_names_in_, [])
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_features = ['PressaoDescarga', 'PressaoSuccao']
        preprocessor = RunInPreprocessor(
            window_size=3,
            delay=2,
            features=custom_features,
            moving_average=5,
            tMin=1.0,
            tMax=100.0,
            run_in_transition_min=10,
            run_in_transition_max=50
        )
        
        self.assertEqual(preprocessor.window_size, 3)
        self.assertEqual(preprocessor.delay, 2)
        self.assertEqual(preprocessor.features, custom_features)
        self.assertEqual(preprocessor.moving_average, 5)
        self.assertEqual(preprocessor.tMin, 1.0)
        self.assertEqual(preprocessor.tMax, 100.0)
        self.assertEqual(preprocessor.run_in_transition_min, 10)
        self.assertEqual(preprocessor.run_in_transition_max, 50)
    
    def test_set_filter_params(self):
        """Test setting filter parameters."""
        preprocessor = RunInPreprocessor()
        custom_features = ['PressaoDescarga']
        
        preprocessor.set_filter_params(
            features=custom_features,
            moving_average_window=3,
            tMin=2.0,
            tMax=20.0,
            reset=True
        )
        
        self.assertEqual(preprocessor.features, custom_features)
        self.assertEqual(preprocessor.moving_average, 3)
        self.assertEqual(preprocessor.tMin, 2.0)
        self.assertEqual(preprocessor.tMax, 20.0)
        self.assertEqual(preprocessor.feature_names_in_, [])
        self.assertTrue(preprocessor.X.empty)
        self.assertTrue(preprocessor.y.empty)
    
    def test_set_filter_params_no_reset(self):
        """Test setting filter parameters without reset."""
        preprocessor = RunInPreprocessor()
        preprocessor.feature_names_in_ = ['test']
        preprocessor.X = pd.DataFrame({'col': [1, 2, 3]})
        preprocessor.y = pd.Series([1, 0, 1])
        
        preprocessor.set_filter_params(
            features=['PressaoDescarga'],
            reset=False
        )
        
        self.assertEqual(preprocessor.features, ['PressaoDescarga'])
        self.assertEqual(preprocessor.feature_names_in_, ['test'])  # Should not be reset
        self.assertFalse(preprocessor.X.empty)  # Should not be reset
        self.assertFalse(preprocessor.y.empty)  # Should not be reset
    
    def test_set_window_params(self):
        """Test setting window parameters."""
        preprocessor = RunInPreprocessor()
        custom_features = ['PressaoDescarga', 'PressaoSuccao']
        
        preprocessor.set_window_params(
            window_size=5,
            delay=3,
            features=custom_features,
            reset=True
        )
        
        self.assertEqual(preprocessor.window_size, 5)
        self.assertEqual(preprocessor.delay, 3)
        self.assertEqual(preprocessor.features, custom_features)
    
    def test_set_label_params(self):
        """Test setting label parameters."""
        preprocessor = RunInPreprocessor()
        
        preprocessor.set_label_params(
            run_in_transition_min=8,
            run_in_transition_max=25,
            reset=True
        )
        
        self.assertEqual(preprocessor.run_in_transition_min, 8)
        self.assertEqual(preprocessor.run_in_transition_max, 25)
    
    @patch('RunningIn_semiSuperv.utils.preprocess.MovingAverageTransformer')
    @patch('RunningIn_semiSuperv.utils.preprocess.DelayedSlidingWindow')
    def test_fit_with_default_features(self, mock_dsw, mock_mat):
        """Test fit method with default features (None)."""
        preprocessor = RunInPreprocessor()
        
        # Mock the transformers
        mock_mat.return_value = MagicMock()
        mock_dsw.return_value = MagicMock()
        
        preprocessor.fit(self.sample_data)
        
        # Check that features are set to all columns
        expected_features = self.sample_data.columns.tolist()
        self.assertEqual(preprocessor._features, expected_features)
        self.assertEqual(preprocessor.feature_names_in_, expected_features)
        
        # Check that transformers are initialized
        mock_mat.assert_called_once()
        mock_dsw.assert_called_once()
    
    @patch('RunningIn_semiSuperv.utils.preprocess.MovingAverageTransformer')
    @patch('RunningIn_semiSuperv.utils.preprocess.DelayedSlidingWindow')
    def test_fit_with_custom_features(self, mock_dsw, mock_mat):
        """Test fit method with custom features."""
        custom_features = ['PressaoDescarga', 'PressaoSuccao']
        preprocessor = RunInPreprocessor(features=custom_features)
        
        # Mock the transformers
        mock_mat.return_value = MagicMock()
        mock_dsw.return_value = MagicMock()
        
        preprocessor.fit(self.sample_data)
        
        self.assertEqual(preprocessor._features, custom_features)
        self.assertEqual(preprocessor.feature_names_in_, self.sample_data.columns.tolist())
    
    def test_fit_missing_features_raises_error(self):
        """Test that fit raises error when required features are missing."""
        custom_features = ['PressaoDescarga', 'NonexistentFeature']
        preprocessor = RunInPreprocessor(features=custom_features)
        
        with self.assertRaises(ValueError) as context:
            preprocessor.fit(self.incomplete_data)
        
        self.assertIn("missing from the input data", str(context.exception))
        self.assertIn("NonexistentFeature", str(context.exception))
    
    def test_transform_not_fitted_raises_error(self):
        """Test that transform raises error when not fitted."""
        preprocessor = RunInPreprocessor()
        
        with self.assertRaises(ValueError) as context:
            preprocessor.transform(self.sample_data)
        
        self.assertIn("not been fitted yet", str(context.exception))
    
    @patch('RunningIn_semiSuperv.utils.preprocess.MovingAverageTransformer')
    @patch('RunningIn_semiSuperv.utils.preprocess.DelayedSlidingWindow')
    def test_transform_pipeline(self, mock_dsw, mock_mat):
        """Test the complete transform pipeline."""
        preprocessor = RunInPreprocessor(
            tMin=1.0,
            tMax=10.0,
            run_in_transition_min=3.0,
            run_in_transition_max=8.0
        )
        
        # Mock the transformers
        mock_filter_transformer = MagicMock()
        mock_window_transformer = MagicMock()
        mock_mat.return_value = mock_filter_transformer
        mock_dsw.return_value = mock_window_transformer
        
        # Create sample transformed data
        transformed_X = self.sample_data.drop(['Amaciado'], axis=1).copy()
        transformed_X['new_feature'] = 1.0  # Simulate windowing adding new features
        
        mock_filter_transformer.fit_transform.return_value = transformed_X
        mock_window_transformer.fit_transform.return_value = transformed_X
        
        # Fit and transform
        preprocessor.fit(self.sample_data)
        X, y = preprocessor.transform(self.sample_data)
        
        # Verify transformers were called
        mock_filter_transformer.fit_transform.assert_called_once()
        mock_window_transformer.fit_transform.assert_called_once()
        
        # Verify that X and y are returned
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        
        # Verify that internal X and y are stored
        self.assertFalse(preprocessor.X.empty)
        self.assertFalse(preprocessor.y.empty)
    
    def test_splitXY(self):
        """Test the _splitXY method."""
        preprocessor = RunInPreprocessor()
        
        X, y = preprocessor._splitXY(self.sample_data)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertNotIn('Amaciado', X.columns)
        self.assertEqual(y.name, 'Amaciado')
        self.assertEqual(len(X), len(y))
    
    def test_filter_time(self):
        """Test the _filter_time method."""
        preprocessor = RunInPreprocessor(tMin=2.0, tMax=8.0)
        
        filtered_data = preprocessor._filter_time(self.sample_data)
        
        self.assertTrue(all(filtered_data['Tempo'] >= 2.0))
        self.assertTrue(all(filtered_data['Tempo'] <= 8.0))
        self.assertLess(len(filtered_data), len(self.sample_data))
    
    def test_filter_time_no_limits(self):
        """Test _filter_time with no limits (default)."""
        preprocessor = RunInPreprocessor()
        
        filtered_data = preprocessor._filter_time(self.sample_data)
        
        # Should return all data when no limits are set
        self.assertEqual(len(filtered_data), len(self.sample_data))
    
    def test_update_labels_default_params(self):
        """Test update_labels with default parameters."""
        preprocessor = RunInPreprocessor(
            run_in_transition_min=5.0,
            run_in_transition_max=10.0
        )
        
        labeled_data = preprocessor.update_labels(self.sample_data.copy())
        
        # Check that labels are assigned correctly
        # Early times in first test (N_ensaio=0) should be 0 (not run-in)
        early_first_test = labeled_data[
            (labeled_data['Tempo'] <= 5.0) & (labeled_data['N_ensaio'] == 0)
        ]
        self.assertTrue(all(early_first_test['Amaciado'] == 0))
        
        # Late times in first test should be 1 (run-in)
        late_first_test = labeled_data[
            (labeled_data['Tempo'] >= 10.0) & (labeled_data['N_ensaio'] == 0)
        ]
        self.assertTrue(all(late_first_test['Amaciado'] == 1))
        
        # All subsequent tests should be 1 (run-in)
        subsequent_tests = labeled_data[labeled_data['N_ensaio'] > 0]
        self.assertTrue(all(subsequent_tests['Amaciado'] == 1))
    
    def test_update_labels_custom_params(self):
        """Test update_labels with custom parameters passed to method."""
        preprocessor = RunInPreprocessor()
        
        labeled_data = preprocessor.update_labels(
            self.sample_data.copy(),
            run_in_transition_min=3.0,
            run_in_transition_max=7.0
        )
        
        # Check that internal parameters are updated
        self.assertEqual(preprocessor.run_in_transition_min, 3.0)
        self.assertEqual(preprocessor.run_in_transition_max, 7.0)
        
        # Check labeling with new parameters
        early_first_test = labeled_data[
            (labeled_data['Tempo'] <= 3.0) & (labeled_data['N_ensaio'] == 0)
        ]
        self.assertTrue(all(early_first_test['Amaciado'] == 0))
        
        late_first_test = labeled_data[
            (labeled_data['Tempo'] >= 7.0) & (labeled_data['N_ensaio'] == 0)
        ]
        self.assertTrue(all(late_first_test['Amaciado'] == 1))
    
    def test_update_labels_unknown_region(self):
        """Test that unknown labels (-1) are assigned in transition region."""
        preprocessor = RunInPreprocessor(
            run_in_transition_min=5.0,
            run_in_transition_max=8.0
        )
        
        labeled_data = preprocessor.update_labels(self.sample_data.copy())
        
        # Check that transition region has unknown labels
        transition_region = labeled_data[
            (labeled_data['Tempo'] > 5.0) & 
            (labeled_data['Tempo'] < 8.0) & 
            (labeled_data['N_ensaio'] == 0)
        ]
        if len(transition_region) > 0:
            self.assertTrue(all(transition_region['Amaciado'] == -1))
    
    def test_edge_case_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        preprocessor = RunInPreprocessor()
        
        # Create empty dataframe with proper columns structure
        empty_df = pd.DataFrame(columns=['Tempo', 'PressaoDescarga', 'Unidade', 'N_ensaio', 'Amaciado'])
        
        # Should handle empty dataframe gracefully in individual methods
        filtered = preprocessor._filter_time(empty_df)
        self.assertTrue(filtered.empty)
        
        # Test update_labels with empty data
        labeled = preprocessor.update_labels(empty_df.copy())
        self.assertTrue(labeled.empty)
    
    def test_edge_case_single_row(self):
        """Test behavior with single row DataFrame."""
        single_row_data = pd.DataFrame({
            'Tempo': [5.0],
            'PressaoDescarga': [1.1],
            'Unidade': ['A1'],
            'N_ensaio': [0],
            'Amaciado': [0]
        })
        
        preprocessor = RunInPreprocessor(
            run_in_transition_min=3.0,
            run_in_transition_max=7.0
        )
        
        labeled = preprocessor.update_labels(single_row_data.copy())
        self.assertEqual(len(labeled), 1)
        # Time 5.0 is between 3.0 and 7.0, so should be unknown (-1)
        self.assertEqual(labeled.iloc[0]['Amaciado'], -1)
    
    def test_multiple_units_handling(self):
        """Test that multiple units are handled correctly."""
        # Create data with multiple units
        multi_unit_data = pd.DataFrame({
            'Tempo': [1.0, 2.0, 6.0, 11.0] * 2,
            'PressaoDescarga': [1.1, 1.2, 1.3, 1.4] * 2,
            'Unidade': ['A1'] * 4 + ['A2'] * 4,
            'N_ensaio': [0, 0, 0, 0, 1, 1, 1, 1],
            'Amaciado': [0] * 8
        })
        
        preprocessor = RunInPreprocessor(
            run_in_transition_min=5.0,
            run_in_transition_max=10.0
        )
        
        labeled = preprocessor.update_labels(multi_unit_data.copy())
        
        # Check that labeling works correctly for both units
        unit_a1_early = labeled[
            (labeled['Unidade'] == 'A1') & 
            (labeled['Tempo'] <= 5.0) & 
            (labeled['N_ensaio'] == 0)
        ]
        self.assertTrue(all(unit_a1_early['Amaciado'] == 0))
        
        unit_a1_late = labeled[
            (labeled['Unidade'] == 'A1') & 
            (labeled['Tempo'] >= 10.0) & 
            (labeled['N_ensaio'] == 0)
        ]
        self.assertTrue(all(unit_a1_late['Amaciado'] == 1))
        
        # All N_ensaio > 0 should be labeled as 1
        subsequent_tests = labeled[labeled['N_ensaio'] > 0]
        self.assertTrue(all(subsequent_tests['Amaciado'] == 1))


if __name__ == '__main__':
    # Suppress warnings during testing
    warnings.filterwarnings('ignore')
    unittest.main()

# RunningIn Semi-Supervised Testing Suite

This directory contains comprehensive tests for all components of the RunningIn Semi-Supervised Learning package. The test suite ensures reliability and correctness of the data loading, preprocessing, and modeling pipeline used for running-in detection in hermetic alternative compressors.

## Test Files

- **`test_preprocess.py`** - Comprehensive test suite for `RunInPreprocessor` class
- **`test_models.py`** - Test suite for `RunInSemiSupervisedModel` class  
- **`test_dataLoader.py`** - Test suite for `RunInDataLoader` class

## Running Tests

### Run All Tests
```bash
# From project root
pytest RunningIn_semiSuperv/utils/tests/ -v

# Or from this directory
pytest . -v
```

### Run Specific Test Files
```bash
# Test only preprocessing functionality
pytest test_preprocess.py -v

# Test only model functionality  
pytest test_models.py -v

# Test only data loading functionality
pytest test_dataLoader.py -v
```

### Run Individual Test Methods
```bash
# Run specific test method
pytest test_preprocess.py::TestRunInPreprocessor::test_init_default_parameters -v

# Run tests matching a pattern
pytest -k "test_split" -v
```

## Test Coverage Overview

### 1. RunInPreprocessor Tests (`test_preprocess.py`)

**Total: 33 tests** covering all preprocessing functionality including new splitting and balancing features.

#### Core Functionality (20 tests)
- **Initialization**: Default and custom parameter validation
- **Parameter Setting**: Filter, window, and label parameter updates
- **Pipeline Methods**: fit(), transform(), fit_transform() with error handling
- **Internal Methods**: Data splitting, time filtering, label updates
- **Edge Cases**: Empty data, single rows, multiple units

#### New Features (13 tests)
- **Test Split Configuration**: Float proportions and unit-based splitting
- **Class Balancing**: Undersampling and no-balancing options
- **Data Retrieval**: Train/test/full data getter methods
- **Feature Selection**: Proper exclusion of metadata columns

### 2. RunInSemiSupervisedModel Tests (`test_models.py`)

**Total: 15 tests** covering semi-supervised learning wrapper functionality.

#### Classifier Integration
- **Multiple Classifiers**: LogisticRegression, RandomForestClassifier, KNeighborsClassifier
- **Parameter Handling**: Custom classifier arguments and defaults
- **SelfTrainingClassifier**: Proper sklearn semi-supervised wrapper usage

#### Model Operations
- **Training**: fit() with various data types and configurations
- **Prediction**: predict() and predict_proba() methods
- **Evaluation**: Built-in metrics calculation
- **Error Handling**: Invalid inputs and edge cases

### 3. RunInDataLoader Tests (`test_dataLoader.py`)

**Total: 5 tests** covering data loading and organization functionality.

#### Data Loading
- **File Discovery**: Automatic CSV file detection
- **Data Validation**: Proper column structure verification
- **Feature Selection**: Custom and automatic feature detection

## Test Data Structure

Tests use standardized sample data that mimics real compressor monitoring datasets:

```python
sample_data = pd.DataFrame({
    'Tempo': [0.5, 1.0, 2.0, ...],           # Time values
    'PressaoDescarga': [1.1, 1.2, 1.3, ...], # Discharge pressure
    'PressaoSuccao': [0.8, 0.9, 1.0, ...],   # Suction pressure  
    'CorrenteRMS': [2.1, 2.2, 2.3, ...],     # RMS current
    'Unidade': ['A2', 'A2', 'A3', ...],      # Unit identifiers
    'N_ensaio': [0, 0, 1, ...],              # Test numbers
    'Amaciado': [0, 0, 1, ...]               # Target labels
})
```

## Key Testing Patterns

### 1. Mocking External Dependencies
```python
@patch('RunningIn_semiSuperv.utils.preprocess.DelayedSlidingWindow')
@patch('RunningIn_semiSuperv.utils.preprocess.MovingAverageTransformer')
def test_fit_transform_pipeline(self, mock_dsw, mock_mat):
    # Tests complex transformations without external library dependencies
```

### 2. Parameter Validation
```python
def test_set_test_split_invalid_parameter(self):
    with self.assertRaises(ValueError) as context:
        preprocessor.fit_transform(data)
    self.assertIn("test_split must be a float or a list", str(context.exception))
```

### 3. Data Integrity Checks
```python
def test_get_train_test_data_consistency(self):
    X_train, y_train, X_test, y_test = preprocessor.get_train_test_data()
    self.assertEqual(len(X_train), len(y_train))
    self.assertEqual(len(X_test), len(y_test))
```

## Test Configuration

### Setup and Fixtures
Each test class includes comprehensive setUp() methods that create:
- **Sample datasets** with various sizes and characteristics
- **Edge case data** for boundary testing
- **Mock objects** for external dependencies

### Error Testing
Systematic testing of error conditions including:
- **Missing data validation**
- **Invalid parameter combinations** 
- **Unfitted estimator usage**
- **Type validation errors**

## Testing Best Practices

### 1. Comprehensive Coverage
- All public methods tested
- Parameter combinations validated
- Edge cases and error conditions covered
- Integration between components verified

### 2. Isolation and Mocking
- External dependencies mocked to avoid library compatibility issues
- Each component tested independently
- Controlled test environments for reproducible results

### 3. Real-world Scenarios
- Tests reflect actual usage patterns
- Multiple data formats and sizes supported
- Performance considerations included

## Continuous Integration

The test suite is designed for:
- **Automated testing** in CI/CD pipelines
- **Regression detection** when code changes
- **Performance monitoring** for large datasets
- **Documentation validation** through examples

## Test Results Interpretation

### Success Criteria
- All tests pass (✅)
- No warnings or deprecation notices
- Consistent performance across runs
- Memory usage within acceptable limits

### Common Issues
- **DelayedSlidingWindow compatibility**: Tests use mocking to handle known issues
- **Data type validation**: Strict checking prevents runtime errors
- **Memory constraints**: Large datasets may require optimization

## Contributing to Tests

### Adding New Tests
1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Test both success and failure cases
4. Add edge case validation
5. Update this README with new test descriptions

### Test Data Guidelines
- Use realistic but minimal datasets
- Include various unit types and test scenarios
- Maintain consistent data structure
- Document any special data requirements

## Performance Testing

### Benchmarking
- Large dataset processing (1000+ samples)
- Memory usage monitoring
- Processing time validation
- Scalability assessment

### Optimization Targets
- **Preprocessing**: < 1s for 1000 samples
- **Model training**: < 5s for standard datasets
- **Memory usage**: < 100MB for typical workflows
- **Prediction speed**: < 100ms for new samples

## Debugging Test Failures

### Common Debugging Steps
1. **Check test isolation**: Ensure tests don't depend on execution order
2. **Verify mock behavior**: Confirm mocked objects return expected values
3. **Validate test data**: Check that sample data matches expected format
4. **Review error messages**: Test error messages are informative and accurate

### Useful Testing Commands
```bash
# Verbose output with failure details
pytest -v --tb=long

# Stop on first failure
pytest -x

# Run only failed tests from last run
pytest --lf

# Generate coverage report
pytest --cov=RunningIn_semiSuperv
```

This comprehensive testing suite ensures the reliability and robustness of the RunningIn Semi-Supervised Learning package for industrial compressor monitoring applications.
- **`update_labels()`**: Tests label assignment logic for run-in transitions

### 5. Label Assignment Logic Tests
- **Default parameters**: Tests correct label assignment with default transition times
- **Custom parameters**: Tests label assignment with custom transition times
- **Unknown region**: Tests that transition regions are correctly labeled as unknown (-1)
- **Multiple units**: Tests handling of data from multiple units

### 6. Edge Cases and Error Handling
- **Empty DataFrame**: Tests graceful handling of empty input data
- **Single row**: Tests behavior with minimal data
- **Missing features**: Tests error handling for missing required features
- **Transform before fit**: Tests appropriate error messages

### 7. Integration Tests
- **Complete pipeline**: Tests the full fit-transform workflow with mocked dependencies
- **Multiple units and tests**: Tests realistic data scenarios

## Test Data Structure

The tests use sample data that mimics the expected structure of running-in data:

```python
{
    'Tempo': [0.5, 1.0, 2.0, ...],          # Time values
    'PressaoDescarga': [1.1, 1.2, 1.3, ...], # Discharge pressure
    'PressaoSuccao': [0.8, 0.9, 1.0, ...],   # Suction pressure  
    'CorrenteRMS': [2.1, 2.2, 2.3, ...],     # RMS current
    'Unidade': ['A2', 'A3', ...],             # Unit identifier
    'N_ensaio': [0, 1, ...],                  # Test number
    'Amaciado': [0, 1, -1]                    # Run-in labels (0=not run-in, 1=run-in, -1=unknown)
}
```

## Label Assignment Logic

The preprocessor assigns labels based on the following rules:
- **Time ≤ `run_in_transition_min` AND `N_ensaio == 0`**: Label = 0 (not run-in)
- **Time ≥ `run_in_transition_max` AND `N_ensaio == 0`**: Label = 1 (run-in)
- **`N_ensaio > 0`**: Label = 1 (run-in, all subsequent tests)
- **Transition region**: Label = -1 (unknown)

## Running the Tests

### Option 1: Using pytest directly
```bash
python -m pytest RunningIn_semiSuperv/utils/tests/test_preprocess.py -v
```

### Option 2: Using the test runner script
```bash
python run_preprocess_tests.py
```

### Option 3: Run specific test methods
```bash
python -m pytest RunningIn_semiSuperv/utils/tests/test_preprocess.py::TestRunInPreprocessor::test_update_labels_default_params -v
```

## Dependencies

The tests require the following packages:
- `unittest` (standard library)
- `pandas`
- `numpy`
- `pytest`
- `delayedsw` (for the actual implementation dependencies)

## Mock Usage

The tests use Python's `unittest.mock` to mock external dependencies:
- `MovingAverageTransformer` from `delayedsw`
- `DelayedSlidingWindow` from `delayedsw`

This allows testing the preprocessor logic without depending on the specific implementation details of these external transformers.

## Test Results

When all tests pass, you should see output similar to:
```
================= 21 passed in 0.57s =================
```

Each test method validates specific functionality and edge cases to ensure the `RunInPreprocessor` class behaves correctly under various conditions.

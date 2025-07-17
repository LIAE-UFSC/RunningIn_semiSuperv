# RunningIn Semi-Supervised Learning

A Python package for semi-supervised learning applied to [running-in](https://en.wikipedia.org/wiki/Break-in_(mechanical_run-in)) detection in hermetic alternative compressors. This package provides end-to-end functionality for data loading, preprocessing, model training, and evaluation specifically designed for industrial compressor monitoring applications.

## Features

- **Complete Pipeline**: Integrated data loading, preprocessing, and semi-supervised modeling
- **Advanced Preprocessing**: Time-series filtering, sliding window transformations, and moving averages
- **Flexible Data Splitting**: Support for both proportional and unit-based train/test splits
- **Class Balancing**: Built-in support for handling imbalanced datasets
- **Multiple Classifiers**: Support for various sklearn classifiers with semi-supervised learning
- **Comprehensive Evaluation**: Built-in model evaluation and performance metrics

## Quick Start

### Installation

```bash
git clone https://github.com/liae-labmetro/RunningIn_semiSuperv.git
cd RunningIn_semiSuperv
pip install -r requirements.txt
```

### Basic Usage

```python
from RunningIn_semiSuperv.utils import RunInSemiSupervised

# Initialize with predefined A-series units (A2, A3, A4, A5)
model = RunInSemiSupervised(
    compressor_model='a',  # Use A-series units from LabeledData folder
    features=['PressaoDescarga', 'PressaoSuccao', 'CorrenteRMS'],
    window_size=5,
    delay=2,
    moving_average=3,
    test_split=0.2,
    balance="undersample",
    classifier="LogisticRegression"
)

# Train the model
model.fit()

# Get training and test data
X_train, y_train = model.get_train_data()
X_test, y_test = model.get_test_data()

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Perform cross-validation
cv_results = model.cross_validate()
print(f"Cross-validation results: {cv_results}")
```

### Advanced Configuration

```python
# Custom classifier parameters
classifier_params = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42
}

# Semi-supervised learning parameters
semisupervised_params = {
    'threshold': 0.75,
    'max_iter': 10
}

# Custom file structure with unit-based test split
custom_files = [
    {'unit': 'A2', 'tests': ['/path/to/A2_test1.csv', '/path/to/A2_test2.csv']},
    {'unit': 'A3', 'tests': ['/path/to/A3_test1.csv', '/path/to/A3_test2.csv']},
    {'unit': 'B5', 'tests': ['/path/to/B5_test1.csv']}
]

model = RunInSemiSupervised(
    dict_folder=custom_files,  # Use custom file structure
    features=['PressaoDescarga', 'PressaoSuccao'],
    window_size=10,
    delay=5,
    t_min=2.0,
    t_max=50.0,
    run_in_transition_min=5.0,
    run_in_transition_max=15.0,
    test_split=['A2', 'A3'],  # Specific units for testing
    balance="undersample",
    classifier="RandomForestClassifier",
    classifier_params=classifier_params,
    semisupervised_params=semisupervised_params
)
```

## Package Structure

```
RunningIn_semiSuperv/
├── RunningIn_semiSuperv/
│   ├── __init__.py
│   ├── main_singleTest.py          # Example script
│   └── utils/
│       ├── __init__.py
│       ├── generator.py            # Main RunInSemiSupervised class
│       ├── load.py                 # Data loading utilities
│       ├── preprocess.py           # Data preprocessing pipeline
│       ├── models.py               # Semi-supervised model wrapper
│       └── tests/                  # Comprehensive test suite
│           ├── test_preprocess.py
│           ├── test_models.py
│           └── test_dataLoader.py
├── requirements.txt                # Package dependencies
└── README.md                       # This file
```

## Core Components

### RunInSemiSupervised (Main Class)

The primary interface for the entire pipeline, providing:

- **Data Management**: Automatic data loading and organization
- **Preprocessing Pipeline**: Time filtering, windowing, and feature engineering
- **Model Training**: Semi-supervised learning with customizable classifiers
- **Prediction Interface**: Easy prediction on new data
- **Evaluation Tools**: Comprehensive model assessment

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dict_folder` | list of dict | None | Custom unit/file structure (see below) |
| `compressor_model` | str | None | Predefined unit selection ("all", "a", "b") |
| `features` | list | None | Features to use (auto-detected if None) |
| `window_size` | int | 1 | Sliding window size for time series |
| `delay` | int | 1 | Delay parameter for windowing |
| `moving_average` | int | 1 | Moving average window size |
| `t_min` | float | 0 | Minimum time for filtering |
| `t_max` | float | inf | Maximum time for filtering |
| `run_in_transition_min` | float | 5 | Minimum time for run-in labeling |
| `run_in_transition_max` | float | inf | Maximum time for run-in labeling |
| `test_split` | float/list | 0.2 | Test split ratio or unit list |
| `balance` | str | "none" | Class balancing method |
| `classifier` | str | "LogisticRegression" | Base classifier type |
| `classifier_params` | dict | None | Additional classifier parameters |
| `semisupervised_params` | dict | None | Semi-supervised learning parameters |

### Data Source Configuration

The package supports two ways to specify data sources:

#### 1. Predefined Units (using `compressor_model` parameter)
Uses CSV files from the `LabeledData/` directory with predefined unit configurations:

```python
# Use all available units (A2, A3, A4, A5, B5, B7, B8, B10, B11, B12, B15)
model = RunInSemiSupervised(compressor_model="all")

# Use only A-series units (A2, A3, A4, A5)
model = RunInSemiSupervised(compressor_model="a")

# Use only B-series units (B5, B7, B8, B10, B11, B12, B15)
model = RunInSemiSupervised(compressor_model="b")
```

#### 2. Custom File Structure (using `dict_folder` parameter)
Specify custom units and file paths:

```python
custom_files = [
    {
        'unit': 'A2', 
        'tests': ['/path/to/A2_test1.csv', '/path/to/A2_test2.csv']
    },
    {
        'unit': 'CustomUnit', 
        'tests': ['/path/to/custom_test.csv']
    }
]
model = RunInSemiSupervised(dict_folder=custom_files)
```

## Data Format

The package expects CSV files with the following structure:

```csv
Tempo,Feature0,Feature1...
0.5,1.1,0.8...
1.0,1.2,0.9...
...
```

### Required Columns

- **Tempo**: Time values for ordering and filtering
- **Feature columns**: Sensor measurements (e.g., PressaoDescarga, PressaoSuccao, CorrenteRMS)

### Target Variable (for training data)

- **Amaciado**: Target variable (0=not run-in, 1=run-in) - automatically generated during preprocessing based on time thresholds and test numbers

### Automatically Added Columns

The data loader automatically adds these columns during loading:

- **Unidade**: Unit identifier derived from filename or dict_folder specification
- **N_ensaio**: Test number (0-indexed) for each test within a unit

### File Organization

#### For Predefined Units (`compressor_model` parameter):
```
LabeledData/
├── A2_N_2019_07_09.csv    # A2 unit, test 0 (N = not run-in)
├── A2_A_2019_08_08.csv    # A2 unit, test 1 (A = run-in)
├── A2_A_2019_08_28.csv    # A2 unit, test 2
├── A3_N_2019_12_04.csv    # A3 unit, test 0
└── ...
```

#### For Custom Units (`dict_folder` parameter):
Files can be located anywhere as specified in the dict_folder structure.

## Methods Reference

### Core Methods

- **`fit(load_data=True)`**: Train the complete pipeline
- **`predict(X)`**: Make predictions on preprocessed data
- **`transform_and_predict(X)`**: Preprocess and predict in one step
- **`predict_proba(X)`**: Get prediction probabilities
- **`cross_validate(n_splits=None)`**: Perform cross-validation evaluation
- **`get_train_data()`**: Get training features and labels
- **`get_test_data()`**: Get test features and labels
- **`test()`**: Evaluate model on test set (placeholder for future implementation)

### Configuration Methods

- **`_set_data_loader_params(dict_folder, compressor_model, features)`**: Configure data loading
- **`_set_preprocessor_params(...)`**: Configure preprocessing parameters  
- **`_set_model_params(classifier, classifier_params, semisupervised_params)`**: Configure model

> **Note:** Configuration methods may require reloading data and/or refitting the model for changes to take effect.

### Data Management Methods

- **`_load_data(reset=True)`**: Load or reload data
- **`_preprocess_data(data)`**: Preprocess loaded data
- **`_generate_transformers()`**: Initialize all pipeline components

## Examples

### Basic Classification

```python
# Simple binary classification for run-in detection using predefined units
model = RunInSemiSupervised(
    compressor_model='a',  # Use A-series units (A2, A3, A4, A5)
    features=['PressaoDescarga', 'PressaoSuccao'],
    classifier="LogisticRegression"
)

model.fit()
X_test, y_test = model.get_test_data()
predictions = model.predict(X_test)
```

### Time Series Analysis

```python
# Advanced time series preprocessing with B-series units
model = RunInSemiSupervised(
    compressor_model='b',  # Use B-series units
    window_size=20,
    delay=10,
    moving_average=5,
    t_min=5.0,
    t_max=100.0,
    classifier="RandomForestClassifier"
)

model.fit()
cv_results = model.cross_validate()
```

### Class Balancing

```python
# Handle imbalanced datasets with custom file structure
custom_files = [
    {'unit': 'A4', 'tests': ['/path/to/A4_test1.csv', '/path/to/A4_test2.csv']},
    {'unit': 'A5', 'tests': ['/path/to/A5_test1.csv']}
]

model = RunInSemiSupervised(
    dict_folder=custom_files,
    balance="undersample",
    test_split=['A4'],  # Use A4 for testing, A5 for training
    classifier="KNeighborsClassifier"
)

model.fit()
results = model.cross_validate()  # Unit-based cross-validation
```

### Cross-Validation

```python
# Unit-based cross-validation (leave-one-unit-out)
model = RunInSemiSupervised(
    compressor_model='a',
    features=['PressaoDescarga', 'CorrenteRMS'],
    classifier="LogisticRegression"
)

model.fit()
cv_results = model.cross_validate()  # Uses unit-based splits

# Stratified k-fold cross-validation
cv_results = model.cross_validate(n_splits=5)  # Uses stratified k-fold

# Access results
print(f"Percent labeled per fold: {cv_results['percent_labeled']}")
print(f"Confusion matrices: {cv_results['confusion_matrix']}")
```

## Performance Considerations

- **Memory Usage**: Large window sizes increase memory requirements
- **Class Balance**: Consider using balancing for highly imbalanced datasets

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Issue: Large datasets with extensive windowing
   - Solution: Reduce window_size or process data in batches

2. **Poor Performance**:
   - Issue: Imbalanced classes or inadequate features
   - Solution: Use class balancing and feature selection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Testing

```bash
# Run all tests
pytest RunningIn_semiSuperv/utils/tests/

# Run specific test file
pytest RunningIn_semiSuperv/utils/tests/test_preprocess.py -v
```

## References

<a id="1">[1]</a> 
Thaler, G. (2021). [_Desenvolvimento de métodos não invasivos para avaliação do processo de amaciamento de compressores herméticos alternativos_](https://repositorio.ufsc.br/handle/123456789/230918) [Development of non-invasive methods for evaluating the running-in process in reciprocating hermetic compressors] (Master's thesis, Federal University of Santa Catarina, Florianópolis, Brazil).

<a id="2">[2]</a> 
Thaler, G., Nunes, N. A., Nascimento, A. S. B. de S., Pacheco, A. L. S., Flesch, R. C. C. (2021). [Aplicação de aprendizado não supervisionado para identificação não destrutiva do amaciamento em compressores](https://doi.org/10.20906/sbai.v1i1.2611) (Application of unsupervised learning for non-destructive running-in identification in compressors). Proceedings of SBAI 2021, Brazil, 460–466.

<a id="1">[3]</a> 
Machado, J. L. et al. (2024). [_Semi-Supervised Learning Algorithm for Running-in Analysis on Compressors_](https://docs.lib.purdue.edu/icec/2854/). Proceedings of the International Compressor Engineering Conference 2024. West Lafayette, USA.

## License

This project is developed for research purposes at the Laboratory of Instrumentation and Automation (LIAE), Federal University of Santa Catarina (UFSC).

## Contact

For questions or support, please contact the LIAE team at UFSC.

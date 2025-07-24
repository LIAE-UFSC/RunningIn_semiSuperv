# RunningIn Semi-Supervised Learning

A Python package for semi-supervised learning applied to [running-in](https://en.wikipedia.org/wiki/Break-in_(mechanical_run-in)) detection in hermetic alternative compressors. This package provides end-to-end functionality for data loading, preprocessing, model training, and evaluation specifically designed for industrial compressor monitoring applications.

## Features

- **Complete Pipeline**: Integrated data loading, preprocessing, and semi-supervised modeling
- **Time Series Preprocessing**: Time-series filtering, sliding window transformations, and moving averages
- **Dimensionality Reduction**: Built-in PCA support for feature space reduction
- **Flexible Data Splitting**: Support for both proportional and unit-based train/test splits
- **Class Balancing**: Built-in support for handling imbalanced datasets with undersampling
- **Multiple Classifiers**: Support for various sklearn classifiers with semi-supervised learning
- **Cross-Validation**: Unit-based leave-one-out and stratified k-fold cross-validation

## Quick Start

### Installation

```bash
git clone https://github.com/liae-labmetro/RunningIn_semiSuperv.git
cd RunningIn_semiSuperv
pip install -r requirements.txt

# For hyperparameter optimization with dashboard (optional)
pip install optuna-dashboard
```

> **Note**: The package includes a dependency on `delayedsw` which is installed directly from GitHub. Ensure you have git available in your environment.

#### Installation Requirements

- **Python**: 3.6 or higher
- **Git**: Required for `delayedsw` dependency
- **optuna-dashboard**: Optional, for real-time optimization monitoring
- **PostgreSQL**: Optional, for advanced database storage (see PostgreSQL setup in Hyperparameter Optimization section)

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
    scale=True,           # Apply StandardScaler (default)
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

### Dimensionality Reduction with PCA

```python
# Example with PCA for high-dimensional feature reduction
model_with_pca = RunInSemiSupervised(
    compressor_model='all',    # Use all available data
    window_size=20,           # Large window creates many features
    pca=10,                   # Reduce to 10 principal components
    scale=True,               # Essential for PCA
    features=['CorrenteRMS', 'VibracaoCalotaInferiorRMS', 'VibracaoCalotaSuperiorRMS'],
    balance="undersample",
    classifier="LogisticRegression"
)

# PCA is applied automatically during training and prediction
model_with_pca.fit()
cv_results = model_with_pca.cross_validate()
```

## Package Structure

```
RunningIn_semiSuperv/
├── RunningIn_semiSuperv/
│   ├── __init__.py
│   ├── main_singleTest.py          # Basic usage example script
│   ├── main_optuna_full_optimizer.py  # Hyperparameter optimization script
│   └── utils/
│       ├── __init__.py
│       ├── generator.py            # Main RunInSemiSupervised class
│       ├── load.py                 # Data loading utilities
│       ├── preprocess.py           # Data preprocessing pipeline
│       ├── models.py               # Semi-supervised model wrapper
│       └── tests/                  # Comprehensive test suite (79 tests)
│           ├── test_preprocess.py  # Preprocessing tests (32 tests)
│           ├── test_models.py      # Model wrapper tests (10 tests)
│           ├── test_dataLoader.py  # Data loading tests (18 tests)
│           └── test_generator.py   # Main pipeline tests (19 tests)
├── Results/                        # Optimization results (auto-created)
│   ├── RunIn_LogisticRegression.db # Optuna study database
│   ├── RunIn_RandomForest.db       # Individual classifier results
│   └── ...                        # One database per classifier
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
| `scale` | bool | True | Apply StandardScaler to features |
| `t_min` | float | 0 | Minimum time for filtering |
| `t_max` | float | inf | Maximum time for filtering |
| `run_in_transition_min` | float | 5 | Minimum time for run-in labeling |
| `run_in_transition_max` | float | inf | Maximum time for run-in labeling |
| `test_split` | float/list | 0.2 | Test split ratio or unit list |
| `balance` | str | "none" | Class balancing method ("none", "undersample") |
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
- **`get_train_data(balanced=True)`**: Get training features and labels
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

## Hyperparameter Optimization

The package includes an advanced hyperparameter optimization script using [Optuna](https://optuna.org/) for automated parameter tuning across multiple classifiers. This feature enables systematic exploration of the hyperparameter space to find optimal configurations for your specific dataset.

### Optimization Script

The `main_optuna_full_optimizer.py` script provides comprehensive hyperparameter optimization with the following features:

- **Multi-Objective Optimization**: Simultaneously optimizes Matthews Correlation Coefficient (MCC) and labeled data percentage
- **Multiple Classifiers**: Supports 9 different classifiers with tailored parameter ranges
- **Real-time Monitoring**: Integrated Optuna dashboard for live optimization tracking
- **Parallel Execution**: Multi-process optimization for faster results
- **Error Handling**: Automatic cleanup of processes and subprocesses on errors

### Supported Classifiers

The optimization script natively supports the following classifiers with optimized parameter ranges:

1. **LogisticRegression**: C parameter (1e-4 to 1e2)
2. **DecisionTreeClassifier**: max_depth, min_samples_split, min_samples_leaf, max_features
3. **KNeighborsClassifier**: n_neighbors, weights, metric, p parameter
4. **LinearSVM**: C parameter (1e-5 to 1e3)
5. **RBFSVM**: C and gamma parameters
6. **RandomForest**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
7. **NeuralNet (MLPClassifier)**: hidden_layer_sizes, activation, alpha, learning_rate
8. **AdaBoost**: n_estimators, learning_rate
9. **NaiveBayes**: var_smoothing
10. **QDA**: reg_param, store_covariance

### Usage

#### Basic Optimization

```bash
# Run optimization for all supported classifiers
python RunningIn_semiSuperv/main_optuna_full_optimizer.py
```

#### Configuration

Edit the script parameters at the top of the file:

```python
compressor_model = "a"  # Choose from "a", "b", or "all"
n_processes = 3         # Number of parallel processes
n_tests = 500          # Number of optimization trials per classifier
max_init_samples = 180 # Maximum total window size constraint
```

#### Optimization Parameters

The script optimizes the following hyperparameters:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `window_size` | 1-150 | Sliding window size |
| `moving_average` | 1-30 | Moving average window |
| `delay` | 1-max_delay | Delay parameter (dynamically calculated) |
| `scale` | True/False | Feature scaling |
| `balance` | undersample/none | Class balancing method |
| `pca` | 0-window_size | PCA components (0 = no PCA, >0 = apply PCA) |
| `threshold` | 0.05-0.99 | Semi-supervised threshold |
| `features` | 26 options | Individual feature selection |
| `classifier_params` | varies | Classifier-specific parameters |

### Optuna Dashboard

The optimization script automatically starts an interactive dashboard for real-time monitoring of the optimization process.

#### Accessing the Dashboard

1. **Automatic Launch**: The dashboard starts automatically when running the optimization script
2. **Default URL**: `http://localhost:8080` (port increments for multiple classifiers)
3. **Manual Access**: If needed, you can start the dashboard manually:

```bash
# Install dashboard (if not already installed)
pip install optuna-dashboard

# Start dashboard manually
optuna-dashboard sqlite:///Results/RunIn_<<classifier_name>>.db --port 8080
```

#### Dashboard Features

- **Study Overview**: Real-time progress tracking across all trials
- **Pareto Front Visualization**: Multi-objective optimization results
- **Parameter Importance**: Identify which parameters matter most
- **Optimization History**: Track optimization progress over time
- **Trial Details**: Detailed information about each trial
- **Parallel Coordinate Plot**: Visualize parameter relationships

#### Dashboard Screenshots and Navigation

1. **Study List**: Overview of all optimization studies
2. **Study Detail**: Individual study progress and results
3. **Pareto Front**: Multi-objective optimization visualization
4. **Parameter Importance**: Feature importance analysis
5. **Optimization History**: Trial-by-trial progress tracking

### Results Storage

Optimization results are automatically saved to databases. The script supports both SQLite (default) and PostgreSQL for storing optimization results.

#### SQLite Storage (Default)
```
Results/
├── RunIn_LogisticRegression.db
├── RunIn_DecisionTreeClassifier.db
├── RunIn_KNeighborsClassifier.db
├── RunIn_LinearSVM.db
├── RunIn_RBFSVM.db
├── RunIn_RandomForest.db
├── RunIn_NeuralNet.db
├── RunIn_AdaBoost.db
├── RunIn_NaiveBayes.db
└── RunIn_QDA.db
```

#### PostgreSQL Storage (Advanced)

For large-scale optimizations or team collaboration, PostgreSQL provides better performance and concurrent access.

**Configuration:**
```python
# In main_optuna_full_optimizer.py
USE_POSTGRES = True  # Set to True to use PostgreSQL
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'optuna_db',
    'user': 'optuna_user',
    'password': 'optuna_password'
}
```

**Setup Instructions:**

1. **Install PostgreSQL Server**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   
   # macOS (with Homebrew)
   brew install postgresql
   
   # Windows: Download from https://www.postgresql.org/download/
   ```

2. **Install Python PostgreSQL Adapter**
   ```bash
   pip install psycopg2-binary
   # OR if you have compilation issues:
   pip install psycopg2
   ```

3. **Create Database and User**
   ```bash
   # Connect to PostgreSQL as superuser
   sudo -u postgres psql
   # OR on Windows/some systems:
   psql -U postgres
   ```
   
   ```sql
   -- Create database
   CREATE DATABASE optuna_db;
   
   -- Create user with password
   CREATE USER optuna_user WITH PASSWORD 'optuna_password';
   
   -- Grant permissions
   GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;
   
   -- Connect to the database
   \c optuna_db
   
   -- Grant schema permissions
   GRANT ALL ON SCHEMA public TO optuna_user;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO optuna_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO optuna_user;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO optuna_user;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO optuna_user;
   
   -- Exit
   \q
   ```

4. **Update Configuration**
   Update the `POSTGRES_CONFIG` dictionary in `main_optuna_full_optimizer.py` with your actual credentials.

**Benefits of PostgreSQL:**
- **Better Performance**: Superior handling of concurrent access and large datasets
- **Team Collaboration**: Multiple users can access the same optimization studies
- **Data Persistence**: More reliable for long-running optimizations
- **Network Access**: Can run database on a separate server
- **Advanced Features**: Better query capabilities and data analysis tools

**Dashboard Access with PostgreSQL:**
```bash
# Manual dashboard startup with PostgreSQL
optuna-dashboard postgresql://optuna_user:optuna_password@localhost:5432/optuna_db --port 8080
```

### Performance Tips

1. **Parallel Processing**: Increase `n_processes` for faster optimization (consider your CPU cores)
2. **Trial Budget**: Start with fewer trials (`n_tests=50`) for quick exploration, then increase for final optimization
3. **Dashboard Monitoring**: Use the dashboard to identify promising parameter regions early
4. **Resource Management**: The script automatically handles process cleanup on errors or interruption

### Troubleshooting Optimization

#### Common Issues

1. **Dashboard Not Starting**:
   ```bash
   # Install dashboard package
   pip install optuna-dashboard
   ```

2. **Port Already in Use**:
   - Manually specify a different port

3. **Memory Issues**:
   - Reduce `n_processes` or `n_tests`
   - Lower `max_init_samples` constraint

#### PostgreSQL Issues

1. **"Failed to import DB access module"**:
   ```bash
   # Install PostgreSQL adapter
   pip install psycopg2-binary
   ```

2. **"password authentication failed"**:
   - Check username and password in `POSTGRES_CONFIG`
   - Verify user exists: `psql -U postgres -c "\du"`
   - Create user if needed (see PostgreSQL setup above)

3. **"permission denied for schema public"**:
   ```bash
   # Connect as superuser
   psql -U postgres -d optuna_db
   ```
   ```sql
   -- Grant permissions
   GRANT ALL ON SCHEMA public TO optuna_user;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO optuna_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO optuna_user;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO optuna_user;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO optuna_user;
   ```

4. **"connection refused"**:
   ```bash
   # Check if PostgreSQL is running
   sudo systemctl status postgresql  # Linux
   brew services list | grep postgres  # macOS
   # On Windows: Check Services manager
   
   # Start PostgreSQL if needed
   sudo systemctl start postgresql  # Linux
   brew services start postgresql  # macOS
   ```

5. **Using Existing PostgreSQL Installation**:
   - Update `POSTGRES_CONFIG` with your existing credentials
   - Common defaults: user=`postgres`, database=`postgres`
   - Check your PostgreSQL installation documentation for credentials

**Quick PostgreSQL Test:**
```bash
# Test connection with your credentials
psql -h localhost -p 5432 -U optuna_user -d optuna_db

# If successful, you should see:
# psql (version)
# Type "help" for help.
# optuna_db=>
```

## Performance Considerations

- **Memory Usage**: Large window sizes increase memory requirements
- **Class Balance**: Consider using balancing for highly imbalanced datasets

## Troubleshooting

### Common Issues

1. **Installation Errors**:
   - Issue: `delayedsw` installation fails
   - Solution: Ensure git is installed and accessible from your environment

2. **Memory Errors**:
   - Issue: Large datasets with extensive windowing
   - Solution: Reduce window_size or process data in batches

3. **Poor Performance**:
   - Issue: Imbalanced classes or inadequate features
   - Solution: Use class balancing and feature selection

4. **Import Errors**:
   - Issue: Cannot import RunInSemiSupervised
   - Solution: Ensure you're in the correct directory and all dependencies are installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Testing

```bash
# Run all tests (79 tests total)
pytest RunningIn_semiSuperv/utils/tests/

# Run specific test modules
pytest RunningIn_semiSuperv/utils/tests/test_preprocess.py -v     # 32 tests
pytest RunningIn_semiSuperv/utils/tests/test_models.py -v        # 10 tests
pytest RunningIn_semiSuperv/utils/tests/test_dataLoader.py -v    # 18 tests
pytest RunningIn_semiSuperv/utils/tests/test_generator.py -v     # 19 tests
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

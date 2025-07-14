from RunningIn_semiSuperv.utils.load import RunInDataLoader
from RunningIn_semiSuperv.utils.preprocess import RunInPreprocessor
from RunningIn_semiSuperv.utils.models import RunInSemiSupervisedModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

class RunInSemiSupervised:
    """
    Complete pipeline for semi-supervised learning applied to running-in detection 
    in hermetic alternative compressors.
    
    This class integrates data loading, preprocessing, and semi-supervised modeling
    into a unified interface for industrial compressor monitoring applications. It
    provides end-to-end functionality from raw sensor data to trained models capable
    of detecting running-in states in compressor operation.
    
    The pipeline consists of three main components:
    1. RunInDataLoader: Handles data loading and organization
    2. RunInPreprocessor: Applies time filtering, windowing, and feature engineering  
    3. RunInSemiSupervisedModel: Implements semi-supervised learning with sklearn classifiers
    
    Parameters
    ----------
    dict_folder : list of dict, optional
        Custom list of dictionaries defining unit names and their associated test 
        file paths. Each dictionary should contain 'unit' and 'tests' keys, where
        'tests' is a list of file paths. If None, uses predefined units based on
        the 'model' parameter. Default is None.
        
        Example structure:
        [{'unit': 'A2', 'tests': ['/path/to/A2_test1.csv', '/path/to/A2_test2.csv']},
         {'unit': 'A3', 'tests': ['/path/to/A3_test1.csv']}]
        
    model : {"all", "a", "b"}, optional
        Model type for selecting predefined units when dict_folder is None:
        - "all": Load all available units (A2, A3, A4, A5, B5, B7, B8, B10, B11, B12, B15)
        - "a": Load only A-series units (A2, A3, A4, A5)  
        - "b": Load only B-series units (B5, B7, B8, B10, B11, B12, B15)
        Required if dict_folder is None. Default is None.
        
    features : list of str, optional
        List of feature column names to use for modeling. If None, all numeric
        columns except metadata will be used automatically. Default is None.
        
    window_size : int, default=1
        Size of the sliding window for time series feature engineering. Larger
        windows capture more temporal context but increase computational cost.
        
    delay : int, default=1
        Delay parameter for the sliding window transformation. Controls the spacing
        between observations in the window.
        
    moving_average : int, default=1
        Window size for moving average filtering. Applied before windowing to
        smooth sensor signals and reduce noise.
        
    t_min : float, default=0
        Minimum time value for data filtering. Observations before this time
        are excluded from analysis.
        
    t_max : float, default=np.inf
        Maximum time value for data filtering. Observations after this time
        are excluded from analysis.
        
    run_in_transition_min : float, default=5
        Minimum time for run-in transition labeling. Observations in the first
        test (N_ensaio=0) before this time are labeled as not run-in (0).
        
    run_in_transition_max : float, default=np.inf
        Maximum time for run-in transition labeling. Observations in the first
        test (N_ensaio=0) after this time are labeled as run-in (1). Values
        between min and max are labeled as unknown (-1).
        
    test_split : float or list, default=0.2
        Test set splitting strategy:
        - float: Proportion of data for testing (0.0 to 1.0)
        - list: Specific unit names to use for testing (e.g., ['A2', 'A3'])
        
    balance : {"none", "undersample"}, default="none"
        Class balancing strategy for handling imbalanced datasets:
        - "none": No balancing applied
        - "undersample": Random undersampling of majority class
        
    classifier : str, default="LogisticRegression"
        Base classifier for semi-supervised learning. Supported options include:
        "LogisticRegression", "RandomForestClassifier", "KNeighborsClassifier",
        and other sklearn classifiers.
        
    classifier_params : dict, optional
        Additional parameters to pass to the base classifier constructor.
        Default is None (empty dict).
        
    semisupervised_params : dict, optional
        Parameters for the SelfTrainingClassifier wrapper, such as threshold
        and max_iter. Default is None (empty dict).
        
    Attributes
    ----------
    data_loader : RunInDataLoader
        Configured data loader instance for reading CSV files.
        
    preprocessor : RunInPreprocessor
        Configured preprocessor instance for data transformation.
        
    model : RunInSemiSupervisedModel
        Configured semi-supervised model instance for training and prediction.
        
    Examples
    --------
    Basic usage with predefined units:
    
    >>> model = RunInSemiSupervised(
    ...     model="a",  # Use A-series units (A2, A3, A4, A5)
    ...     features=['PressaoDescarga', 'PressaoSuccao'],
    ...     window_size=5,
    ...     classifier="LogisticRegression"
    ... )
    >>> model.fit()
    >>> predictions = model.predict(X_new)
    
    Using all available units:
    
    >>> model = RunInSemiSupervised(
    ...     model="all",  # Use all predefined units
    ...     features=['PressaoDescarga', 'PressaoSuccao', 'CorrenteRMS'],
    ...     window_size=10,
    ...     balance="undersample"
    ... )
    
    Advanced configuration with custom file structure:
    
    >>> custom_files = [
    ...     {'unit': 'CustomA', 'tests': ['/path/to/testA1.csv', '/path/to/testA2.csv']},
    ...     {'unit': 'CustomB', 'tests': ['/path/to/testB1.csv']}
    ... ]
    >>> classifier_params = {'C': 1.0, 'random_state': 42}
    >>> semisupervised_params = {'threshold': 0.75, 'max_iter': 10}
    >>> model = RunInSemiSupervised(
    ...     dict_folder=custom_files,
    ...     features=['PressaoDescarga', 'PressaoSuccao', 'CorrenteRMS'],
    ...     window_size=10,
    ...     delay=5,
    ...     moving_average=3,
    ...     test_split=['CustomA'],  # Use CustomA for testing
    ...     balance="undersample",
    ...     classifier="RandomForestClassifier",
    ...     classifier_params=classifier_params,
    ...     semisupervised_params=semisupervised_params
    ... )
    
    Time-based filtering and unit-based test splitting:
    
    >>> model = RunInSemiSupervised(
    ...     model="b",  # Use B-series units
    ...     test_split=['B5', 'B7'],  # Use specific units for testing
    ...     t_min=5.0,
    ...     t_max=100.0,
    ...     run_in_transition_min=5.0,
    ...     run_in_transition_max=15.0
    ... )
    
    Notes
    -----
    The class expects CSV files with specific column structure including 'Tempo',
    'Unidade', 'N_ensaio', 'Amaciado', and sensor measurement columns. The pipeline
    automatically handles data preprocessing, feature engineering, and model training.
    
    For best performance, consider:
    - Using appropriate window_size based on signal characteristics
    - Applying class balancing for highly imbalanced datasets  
    - Selecting relevant features based on domain knowledge
    - Using unit-based splitting for more realistic validation
    
    See Also
    --------
    RunInDataLoader : Data loading and organization utilities
    RunInPreprocessor : Data preprocessing and feature engineering pipeline
    RunInSemiSupervisedModel : Semi-supervised learning model wrapper
    """
    def __init__(self, 
                 dict_folder: Optional[List[Dict[str, Union[str, List[str]]]]] = None, 
                 model: Optional[str] = None, 
                 features: Optional[List[str]] = None,
                 window_size: int = 1, 
                 delay: int = 1, 
                 moving_average: int = 1, 
                 t_min: float = 0, 
                 t_max: float = np.inf, 
                 run_in_transition_min: float = 5, 
                 run_in_transition_max: float = np.inf,
                 test_split: Union[float, List[str]] = 0.2, 
                 balance: str = "none", 
                 classifier: str = "LogisticRegression",
                 classifier_params: Optional[Dict[str, Any]] = None,
                 semisupervised_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the RunInSemiSupervised pipeline with specified parameters.
        
        Sets up the complete pipeline by configuring data loading, preprocessing,
        and modeling components. All parameters are stored as instance attributes
        and used to configure the respective pipeline components.
        
        Parameters are passed through to the appropriate components:
        - Data loading parameters → RunInDataLoader
        - Preprocessing parameters → RunInPreprocessor  
        - Model parameters → RunInSemiSupervisedModel
        
        The initialization automatically calls _generate_transformers() to create
        and configure all pipeline components.
        """
        self.dict_folder = dict_folder
        self.model = model
        self.features = features
        self.window_size = window_size
        self.delay = delay
        self.moving_average = moving_average
        self.t_min = t_min
        self.t_max = t_max
        self.run_in_transition_min = run_in_transition_min
        self.run_in_transition_max = run_in_transition_max
        self.test_split = test_split
        self.balance = balance
        self.classifier = classifier
        self.classifier_params = classifier_params if classifier_params is not None else {}
        self.semisupervised_params = semisupervised_params if semisupervised_params is not None else {}

        self._generate_transformers()

    def _set_data_loader_params(self, dict_folder=None, model=None, features=None):
        """
        Set parameters for the data loader.
        
        Args:
            dict_folder (str): Path to the folder containing data.
            model (str): Model type to be used.
            features (list): List of feature names to be used.
        """
        if dict_folder is not None:
            self.dict_folder = dict_folder
        if model is not None:
            self.model = model
        if features is not None:
            self.features = features

        self.data_loader = RunInDataLoader(
            dict_folder=self.dict_folder,
            model=self.model,
            features=self.features
        )

    def _load_data(self):
        """
        Load data using the RunInDataLoader.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        
        data_loader = RunInDataLoader(dict_folder=self.dict_folder, model=self.model, features=self.features)
        return data_loader.load_data()

    def _set_preprocessor_params(self, window_size=None, delay=None, moving_average=None,
                                    t_min=None, t_max=None, run_in_transition_min=None,
                                    run_in_transition_max=None, test_split=None, balance=None):
        """
        Set parameters for the data preprocessor.
        
        Updates preprocessing parameters and creates a new RunInPreprocessor instance
        with the current configuration. This method allows dynamic reconfiguration
        of preprocessing settings after initialization.
        
        Args:
            window_size (int, optional): Size of the sliding window for time series.
            delay (int, optional): Delay parameter for windowing.
            moving_average (int, optional): Window size for moving average filtering.
            t_min (float, optional): Minimum time for data filtering.
            t_max (float, optional): Maximum time for data filtering.
            run_in_transition_min (float, optional): Minimum time for run-in labeling.
            run_in_transition_max (float, optional): Maximum time for run-in labeling.
            test_split (float or list, optional): Test set splitting strategy.
            balance (str, optional): Class balancing method.
        """

        if window_size is not None:
            self.window_size = window_size
        if delay is not None:
            self.delay = delay
        if moving_average is not None:
            self.moving_average = moving_average
        if t_min is not None:
            self.t_min = t_min
        if t_max is not None:
            self.t_max = t_max
        if run_in_transition_min is not None:
            self.run_in_transition_min = run_in_transition_min
        if run_in_transition_max is not None:
            self.run_in_transition_max = run_in_transition_max
        if test_split is not None:
            self.test_split = test_split
        if balance is not None:
            self.balance = balance

        self.preprocessor = RunInPreprocessor(
            window_size=self.window_size,
            delay=self.delay,
            moving_average=self.moving_average,
            run_in_transition_min=self.run_in_transition_min,
            run_in_transition_max=self.run_in_transition_max,
            test_split=self.test_split,
            balance=self.balance,
            features=self.features
        )

    def _preprocess_data(self, data):
        """
        Preprocess the data using the RunInPreprocessor.
        
        Args:
            data (pd.DataFrame): Raw data to be preprocessed.
        
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        
        self.preprocessor.fit_transform(data)
        
        # Return the preprocessed features and labels
        return self.preprocessor.X_train, self.preprocessor.y_train
    
    def _set_model_params(self, classifier=None, classifier_params=None, semisupervised_params=None):
        """
        Set parameters for the semi-supervised model.
        
        Args:
            classifier (str): Classifier type to be used.
            classifier_params (dict): Parameters for the classifier.
            semisupervised_params (dict): Parameters for the semi-supervised model.
        """
        
        if classifier is not None:
            self.classifier = classifier
        if classifier_params is not None:
            self.classifier_params = classifier_params
        if semisupervised_params is not None:
            self.semisupervised_params = semisupervised_params

        self.model = RunInSemiSupervisedModel(
            classifier=self.classifier,
            classifier_args=self.classifier_params,
            **self.semisupervised_params
        )
    
    def _generate_transformers(self):
        self._set_data_loader_params()
        self._set_preprocessor_params()
        self._set_model_params()

    def fit(self, load_data: bool = True) -> 'RunInSemiSupervised':

        if load_data:
            # Load data
            data = self.data_loader.load_data()

            # Preprocess data
            self.preprocessor.fit_transform(data)

        # Train model
        self.model.fit(self.preprocessor.X_train, self.preprocessor.y_train)
        
        return self

    def transform_and_predict(self, X: Any) -> Any:
        """
        Transform and predict using the trained model.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        # Ensure X is preprocessed
        X_transformed = self.preprocessor.transform(X)
        
        # Make predictions
        return self.model.predict(X_transformed)
    
    def predict(self, X: Any) -> Any:
        """
        Predict labels using the trained model.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_proba(self, X: Any) -> Any:
        """
        Predict probabilities using the trained model.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted probabilities for each class.
        """
        # Ensure X is preprocessed
        X_transformed = self.preprocessor.transform(X)
        
        # Make probability predictions
        return self.model.predict_proba(X_transformed)
    
    def evaluate(self) -> Any:
        """
        Evaluate the model on the test set.
        
        Returns:
            dict: Evaluation metrics.
        """
        
        # Get test data
        X_test, y_test = self.preprocessor.get_test_data()
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate evaluation metrics
        metrics = self.model.evaluate(y_test, y_pred)
        
        return metrics
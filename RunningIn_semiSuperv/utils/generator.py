from RunningIn_semiSuperv.utils.load import RunInDataLoader
from RunningIn_semiSuperv.utils.preprocess import RunInPreprocessor
from RunningIn_semiSuperv.utils.models import RunInSemiSupervisedModel
import pandas as pd
import numpy as np

class RunInSemiSupervised:
    def __init__(self, dict_folder=None, model=None, features=None,
                 window_size=1, delay=1, moving_average=1, t_min=0, t_max=np.inf, 
                 run_in_transition_min=5, run_in_transition_max=np.inf,
                 test_split=0.2, balance="none", 
                 classifier = "LogisticRegression",
                 classifier_params=None,
                 semisupervised_params=None):
        # TODO: write docstring
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
        # TODO: write docstring

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

    def fit(self, load_data=True):

        if load_data:
            # Load data
            data = self.data_loader.load_data()

            # Preprocess data
            self.preprocessor.fit_transform(data)

        # Train model
        self.model.fit(self.preprocessor.X_train, self.preprocessor.y_train)

    def transform_and_predict(self, X):
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
    
    def predict(self, X):
        """
        Predict labels using the trained model.
        
        Args:
            X (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_proba(self, X):
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
    
    def evaluate(self):
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
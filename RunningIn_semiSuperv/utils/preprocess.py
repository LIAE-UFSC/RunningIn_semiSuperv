from delayedsw import DelayedSlidingWindow, MovingAverageTransformer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

class RunInPreprocessor:
    """
    A preprocessor for run-in analysis data that handles time filtering, feature selection,
    sliding window transformations, and label generation for semi-supervised learning.
    
    This class is designed to preprocess data for run-in period detection in mechanical
    systems, applying moving averages, sliding windows, and labeling strategies to
    prepare data for machine learning models.
    
    Attributes:
        window_size (int): Size of the sliding window for feature transformation.
        delay (int): Delay parameter for the sliding window.
        features (list or None): List of feature columns to use. If None, all columns are used.
        moving_average (int): Window size for moving average filtering.
        t_min (float): Minimum time threshold for filtering data.
        t_max (float): Maximum time threshold for filtering data.
        scale (bool): Whether to apply scaling to the features.
        run_in_transition_min (float): Minimum time for run-in transition labeling.
        run_in_transition_max (float): Maximum time for run-in transition labeling.
        X (pd.DataFrame): Processed feature matrix.
        y (pd.Series): Target labels.
        feature_names_in_ (list): Names of input features from the fitted data.
    """
    
    def __init__(self, 
                 window_size: int = 1, 
                 delay: int = 1, 
                 features: Optional[List[str]] = None, 
                 moving_average: int = 1, 
                 t_min: float = 0, 
                 t_max: float = np.inf, 
                 scale: bool = False,
                 run_in_transition_min: float = 5, 
                 run_in_transition_max: float = np.inf,
                 test_split: Union[float, List[str]] = 0.2, 
                 balance: str = "none") -> None:
        """
        Initialize the RunInPreprocessor with specified parameters.
        
        Args:
            window_size (int, optional): Size of the sliding window. Defaults to 1.
            delay (int, optional): Delay space for the sliding window. Defaults to 1.
            features (list, optional): List of feature columns to transform. If None,
                all columns will be used. Defaults to None.
            moving_average (int, optional): Window size for moving average filter. Defaults to 1.
            t_min (float, optional): Minimum time threshold for data filtering. Defaults to 0.
            t_max (float, optional): Maximum time threshold for data filtering. Defaults to np.inf.
            scale (bool, optional): Whether to apply scaling to the features. Defaults to False.
            run_in_transition_min (float, optional): Minimum time threshold for run-in
                transition labeling. Defaults to 5.
            run_in_transition_max (float, optional): Maximum time threshold for run-in
                transition labeling. Defaults to np.inf.
            test_split (float, list, optional): Proportion of data to use for testing. If list, it 
                should be a list of strings containing names of units in the dataset. Defaults to 0.2.
            balance (str, optional): Method for balancing classes. Options are "none" and "undersample".
                "undersample" will reduce the majority class using random undersampling.
        """

        self.window_size = window_size
        self.delay = delay
        self.features = features
        self.moving_average = moving_average
        self.t_min = t_min
        self.t_max = t_max
        self.run_in_transition_min = run_in_transition_min
        self.run_in_transition_max = run_in_transition_max
        self.scale = scale
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.index_train = []
        self.index_test = []
        self.index_train_balanced = []
        self.metadata = []
        self.feature_names_in_ = []
        self.test_split = test_split
        self.balance = balance
        self.metadata = pd.DataFrame()


    def set_filter_params(self, 
                          features: Optional[List[str]] = None, 
                          moving_average_window: int = 1, 
                          t_min: float = 0, 
                          t_max: float = np.inf, 
                          reset: bool = True) -> None:
        """
        Set filtering parameters for the preprocessor.
        
        Args:
            features (list, optional): List of feature columns to use. If None,
                all columns will be used. Defaults to None.
            moving_average_window (int, optional): Window size for moving average filter.
                Defaults to 1.
            t_min (float, optional): Minimum time threshold for data filtering. Defaults to 0.
            t_max (float, optional): Maximum time threshold for data filtering. Defaults to np.inf.
            reset (bool, optional): Whether to reset the internal states of the preprocessor.
                Defaults to True.
        """

        self.features = features
        self.moving_average = moving_average_window
        self.t_min = t_min
        self.t_max = t_max

        if reset:
            self._clear_data()

    def set_test_split(self, test_split: Union[float, List[str]] = 0.2, reset: bool = True) -> None:
        """
        Set the test split parameter for the preprocessor.
        
        Args:
            test_split (float or list, optional): Proportion of data to use for testing.
                If float, it specifies the proportion. If list, it should be a list of
                strings containing names of units in the dataset. Defaults to 0.2.
            reset (bool, optional): Whether to reset the internal states of the preprocessor.
                Defaults to True.
        """

        self.test_split = test_split

        if reset:
            self._clear_data()


    def set_window_params(self, window_size=1, delay=1, features=None, reset=True):
        """
        Set sliding window parameters for the preprocessor.
        
        Args:
            window_size (int, optional): Size of the sliding window for feature transformation.
                Defaults to 1.
            delay (int, optional): Delay space for the sliding window transformation.
                Defaults to 1.
            features (list, optional): List of feature columns to transform. If None,
                all columns will be used. Defaults to None.
            reset (bool, optional): Whether to reset the internal states of the preprocessor.
                Defaults to True.
        """

        self.window_size = window_size
        self.delay = delay
        self.features = features

        if reset:
            self._clear_data()

    def set_label_params(self, run_in_transition_min=5, run_in_transition_max=np.inf, reset=True):
        """
        Set labeling parameters for run-in transition detection.
        
        Args:
            run_in_transition_min (float, optional): Minimum time threshold for run-in
                transition labeling. Data before this time in the first test will be
                labeled as not run-in (0). Defaults to 5.
            run_in_transition_max (float, optional): Maximum time threshold for run-in
                transition labeling. Data after this time in the first test will be
                labeled as run-in (1). Defaults to np.inf.
            reset (bool, optional): Whether to reset the internal states of the preprocessor.
                Defaults to True.
        """

        self.run_in_transition_min = run_in_transition_min
        self.run_in_transition_max = run_in_transition_max

        if reset:
            self._clear_data()

    def _clear_data(self):
        """
        Clear the internal data structures of the preprocessor.
        
        This method resets the feature matrix, target labels, and index lists to their
        initial empty states. It is useful for reinitializing the preprocessor without
        creating a new instance.
        """
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.index_train = []
        self.index_test = []
        self.index_train_balanced = []

    def set_balance_params(self, balance="none", reset=True):
        """
        Set the class balancing parameter for the preprocessor.
        
        Args:
            balance (str, optional): Method for balancing classes. Options are "none" and 
                "undersample". "undersample" will reduce the majority class using random 
                undersampling. Defaults to "none".
            reset (bool, optional): Whether to reset the internal states of the preprocessor.
                Defaults to True.
        """
        self.balance = balance

        if reset:
            self._clear_data()

    def fit(self, data: pd.DataFrame) -> 'RunInPreprocessor':
        """
        Fit the preprocessor to the input data.
        
        This method initializes the feature transformers (MovingAverageTransformer and
        DelayedSlidingWindow) based on the current configuration and validates that
        all required features are present in the input data.
        
        Args:
            data (pd.DataFrame): Input data containing features and target columns.
                Must contain columns 'Unidade', 'N_ensaio', 'Tempo', and 'Amaciado'.
                
        Raises:
            ValueError: If any specified features are missing from the input data.
        """

        if self.features is None:
            # Exclude target and metadata columns from feature transformation
            exclude_columns = ['Amaciado', 'Unidade', 'N_ensaio', 'Tempo']
            self._features = [col for col in data.columns.tolist() if col not in exclude_columns]
        else:
            self._features = self.features

        self.feature_names_in_ = data.columns.tolist()

        # Check if any features are missing from the data
        missing_features = [feat for feat in self._features if feat not in data.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the input data: {missing_features}")
        
        if self.scale:
            # Initialize scaler if scaling is enabled
            self._ScalerTransformer = StandardScaler()
        else:
            self._ScalerTransformer = FunctionTransformer() # No scaling, just a passthrough
        self._FilterTransformer = MovingAverageTransformer(window=self.moving_average, columns_to_transform=self._features)
        self._WindowTransformer = DelayedSlidingWindow(window_size=self.window_size, delay_space=self.delay,
                                              columns_to_transform=self._features,
                                              split_by=['Unidade', 'N_ensaio'], order_by=['Tempo'])
        
        return self

    def transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform input data using the fitted preprocessor.
        
        This method applies the complete preprocessing pipeline:
        1. Filter data based on time constraints
        2. Generate labels for run-in transition detection
        3. Sort data by group and time columns
        4. Apply moving average filtering
        5. Apply sliding window transformation
        6. Remove NaN values
        7. Split the data into training and testing sets
        8. Balance classes if specified
        
        Args:
            data (pd.DataFrame): Input data to transform. Must contain the same
                columns as the data used during fitting.
                
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Processed feature matrix
                - y (pd.Series): Target labels for run-in detection
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """

        if not self.feature_names_in_:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() before transform().")

        group_columns = ['Unidade','N_ensaio']
        order_columns = ['Tempo']
        
        # Filter rows based on time
        data = self._filter_time(data)

        # Label the data
        data = self.update_labels(data)

        data[self._features] = self._ScalerTransformer.fit_transform(data[self._features]) # Scale the features if scaling is enabled
        data = self._FilterTransformer.fit_transform(data)
        data = self._WindowTransformer.fit_transform(data.reset_index(drop=True)) # TODO: Remove reset_index after issue is resolved in delayedsw

        data = data.dropna()
        data = data.reset_index(drop=True)

        # Drop all columns which are not in the feature list
        data_columns = data.columns.tolist()

        
        # Keep only columns that start with any of the names in self._features, plus metadata columns
        metacolumns = ['Unidade', 'N_ensaio', 'Tempo', 'Amaciado']
        self.metadata = data[metacolumns]

        feature_cols = []
        for feat in self._features:
            feature_cols.extend([col for col in data_columns if col.startswith(feat)])
        cols_to_keep = list(set(feature_cols + ['Amaciado']))
        data = data[[col for col in data_columns if col in cols_to_keep]]

        
        
        # Split X and y
        self.X, self.y = self._splitXY(data)

        # Split the data into training and testing sets
        self._train_test_split(self.X, self.y)

        # Balance classes if specified
        self.balance_classes()

        return self.get_full_data()  # Return preprocessed X and y

    def get_train_test_data(self):
        """
        Get the training and testing data after preprocessing.
        
        This method returns the preprocessed training and testing feature matrices
        and target labels. It is useful for accessing the data after calling fit_transform.
        
        Returns:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
                - X_train (pd.DataFrame): Training feature matrix
                - y_train (pd.Series): Training target labels
                - X_test (pd.DataFrame): Testing feature matrix
                - y_test (pd.Series): Testing target labels
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if self.X.empty or self.y.empty:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() and transform() before accessing the data.")

        return self.X.iloc[self.train_index], self.y.iloc[self.train_index], self.X.iloc[self.test_index], self.y.iloc[self.test_index]

    def get_train_metadata(self) -> pd.DataFrame:
        """
        Get the metadata for the training set after preprocessing.
        
        This method returns the metadata associated with the training data,
        including unit, test number, time, and run-in labels.
        
        Returns:
            pd.DataFrame: Metadata for the training set.
        """
        return self.metadata.iloc[self.train_index].reset_index(drop=True)
    
    def get_balanced_train_metadata(self) -> pd.DataFrame:
        """
        Get the metadata for the balanced training set after preprocessing.
        
        This method returns the metadata associated with the balanced training data,
        including unit, test number, time, and run-in labels.
        
        Returns:
            pd.DataFrame: Metadata for the balanced training set.
        """

        if self.balance == "none":
            return self.get_train_metadata()
        elif self.index_train_balanced is None or len(self.index_train_balanced) == 0:
            raise ValueError("No balanced training data available. Please call balance_classes() first.")
        elif self.balance == "undersample":
            return self.metadata.iloc[self.index_train_balanced].reset_index(drop=True)
    
    def get_test_metadata(self) -> pd.DataFrame:
        """
        Get the metadata for the test set after preprocessing.
        
        This method returns the metadata associated with the test data,
        including unit, test number, time, and run-in labels.
        
        Returns:
            pd.DataFrame: Metadata for the test set.
        """
        return self.metadata.iloc[self.test_index].reset_index(drop=True)
    
    def get_full_metadata(self) -> pd.DataFrame:
        """
        Get the full metadata for both training and test sets after preprocessing.
        
        This method returns the complete metadata associated with both training
        and test data, including unit, test number, time, and run-in labels.
        
        Returns:
            pd.DataFrame: Full metadata for both training and test sets.
        """
        return pd.concat([self.metadata.iloc[self.train_index], self.metadata.iloc[self.test_index]], ignore_index=True).reset_index(drop=True)

    def get_train_data(self, balanced = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the preprocessed data after fitting.
        
        This method returns the processed feature matrix and target labels.
        It is useful for accessing the data after calling fit_transform.

        Args:
            balanced (bool, optional): If True, returns the balanced training data.
                If None, returns the original training data without balancing.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Processed feature matrix
                - y (pd.Series): Target labels for run-in detection
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if self.X.empty or self.y.empty:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() and transform() before accessing the data.")

        return self.X.iloc[self.train_index], self.y.iloc[self.train_index]
    
    def get_balanced_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the balanced training data after fitting.
        
        This method returns the processed feature matrix and target labels for the
        training set, ensuring that the classes are balanced according to the specified
        balancing method (e.g., undersampling).
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X_train_balanced (pd.DataFrame): Balanced training feature matrix
                - y_train_balanced (pd.Series): Balanced training target labels for run-in detection
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if self.X.empty or self.y.empty:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() and transform() before accessing the data.")

        if self.balance == "none":
            # If no balancing is applied, return the original training data
            return self.get_train_data()
        # If balancing is applied, return the balanced training data
        elif self.index_train_balanced is None or len(self.index_train_balanced) == 0:
            raise ValueError("No balanced training data available. Please call balance_classes() first.")
        elif self.balance == "undersample":
            return self.X.iloc[self.index_train_balanced], self.y[self.index_train_balanced]

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the preprocessed test data after fitting.
        
        This method returns the processed feature matrix and target labels for the test set.
        It is useful for accessing the test data after calling fit_transform.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X_test (pd.DataFrame): Processed test feature matrix
                - y_test (pd.Series): Test target labels for run-in detection
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if self.X.empty or self.y.empty:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() and transform() before accessing the data.")

        return self.X.iloc[self.test_index], self.y.iloc[self.test_index]

    def get_full_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get the full preprocessed data after fitting.
        
        This method returns the complete processed feature matrix and target labels,
        including both training and testing data.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Complete processed feature matrix
                - y (pd.Series): Complete target labels for run-in detection
                
        Raises:
            ValueError: If the preprocessor has not been fitted yet.
        """
        if self.X.empty or self.y.empty:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() and transform() before accessing the data.")

        indexes = np.concatenate((self.train_index, self.test_index))
        return (self.X.iloc[indexes], self.y.iloc[indexes])

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the preprocessor to the data and transform it in one step.
        
        This method combines the fit and transform operations for convenience,
        equivalent to calling fit(data) followed by transform(data).
        
        Args:
            data (pd.DataFrame): Input data to fit and transform.
            
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Processed feature matrix
                - y (pd.Series): Target labels for run-in detection
        """
        self.fit(data)
        return self.transform(data)

    def _splitXY(self, data):
        """
        Split the data into features (X) and target (y).
        
        This private method separates the 'Amaciado' column as the target variable
        and uses all other columns as features, preserving metadata columns that
        are needed by DelayedSlidingWindow for ordering and grouping.
        
        Args:
            data (pd.DataFrame): Input data containing both features and target.
            
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Feature matrix (all columns except 'Amaciado')
                - y (pd.Series): Target variable ('Amaciado' column)
        """
        
        y = data.loc[:,'Amaciado']
        X = data.drop(['Amaciado'], axis= 1)
        return X, y
    
    def _train_test_split(self, X, y):
        """
        Split the data into training and testing sets based on the test_split parameter.
        
        This private method handles both float and list inputs for test_split:
        - If float, it splits the data into training and testing sets based on the proportion.
        - If list, it filters the data to include only specified units for testing.
        
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            
        Returns:
            tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: A tuple containing:
                - X_train (pd.DataFrame): Training feature matrix
                - y_train (pd.Series): Training target labels
                - X_test (pd.DataFrame): Testing feature matrix
                - y_test (pd.Series): Testing target labels
        """
        if isinstance(self.test_split, float):
            # Randomly split data into train and test sets
            # Use stratification only if there are sufficient samples in each class
            X_train, X_test, = train_test_split(X, test_size=self.test_split, random_state=42, stratify=y)
            # Store test indexes as a boolean mask over the original X
            test_index = X.index.isin(X_test.index)
        elif isinstance(self.test_split, list):
            # Filter data to include only specified units for testing
            test_units = self.test_split
            test_index = X['Unidade'].isin(test_units)
        else:
            raise ValueError("test_split must be a float or a list of unit names.")

        self.train_index = np.where(~test_index)[0]
        self.test_index = np.where(test_index)[0]

    def balance_classes(self):
        """
        Balance the classes of training data.
        
        If self.balance is "undersample", this private method reduces the majority
        class using random undersampling to balance the dataset.

        Args:
            None

        Returns:
            None
        """
        if self.balance == "undersample":
            # Use RandomUnderSampler to balance classes
            # Separate known and unknown samples
            y_train = self.y[self.train_index]
            known_mask = y_train != -1
            unknown_mask = y_train == -1

            y_known = y_train.loc[known_mask]
            y_unknown = y_train.loc[unknown_mask]
            index_known = self.train_index[known_mask]
            index_unknown = self.train_index[unknown_mask]

            # Only balance if there are known samples and multiple classes
            if len(index_known) > 0 and len(y_known.unique()) > 1:
                rus = RandomUnderSampler(random_state=42)
                index_known_balanced, y_known_balanced = rus.fit_resample(index_known.reshape(-1, 1), y_known)

            self.index_train_balanced = np.concatenate([np.squeeze(index_known_balanced), index_unknown])

    def balance_dataset(self, *args: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, ...]:
        """
        Balance a number of datasets using the specified balancing method.

        Args:
            *args (pd.DataFrame): One or more DataFrames to be balanced.
            y (pd.Series): Target labels for balancing.

        Returns:
            Tuple[pd.DataFrame, ...]: Tuple of balanced DataFrames and balanced y.
        """

        known_mask = y != -1
        unknown_mask = y == -1
        y_known = y[known_mask]
        y_unknown = y[unknown_mask]

        if self.balance == "undersample":
            # Use RandomUnderSampler to balance classes
            args_balanced = [pd.DataFrame() for _ in args]
            rus = RandomUnderSampler(random_state=42)
            _, y_balanced = rus.fit_resample(y_known.to_frame(), y=y_known)
            resample_index = rus.sample_indices_
            for i, arg in enumerate(args):
                args_balanced[i] = pd.concat([arg.loc[known_mask].iloc[resample_index], arg.loc[unknown_mask]], ignore_index=True)
            return (*args_balanced, pd.concat((y_balanced, y_unknown), ignore_index=True))
        else:
            # No balancing applied
            return (*args, y)


    def _filter_time(self, data):
        """
        Filter data based on time constraints.
        
        This private method filters the input data to include only rows where
        the 'Tempo' column values fall within the specified time range [t_min, t_max].
        
        Args:
            data (pd.DataFrame): Input data containing a 'Tempo' column.
            
        Returns:
            pd.DataFrame: Filtered data with rows where t_min <= Tempo <= t_max.
        """

        return data[(data['Tempo'] >= self.t_min) & (data['Tempo'] <= self.t_max)]

    def update_labels(self, data, run_in_transition_min=None, run_in_transition_max=None):
        """
        Update the 'Amaciado' labels based on run-in transition criteria.
        
        This method applies a labeling strategy for semi-supervised learning:
        - Initially sets all labels to -1 (unknown)
        - Labels early data in the first test (N_ensaio=0) as not run-in (0)
        - Labels late data in the first test as run-in (1)
        - Labels all data from subsequent tests as run-in (1)
        
        Args:
            data (pd.DataFrame): Input data containing 'Tempo', 'N_ensaio', and 'Amaciado' columns.
            run_in_transition_min (float, optional): Override the minimum time threshold
                for run-in transition. If None, uses the instance's value.
            run_in_transition_max (float, optional): Override the maximum time threshold
                for run-in transition. If None, uses the instance's value.
                
        Returns:
            pd.DataFrame: Data with updated 'Amaciado' labels where:
                - -1: Unknown (unlabeled data for semi-supervised learning)
                - 0: Not run-in (early first test data)
                - 1: Run-in (late first test data and all subsequent test data)
        """

        if run_in_transition_min is not None:
            self.run_in_transition_min = run_in_transition_min
        if run_in_transition_max is not None:
            self.run_in_transition_max = run_in_transition_max

        data = data.assign(Amaciado = -1)  # Unknown label
        data.loc[(data.Tempo <= self.run_in_transition_min) & (data.N_ensaio == 0), 'Amaciado'] = 0 # Known not run-in at beginning of first test
        data.loc[(data.Tempo >= self.run_in_transition_max) & (data.N_ensaio == 0), 'Amaciado'] = 1 # Known run-in in first test
        data.loc[(data.N_ensaio > 0), 'Amaciado'] = 1  # Known run-in after first test

        return data
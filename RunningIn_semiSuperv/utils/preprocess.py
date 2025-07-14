from delayedsw import DelayedSlidingWindow, MovingAverageTransformer
import pandas as pd
import numpy as np

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
        tMin (float): Minimum time threshold for filtering data.
        tMax (float): Maximum time threshold for filtering data.
        run_in_transition_min (float): Minimum time for run-in transition labeling.
        run_in_transition_max (float): Maximum time for run-in transition labeling.
        X (pd.DataFrame): Processed feature matrix.
        y (pd.Series): Target labels.
        feature_names_in_ (list): Names of input features from the fitted data.
    """
    
    def __init__(self, window_size=1, delay=1, features=None, moving_average=1, tMin=0, tMax=np.inf, run_in_transition_min=5, run_in_transition_max = np.inf):
        """
        Initialize the RunInPreprocessor with specified parameters.
        
        Args:
            window_size (int, optional): Size of the sliding window. Defaults to 1.
            delay (int, optional): Delay space for the sliding window. Defaults to 1.
            features (list, optional): List of feature columns to transform. If None,
                all columns will be used. Defaults to None.
            moving_average (int, optional): Window size for moving average filter. Defaults to 1.
            tMin (float, optional): Minimum time threshold for data filtering. Defaults to 0.
            tMax (float, optional): Maximum time threshold for data filtering. Defaults to np.inf.
            run_in_transition_min (float, optional): Minimum time threshold for run-in 
                transition labeling. Defaults to 5.
            run_in_transition_max (float, optional): Maximum time threshold for run-in
                transition labeling. Defaults to np.inf.
        """

        self.window_size = window_size
        self.delay = delay
        self.features = features
        self.moving_average = moving_average
        self.tMin = tMin
        self.tMax = tMax
        self.run_in_transition_min = run_in_transition_min
        self.run_in_transition_max = run_in_transition_max
        self.X = pd.DataFrame()
        self.y = pd.Series()
        self.feature_names_in_ = []

        # TODO: initialize filter and windowing methods

    def set_filter_params(self, features=None, moving_average_window=1, tMin=0, tMax=np.inf, reset=True):
        """
        Set filtering parameters for the preprocessor.
        
        Args:
            features (list, optional): List of feature columns to use. If None,
                all columns will be used. Defaults to None.
            moving_average_window (int, optional): Window size for moving average filter.
                Defaults to 1.
            tMin (float, optional): Minimum time threshold for data filtering. Defaults to 0.
            tMax (float, optional): Maximum time threshold for data filtering. Defaults to np.inf.
            reset (bool, optional): Whether to reset stored data and feature names.
                Defaults to True.
        """

        self.features = features
        self.moving_average = moving_average_window
        self.tMin = tMin
        self.tMax = tMax

        if reset:
            self.feature_names_in_ = []
            self.X = pd.DataFrame()
            self.y = pd.Series()

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
            reset (bool, optional): Whether to reset stored data and feature names.
                Defaults to True.
        """

        self.window_size = window_size
        self.delay = delay
        self.features = features

        if reset:
            self.feature_names_in_ = []
            self.X = pd.DataFrame()
            self.y = pd.Series()

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
            reset (bool, optional): Whether to reset stored data and feature names.
                Defaults to True.
        """

        self.run_in_transition_min = run_in_transition_min
        self.run_in_transition_max = run_in_transition_max

        if reset:
            self.feature_names_in_ = []
            self.X = pd.DataFrame()
            self.y = pd.Series()

    def fit(self, data: pd.DataFrame):
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
            self._features = data.columns.tolist()
        else:
            self._features = self.features

        self.feature_names_in_ = data.columns.tolist()

        # Check if any features are missing from the data
        missing_features = [feat for feat in self._features if feat not in data.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the input data: {missing_features}")

        self.FilterTransformer = MovingAverageTransformer(window=self.moving_average, columns_to_transform=self._features)
        self.WindowTransformer = DelayedSlidingWindow(window_size=self.window_size, delay_space=self.delay,
                                              columns_to_transform=self._features,
                                              split_by=['Unidade', 'N_ensaio'], order_by=['Tempo'],
                                              include_order=True, include_split=True)

    def transform(self, data: pd.DataFrame):
        """
        Transform input data using the fitted preprocessor.
        
        This method applies the complete preprocessing pipeline:
        1. Filter data based on time constraints
        2. Generate labels for run-in transition detection
        3. Sort data by group and time columns
        4. Apply moving average filtering
        5. Apply sliding window transformation
        6. Remove NaN values and return processed features and labels
        
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

        if len(self.feature_names_in_) == 0:
            raise ValueError("The preprocessor has not been fitted yet. Please call fit() before transform().")

        group_columns = ['Unidade','N_ensaio']
        order_columns = ['Tempo']
        
        # Filter rows based on time
        data = self._filter_time(data)

        # Label the data
        data = self.update_labels(data)

        # Group by 'Unidade' and 'N_ensaio' and order by 'Tempo'
        data = data.sort_values(by=group_columns + order_columns)

        # Split X and y
        X, y = self._splitXY(data)

        X = self.FilterTransformer.fit_transform(X)

        X = self.WindowTransformer.fit_transform(X)

        # Join X and y and drop NaN values
        data = pd.concat([X, y], axis=1)
        data = data.dropna()
        data = data.reset_index(drop=True)

        # TODO: refactor code in order to avoid splitting X and y again after preprocessing

        # Split X and y again after preprocessing
        X, y = self._splitXY(data)

        self.X = X
        self.y = y

        return X, y

    def fit_transform(self, data):
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
        and uses all other columns as features.
        
        Args:
            data (pd.DataFrame): Input data containing both features and target.
            
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X (pd.DataFrame): Feature matrix (all columns except 'Amaciado')
                - y (pd.Series): Target variable ('Amaciado' column)
        """
        
        y = data.loc[:,'Amaciado']
        X = data.drop(['Amaciado'], axis= 1 )
        return X,y

    def _filter_time(self, data):
        """
        Filter data based on time constraints.
        
        This private method filters the input data to include only rows where
        the 'Tempo' column values fall within the specified time range [tMin, tMax].
        
        Args:
            data (pd.DataFrame): Input data containing a 'Tempo' column.
            
        Returns:
            pd.DataFrame: Filtered data with rows where tMin <= Tempo <= tMax.
        """

        return data[(data['Tempo'] >= self.tMin) & (data['Tempo'] <= self.tMax)]

    def update_labels(self, data, run_in_transition_min=None, run_in_transition_max = None) :
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

        data.loc[:, 'Amaciado'] = -1  # Unknown label
        data.loc[(data.Tempo <= self.run_in_transition_min) & (data.N_ensaio == 0), 'Amaciado'] = 0 # Known not run-in at beginning of first test
        data.loc[(data.Tempo >= self.run_in_transition_max) & (data.N_ensaio == 0), 'Amaciado'] = 1 # Known run-in in first test
        data.loc[(data.N_ensaio > 0) & (data.N_ensaio > 0), 'Amaciado'] = 1 # Known run-in after first test

        return data
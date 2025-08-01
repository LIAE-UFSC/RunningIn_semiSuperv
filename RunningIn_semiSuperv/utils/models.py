from sklearn import semi_supervised
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSCanonical
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Union, Any

class RunInSemiSupervisedModel(semi_supervised.SelfTrainingClassifier):
    """
    A semi-supervised learning model for run-in period detection.
    
    This class extends scikit-learn's SelfTrainingClassifier to provide a unified
    interface for different base classifiers in the context of run-in detection.
    It supports multiple classifier types and automatically wraps them in a
    self-training framework for semi-supervised learning.
    
    The model is designed to work with partially labeled data where:
    - Some samples are labeled (run-in vs. not run-in)
    - Some samples are unlabeled (unknown state)
    - The self-training approach iteratively labels confident predictions
    
    Attributes:
        All attributes inherited from sklearn.semi_supervised.SelfTrainingClassifier
        
    Supported strings for `classifier`:
        - LogisticRegression: Logistic Regression (default)
        - KNeighborsClassifier: K-Nearest Neighbors
        - SGDClassifier: Stochastic Gradient Descent
        - DecisionTreeClassifier: Decision Tree
        - PLSCanonical: Partial Least Squares
        - GaussianNB: Gaussian Naive Bayes
        - MLPClassifier: Multi-layer Perceptron
        - RandomForestClassifier: Random Forest
        - GaussianProcessRegressor: Gaussian Process
    Aside from the string names, you can also pass a classifier class directly.

    Classifier arguments can be passed as a dictionary to `classifier_args`.

    Example:
        >>> model = RunInSemiSupervisedModel(classifier="RandomForestClassifier",
        ...                                  classifier_args={"n_estimators": 100},
        ...                                  threshold=0.75)
        >>> model.fit(X_train, y_train)  # y_train can contain -1 for unlabeled
        >>> predictions = model.predict(X_test)
    """
    def __init__(self, classifier: str = "LogisticRegression", classifier_args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Initialize the RunInSemiSupervisedModel.
        
        Args:
            classifier (str or type, optional): The base classifier to use. Can be either:
                - A string specifying one of the supported classifier names
                - A class that will be instantiated with classifier_args
                Defaults to "LogisticRegression".
            classifier_args (dict, optional): Arguments to pass to the base classifier
                constructor. Defaults to None (empty dict).
            **kwargs: Additional arguments passed to the SelfTrainingClassifier
                constructor. Common parameters include:
                - threshold (float): Threshold for confident predictions (default: 0.75)
                - criterion (str): Selection criterion ('threshold' or 'k_best')
                - k_best (int): Number of samples to label in each iteration
                - max_iter (int): Maximum number of iterations (default: 10)
                - verbose (bool): Enable verbose output
                
        Raises:
            ValueError: If the classifier string is not recognized.
            TypeError: If the classifier is neither a string nor a class.
        """
        # Store instance attributes for get_params compatibility
        self.classifier = classifier
        self.classifier_args = classifier_args if classifier_args is not None else {}
        
        estimator = self._select_model(classifier, **self.classifier_args)
        super().__init__(estimator=estimator, **kwargs)

    def _select_model(self, classifier: str, **kwargs: Any) -> Any:
        """
        Select and instantiate the appropriate base classifier.
        
        This private method creates the base classifier instance based on the
        provided classifier specification. It supports both string-based
        classifier selection and direct class instantiation.
        
        Args:
            classifier (str or type): The classifier specification. If string,
                must be one of the supported classifier names. If class,
                will be instantiated with the provided kwargs.
            **kwargs: Arguments passed to the classifier constructor.
            
        Returns:
            sklearn estimator: An instantiated classifier that will be used as the
                base estimator for the SelfTrainingClassifier.
                
        Raises:
            ValueError: If the classifier string is not in the list of supported
                classifiers.
            TypeError: If the classifier is neither a string nor a class.
            
        Supported Classifier Strings:
            - "LogisticRegression": Logistic Regression classifier
            - "KNeighborsClassifier": K-Nearest Neighbors classifier
            - "SGDClassifier": Stochastic Gradient Descent classifier
            - "DecisionTreeClassifier": Decision Tree classifier
            - "PLSCanonical": Partial Least Squares Canonical classifier
            - "GaussianNB": Gaussian Naive Bayes classifier
            - "MLPClassifier": Multi-layer Perceptron classifier
            - "RandomForestClassifier": Random Forest classifier
            - "GaussianProcessRegressor": Gaussian Process Regressor
        """

        supported_classifiers = [
            "LogisticRegression",
            "SGDClassifier",
            "DecisionTreeClassifier",
            "PLSCanonical",
            "KNeighborsClassifier",
            "LinearSVM",
            "RBFSVM",
            "GaussianProcess",
            "RandomForest",
            "NeuralNet",
            "AdaBoost",
            "NaiveBayes",
            "QDA",
        ]

        # Classifier can be either a string or a class
        if isinstance(classifier, str):
            if classifier == "LogisticRegression":
                return LogisticRegression(random_state = 42, **kwargs)
            elif classifier == "DecisionTreeClassifier":
                return DecisionTreeClassifier(random_state = 42, **kwargs)
            elif classifier == "PLSCanonical":
                return PLSCanonical(**kwargs)
            elif classifier == "KNeighborsClassifier":
                return KNeighborsClassifier(**kwargs)
            elif classifier == "LinearSVM":
                return SVC(random_state = 42, kernel='linear', probability=True, **kwargs)
            elif classifier == "RBFSVM":
                return SVC(random_state = 42, kernel=RBF(), probability=True, **kwargs)
            elif classifier == "GaussianProcess":
                kernel = kwargs.pop('kernel', "RBF")
                if kernel == "RBF":
                    return GaussianProcessClassifier(kernel=RBF(), random_state=42, **kwargs)
                elif kernel == "Matern":
                    return GaussianProcessClassifier(kernel=Matern(), random_state=42, **kwargs)
                elif kernel == "RationalQuadratic":
                    return GaussianProcessClassifier(kernel=RationalQuadratic(), random_state=42, **kwargs)
                elif kernel == "ExpSineSquared":
                    return GaussianProcessClassifier(kernel=ExpSineSquared(), random_state=42, **kwargs)
                else:
                    raise ValueError(f"Unsupported Gaussian Process kernel: {kernel}. "
                                     f"Supported kernels are: RBF, Matern, RationalQuadratic, ExpSineSquared.")
            elif classifier == "RandomForest":
                return RandomForestClassifier(random_state = 42, **kwargs)
            elif classifier == "NeuralNet":
                return MLPClassifier(random_state = 42, **kwargs)
            elif classifier == "AdaBoost":
                return AdaBoostClassifier(random_state = 42, **kwargs)
            elif classifier == "NaiveBayes":
                return GaussianNB(**kwargs)
            elif classifier == "QDA":
                return QuadraticDiscriminantAnalysis(**kwargs)
            else:
                raise ValueError(f"Classifier '{classifier}' is not supported. "
                                 f"Supported classifiers are: {', '.join(supported_classifiers)}")
        elif isinstance(classifier, type):
            # If classifier is a class, instantiate it with the provided arguments
            return classifier(**kwargs)
        else:
            raise TypeError("Classifier must be a string or a class.")


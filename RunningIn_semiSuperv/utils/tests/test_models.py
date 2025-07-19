import unittest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import semi_supervised

from RunningIn_semiSuperv.utils.models import RunInSemiSupervisedModel


class MockClassifier(BaseEstimator, ClassifierMixin):
    """Mock classifier for testing purposes."""
    
    def __init__(self, param1=1, param2='default'):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self, X, y):
        self.classes_ = np.unique(y[y >= 0])  # Exclude -1 labels
        return self
        
    def predict(self, X):
        # Simple mock prediction
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        # Simple mock probabilities
        n_classes = len(getattr(self, 'classes_', [0, 1]))
        proba = np.ones((len(X), n_classes)) / n_classes
        return proba


class TestRunInSemiSupervisedModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample semi-supervised dataset
        np.random.seed(42)
        n_samples = 100
        n_features = 4
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Create semi-supervised labels: some labeled (0, 1), some unlabeled (-1)
        y = np.random.choice([0, 1], size=n_samples)
        # Make some labels unknown (-1) for semi-supervised learning
        unlabeled_indices = np.random.choice(n_samples, size=30, replace=False)
        y[unlabeled_indices] = -1
        
        self.X_train = X
        self.y_train = y
        
        # Create test data (all labeled)
        self.X_test = np.random.randn(20, n_features)
        self.y_test = np.random.choice([0, 1], size=20)
        
        # Create minimal dataset
        self.X_minimal = np.array([[1, 2], [3, 4], [5, 6]])
        self.y_minimal = np.array([0, 1, -1])  # One unlabeled
        
    def test_default_initialization(self):
        """Test that default initialization works and creates expected structure."""
        model = RunInSemiSupervisedModel()
        
        # Should be a SelfTrainingClassifier
        self.assertIsInstance(model, semi_supervised.SelfTrainingClassifier)
        
        # Should have default parameters
        self.assertEqual(getattr(model, 'threshold', None), 0.75)
        self.assertEqual(getattr(model, 'max_iter', None), 10)
        
    def test_string_classifier_types(self):
        """Test that all supported string classifier types can be instantiated."""
        supported_classifiers = [
            "LogisticRegression",
            "KNeighborsClassifier", 
            "SGDClassifier",
            "DecisionTreeClassifier",
            "GaussianNB",
            "MLPClassifier",
            "RandomForestClassifier"
        ]
        
        for classifier_name in supported_classifiers:
            with self.subTest(classifier=classifier_name):
                model = RunInSemiSupervisedModel(classifier=classifier_name)
                self.assertIsInstance(model, semi_supervised.SelfTrainingClassifier)
                
    def test_classifier_arguments_passing(self):
        """Test that classifier arguments are properly passed to base estimator."""
        # Test with a classifier that has distinctive parameters
        model = RunInSemiSupervisedModel(
            classifier="RandomForestClassifier",
            classifier_args={"n_estimators": 50, "random_state": 42}
        )
        
        # Model should be created successfully
        self.assertIsInstance(model, semi_supervised.SelfTrainingClassifier)
        
        # Base estimator should have the specified parameters
        base_est = getattr(model, 'estimator', None)
        self.assertIsNotNone(base_est)
        self.assertEqual(getattr(base_est, 'n_estimators', None), 50)
        self.assertEqual(getattr(base_est, 'random_state', None), 42)
        
    def test_self_training_parameters(self):
        """Test that SelfTrainingClassifier parameters are properly set."""
        model = RunInSemiSupervisedModel(
            classifier="LogisticRegression",
            threshold=0.8,
            max_iter=5
        )
        
        self.assertEqual(getattr(model, 'threshold', None), 0.8)
        self.assertEqual(getattr(model, 'max_iter', None), 5)
        
    def test_invalid_classifier_string(self):
        """Test that invalid classifier strings raise appropriate errors."""
        with self.assertRaises(ValueError) as context:
            RunInSemiSupervisedModel(classifier="NonExistentClassifier")
            
        self.assertIn("Unknown classifier", str(context.exception))
        
    def test_basic_fit_predict_workflow(self):
        """Test the complete fit-predict workflow works."""
        model = RunInSemiSupervisedModel(
            classifier="LogisticRegression",
            classifier_args={"max_iter": 1000, "random_state": 42},
            threshold=0.7,
            max_iter=3
        )
        
        # Should fit without errors
        model.fit(self.X_train, self.y_train)
        
        # Should have classes after fitting
        self.assertTrue(hasattr(model, 'classes_'))
        
        # Should predict valid outputs
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        
    def test_probability_predictions(self):
        """Test that probability predictions work correctly."""
        model = RunInSemiSupervisedModel(
            classifier="LogisticRegression",
            classifier_args={"max_iter": 1000, "random_state": 42},
            max_iter=2
        )
        
        model.fit(self.X_train, self.y_train)
        probabilities = model.predict_proba(self.X_test)
        
        # Check probability properties
        self.assertEqual(probabilities.shape[0], len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2)  # Binary classification
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        
    def test_multiple_classifiers_functionality(self):
        """Test that different classifiers all work in the framework."""
        classifiers_to_test = [
            ("LogisticRegression", {"max_iter": 500}),
            ("KNeighborsClassifier", {"n_neighbors": 3}),
            ("RandomForestClassifier", {"n_estimators": 10, "random_state": 42})
        ]
        
        for classifier_name, classifier_args in classifiers_to_test:
            with self.subTest(classifier=classifier_name):
                model = RunInSemiSupervisedModel(
                    classifier=classifier_name,
                    classifier_args=classifier_args,
                    max_iter=2
                )
                
                # Should fit and predict without errors
                model.fit(self.X_train, self.y_train)
                predictions = model.predict(self.X_test)
                
                # Basic checks
                self.assertEqual(len(predictions), len(self.X_test))
                self.assertTrue(all(pred in [0, 1] for pred in predictions))
                
    def test_fully_labeled_data_handling(self):
        """Test that the model works with fully labeled data."""
        y_labeled = np.random.choice([0, 1], size=len(self.X_train))
        
        model = RunInSemiSupervisedModel(
            classifier="LogisticRegression",
            classifier_args={"max_iter": 1000, "random_state": 42}
        )
        
        # Should work without unlabeled data
        model.fit(self.X_train, y_labeled)
        predictions = model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        
    def test_minimal_dataset_handling(self):
        """Test that the model works with minimal datasets."""
        model = RunInSemiSupervisedModel(
            classifier="LogisticRegression",
            classifier_args={"max_iter": 1000, "random_state": 42},
            max_iter=1
        )
        
        model.fit(self.X_minimal, self.y_minimal)
        predictions = model.predict(self.X_minimal)
        
        self.assertEqual(len(predictions), len(self.X_minimal))


if __name__ == '__main__':
    unittest.main()

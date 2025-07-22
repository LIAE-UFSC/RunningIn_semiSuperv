from utils import RunInSemiSupervised
import logging
import optuna
from scipy.stats import gmean
from pathlib import Path
import sys
import numpy as np
import warnings
import multiprocessing as mp
import time
import subprocess
import signal
import os

warnings.filterwarnings("ignore", category=UserWarning)

compressor_model = "a"
n_processes = 3
n_tests = 500
max_init_samples = 180  # Maximum total window size for optimization (moving_average - 1 + (window_size - 1) * delay)

def MCC_score_from_confusion_matrix(cm):
    """
    Calculate Matthews Correlation Coefficient (MCC) from confusion matrix.
    
    Parameters:
    cm (array-like): Confusion matrix
    
    Returns:
    float: MCC score
    """
    tn, fp, fn, tp = cm.ravel()
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator if denominator != 0 else 0

class OptimizationRunIn():

    supported_classifiers = [
            "LogisticRegression",
            "DecisionTreeClassifier",
 #           "PLSCanonical", # Doesn't have probability estimates
            "KNeighborsClassifier",
            "LinearSVM",
            "RBFSVM",
  #          "GaussianProcess",
            "RandomForest",
            "NeuralNet",
            "AdaBoost",
            "NaiveBayes",
            "QDA",
        ]

    def __init__(self, classifier = "LogisticRegression", compressor_model="a"):
        if classifier not in self.supported_classifiers:
            raise ValueError(f"Classifier '{classifier}' not supported. Choose from: {self.supported_classifiers}")
        self.classifier = classifier
        self.parameters = {}
        self.compressor_model = compressor_model

    def select_classifier(self, trial):
        if self.classifier == "LogisticRegression":
            self.classifier_parameters = {
                "C": trial.suggest_float('C', 1e-4, 1e2, log=True),
                "max_iter": 1000,
            }
        elif self.classifier == "DecisionTreeClassifier":
            self.classifier_parameters = {
                "max_depth": trial.suggest_int('max_depth', 3, 50),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 200),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 100),
                "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif self.classifier == "PLSCanonical":
            self.classifier_parameters = {
                "n_components": trial.suggest_int('n_components', 1, self.parameters["window_size"]),
                "scale": trial.suggest_categorical('scale', [True, False]),
                "max_iter": 1000
            }
        elif self.classifier == "KNeighborsClassifier":
            self.classifier_parameters = {
                "n_neighbors": trial.suggest_int('n_neighbors', 3, 100),
                "weights": trial.suggest_categorical('weights', ['uniform', 'distance']),
                "metric": trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                "p": trial.suggest_int('p', 1, 3) if trial.params.get('metric') == 'minkowski' else 2
            }
        elif self.classifier == "LinearSVM":
            self.classifier_parameters = {
                "C": trial.suggest_float('C', 1e-5, 1e3, log=True),
                "max_iter": 5000
            }
        elif self.classifier == "RBFSVM":
            self.classifier_parameters = {
                "C": trial.suggest_float('C', 1e-5, 1e3, log=True),
                "gamma": trial.suggest_float('gamma', 1e-10, 1e1, log=True),
                "max_iter": 5000
            }
        elif self.classifier == "GaussianProcess":
            self.classifier_parameters = {
                # Left out kernel selection for now because of LinAlg errors
                # "kernel": trial.suggest_categorical('kernel', ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared"]),
                "n_restarts_optimizer": trial.suggest_int('n_restarts_optimizer', 0, 10),
                "max_iter_predict": 100
            }
        elif self.classifier == "RandomForest":
            self.classifier_parameters = {
                "n_estimators": trial.suggest_int('n_estimators', 10, 200),
                "max_depth": trial.suggest_int('max_depth', 3, 20),
                "min_samples_split": trial.suggest_int('min_samples_split', 2, 20),
                "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 10),
                "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                "bootstrap": trial.suggest_categorical('bootstrap', [True, False])
            }
        elif self.classifier == "NeuralNet":
            hidden_layer_sizes = tuple([trial.suggest_int(f'n_units_l{i}', 10, 200) 
                                       for i in range(trial.suggest_int('n_layers', 1, 3))])
            self.classifier_parameters = {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                "alpha": trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                "learning_rate": trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                "max_iter": 600
            }
        elif self.classifier == "AdaBoost":
            self.classifier_parameters = {
                "n_estimators": trial.suggest_int('n_estimators', 10, 200),
                "learning_rate": trial.suggest_float('learning_rate', 0.01, 2.0),
            }
        elif self.classifier == "NaiveBayes":
            self.classifier_parameters = {
                "var_smoothing": trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
            }
        elif self.classifier == "QDA":
            self.classifier_parameters = {
                "reg_param": trial.suggest_float('reg_param', 0.0, 1.0),
                "store_covariance": trial.suggest_categorical('store_covariance', [True, False])
            }
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier}")

        return self.classifier_parameters

    def create_runin_semi_supervised_model(self, trial):
        """
        Create a RunInSemiSupervisedModel instance with the selected classifier and parameters.
        
        Returns:
            RunInSemiSupervisedModel: Configured model instance.
        """
        
        # Define hyperparameters to optimize
        self.parameters = {
            'window_size': trial.suggest_int('window_size', 1, 150),
            'moving_average': trial.suggest_int('moving_average', 1, 30),
            'scale': trial.suggest_categorical('scale', [True, False]),
            'balance': trial.suggest_categorical('balance', ['undersample', 'none'])
        }

        max_delay = (max_init_samples - (self.parameters['moving_average'] - 1))// (self.parameters['window_size'] - 1)
        self.parameters["delay"] = trial.suggest_int('delay', 1, max_delay)

        self.select_classifier(trial = trial)
        threshold = trial.suggest_float('threshold', 0.05, 0.99)
        features = trial.suggest_categorical('features',[
                'CorrenteRMS',
                'CorrenteCurtose',
                'CorrenteAssimetria',
                'CorrenteForma',
                'CorrenteTHD',
                'CorrentePico',
                'CorrenteCrista',
                'CorrenteVariancia',
                'CorrenteDesvio',
                'VibracaoCalotaInferiorRMS',
                'VibracaoCalotaInferiorCurtose',
                'VibracaoCalotaInferiorAssimetria',
                'VibracaoCalotaInferiorForma',
                'VibracaoCalotaInferiorPico',
                'VibracaoCalotaInferiorCrista',
                'VibracaoCalotaInferiorVariancia',
                'VibracaoCalotaInferiorDesvio',
                'VibracaoCalotaSuperiorRMS',
                'VibracaoCalotaSuperiorCurtose',
                'VibracaoCalotaSuperiorAssimetria',
                'VibracaoCalotaSuperiorForma',   
                'VibracaoCalotaSuperiorPico',
                'VibracaoCalotaSuperiorCrista',
                'VibracaoCalotaSuperiorVariancia',
                'VibracaoCalotaSuperiorDesvio',
                'Vazao'])
        
        
        # Initialize the semi-supervised learning pipeline with trial parameters
        self.model = RunInSemiSupervised(
            **(self.parameters),
            compressor_model=compressor_model,
            classifier=self.classifier,
            semisupervised_params={
                'threshold': threshold,
                'criterion': 'threshold',
                'max_iter': 1000
            },
            classifier_params=self.classifier_parameters,
            test_split=["a3"],
            features=[features],
        )

    def objective(self, trial):
        self.create_runin_semi_supervised_model(trial)

        if (self.model.window_size - 1)*self.model.delay + (self.model.moving_average - 1) > max_init_samples:
            print(f"Skipping trial due to excessive window size: {self.model.window_size}, delay: {self.model.delay}, moving average: {self.model.moving_average}")
            return None, None

        # Train the model
        self.model.fit()

        result = self.model.cross_validate()

        labeled_percentage = gmean(result["percent_labeled"])

        joined_confusion_matrix = np.array(result["confusion_matrix"]).sum(axis=0)

        mcc = MCC_score_from_confusion_matrix(joined_confusion_matrix)
        
        return mcc, labeled_percentage


def start_optuna_dashboard(storage_name, port=8080):
    """
    Start Optuna dashboard in a subprocess.
    
    Args:
        storage_name (str): Path to the database file
        port (int): Port number for the dashboard
    
    Returns:
        subprocess.Popen: Dashboard process object
    """
    try:
        print(f"Starting Optuna dashboard at http://localhost:{port}")
        
        # Handle different operating systems for process group creation
        if os.name == 'nt':  # Windows
            dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_name, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix-like systems (Linux, macOS)
            dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_name, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group for easier termination
            )
        return dashboard_process
    except FileNotFoundError:
        print("Warning: optuna-dashboard not found. Please install with: pip install optuna-dashboard")
        return None
    except Exception as e:
        print(f"Warning: Could not start Optuna dashboard: {e}")
        return None


def stop_optuna_dashboard(dashboard_process):
    """
    Stop the Optuna dashboard process.
    
    Args:
        dashboard_process (subprocess.Popen): Dashboard process object
    """
    if dashboard_process is not None:
        try:
            if os.name == 'nt':  # Windows
                # On Windows, terminate the process directly
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                print("Optuna dashboard stopped successfully")
            else:  # Unix-like systems (Linux, macOS)
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(dashboard_process.pid), signal.SIGTERM)
                dashboard_process.wait(timeout=5)  # Wait up to 5 seconds for graceful shutdown
                print("Optuna dashboard stopped successfully")
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown failed
            if os.name == 'nt':  # Windows
                dashboard_process.kill()
                print("Optuna dashboard force killed")
            else:  # Unix-like systems
                os.killpg(os.getpgid(dashboard_process.pid), signal.SIGKILL)
                print("Optuna dashboard force killed")
        except Exception as e:
            print(f"Warning: Could not stop dashboard process: {e}")


if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    for classifier_type in OptimizationRunIn.supported_classifiers:
        dashboard_process = None
        processes = []
        
        try:
            # Create an instance of the optimization class
            optimizer = OptimizationRunIn(classifier=classifier_type, compressor_model=compressor_model)
            
            # Create Optuna study
            study_name = f"RunIn_{classifier_type}_{compressor_model}"  # Unique
            results_dir = Path(__file__).resolve().parent.parent / "Results"
            results_dir.mkdir(exist_ok=True)
            storage_name = f"sqlite:///{results_dir.as_posix()}/RunIn_{classifier_type}.db"
            
            # Create Optuna study
            study = optuna.create_study(directions=["maximize", "maximize"], 
                                        study_name=study_name, 
                                        storage=storage_name, 
                                        load_if_exists=True)
        
            # Start Optuna dashboard
            dashboard_port = 8080
            dashboard_process = start_optuna_dashboard(storage_name, port=dashboard_port)
            
            start_time = time.time()

            # Calculate number of studies per process
            N_studies_left = n_tests
            n_studies_process = round(N_studies_left / n_processes)

            for i in range(n_processes):  # Number of optimization trials
                N_studies_left -= n_studies_process
                if i == n_processes - 1: # Last process takes remaining trials
                    n_studies_process = N_studies_left
                p = mp.Process(target=study.optimize, args=(optimizer.objective,), kwargs={'n_trials': n_studies_process, 'n_jobs': 1})
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            elapsed = time.time() - start_time
            print(f"Tests finished for classifier {classifier_type}. Elapsed time: {elapsed:.2f} seconds")
            
        except KeyboardInterrupt:
            print(f"\nKeyboard interrupt received. Cleaning up processes for {classifier_type}...")
            # Terminate all optimization processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()  # Force kill if terminate didn't work
            raise  # Re-raise to exit the main loop
            
        except Exception as e:
            print(f"Error occurred during optimization for {classifier_type}: {e}")
            # Terminate all optimization processes
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()  # Force kill if terminate didn't work
            
        finally:
            # Always stop the dashboard, regardless of success or failure
            if dashboard_process is not None:
                stop_optuna_dashboard(dashboard_process)
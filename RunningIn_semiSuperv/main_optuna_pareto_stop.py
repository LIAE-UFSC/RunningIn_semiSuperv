"""
Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning

Database Configuration:
1. SQLite (default): Set USE_POSTGRES = False
   - No additional setup required
   - Files saved to Results/ directory

2. PostgreSQL: Set USE_POSTGRES = True
   - Requires PostgreSQL server setup
   - Install dependencies: pip install psycopg2-binary
   - Update POSTGRES_CONFIG with your database credentials
   
PostgreSQL Setup:
1. Install PostgreSQL server
2. Install Python PostgreSQL adapter:
   ```bash
   pip install psycopg2-binary
   # OR if you have compilation issues:
   pip install psycopg2
   ```
3. Create database and user:
   ```sql
   CREATE DATABASE optuna_db;
   CREATE USER optuna_user WITH PASSWORD 'optuna_password';
   GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;
   ```
4. Update POSTGRES_CONFIG dictionary below

Common Issues:
- "Failed to import DB access module": Install psycopg2-binary
- Connection refused: Check PostgreSQL server is running
- Authentication failed: Verify username/password in POSTGRES_CONFIG
"""

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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for RunIn Semi-Supervised Learning")
    parser.add_argument("--compressor_model", type=str, default="a", choices=["a", "b", "all"],
                        help="Compressor model to use: 'a', 'b', or 'all'")
    parser.add_argument("--n_processes", type=int, default=4, help="Number of parallel processes for optimization")
    parser.add_argument("--max_init_samples", type=int, default=180, help="Maximum total window size for optimization")
    parser.add_argument("--auto_broadcast", action="store_true", help="Automatically start Optuna dashboard")
    parser.add_argument("--pareto_stop", type=int, default=-1, help="Number of trials to stop after not advancing the Pareto front", required=True)
    parser.add_argument("--max_iter", type=int, default=100000, help="Maximum iterations for optimization")
    return parser.parse_args()

args = parse_args()
compressor_model = args.compressor_model
n_processes = args.n_processes
max_init_samples = args.max_init_samples
auto_broadcast = args.auto_broadcast
pareto_stop = args.pareto_stop
max_iter = args.max_iter

# Database configuration
USE_POSTGRES = True  # Set to True to use PostgreSQL, False for SQLite
POSTGRES_CONFIG = {
    'host': '100.107.250.125',
    'port': 5432,
    'database': 'optuna_db',
    'user': 'optuna_user',
    'password': 'optuna_password'
}

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

def get_database_url(classifier_type):
    """
    Get the database URL for either SQLite or PostgreSQL.
    
    Args:
        classifier_type (str): Name of the classifier
    
    Returns:
        str: Database URL string
    """
    if USE_POSTGRES:
        # Check if psycopg2 is available
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "PostgreSQL support requires psycopg2. Install it with:\n"
                "pip install psycopg2-binary\n"
                "Or set USE_POSTGRES = False to use SQLite instead."
            )
        
        # PostgreSQL URL format
        return (f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
                f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}"
                f"/{POSTGRES_CONFIG['database']}")
    else:
        # SQLite URL format
        results_dir = Path(__file__).resolve().parent.parent / "Results"
        results_dir.mkdir(exist_ok=True)
        return f"sqlite:///{results_dir.as_posix()}/RunIn_{classifier_type}.db"

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
            max_window = self.parameters["window_size"] if self.parameters["pca"] == 0 else self.parameters["pca"] 
            self.classifier_parameters = {
                "n_components": trial.suggest_int('n_components', 1, max_window),
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
        if self.parameters["window_size"] == 1: # Delay does not make sense for window size 1
            self.parameters["delay"] = 1
        else:
            max_delay = (max_init_samples - (self.parameters['moving_average'] - 1))// (self.parameters['window_size'] - 1)
            self.parameters["delay"] = trial.suggest_int('delay', 1, max_delay)
        
        # PCA dimensionality reduction: suggest number of components from 0 to window_size
        # 0 = no PCA, >0 = apply PCA with specified number of components
        # Limited by window_size since PCA cannot have more components than features
        pca = trial.suggest_categorical('pca', [True, False])
        if pca:
            self.parameters['pca'] = trial.suggest_int('n_pca', 1, self.parameters["window_size"])
        else:
            self.parameters['pca'] = 0

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
        
        
        if compressor_model == "a":
            test_compressor = ["a3"]
        elif compressor_model == "b":
            test_compressor = ["b10"]
        else:
            test_compressor = ["a3","b10"]

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
            test_split=test_compressor,
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
        print(f"Starting Optuna dashboard at http://127.0.0.1:{port}")
        
        # Handle different operating systems for process group creation
        if os.name == 'nt':  # Windows
            dashboard_process = subprocess.Popen(
                ["optuna-dashboard", storage_name, "--host 0.0.0.0 --port", str(port), ],
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
    mp.set_start_method("spawn", force=True)

    for classifier_type in OptimizationRunIn.supported_classifiers:
        dashboard_process = None
        processes = []
        
        try:
            # Create an instance of the optimization class
            optimizer = OptimizationRunIn(classifier=classifier_type, compressor_model=compressor_model)
            
            # Get database URL (SQLite or PostgreSQL)
            storage_name = get_database_url(classifier_type)
            
            # Create Optuna study name
            study_name = f"RunIn_{classifier_type}_{compressor_model}"
            
            # Configure storage with timeout settings
            try:
                if USE_POSTGRES:
                    # PostgreSQL storage with connection pooling and timeout
                    storage = optuna.storages.RDBStorage(
                        url=storage_name,
                        engine_kwargs={
                            "pool_size": 20,
                            "max_overflow": 0,
                            "pool_pre_ping": True,
                            "pool_recycle": 300,
                            "connect_args": {"connect_timeout": 60}
                        }
                    )
                else:
                    # SQLite storage with timeout
                    storage = optuna.storages.RDBStorage(
                        url=storage_name, 
                        engine_kwargs={"connect_args": {"timeout": 100}}
                    )
            except Exception as e:
                if "Failed to import DB access module" in str(e):
                    raise ImportError(
                        f"Database connection failed for {classifier_type}. "
                        f"If using PostgreSQL, install psycopg2-binary:\n"
                        f"pip install psycopg2-binary\n"
                        f"Original error: {e}"
                    )
                else:
                    raise e
            
            # Create Optuna study
            study = optuna.create_study(
                directions=["maximize", "maximize"], 
                study_name=study_name, 
                storage=storage, 
                load_if_exists=True
            )
            
            if auto_broadcast:        
                # Start Optuna dashboard
                dashboard_port = 8081
                dashboard_process = start_optuna_dashboard(storage_name, port=dashboard_port)
            else:
                dashboard_process = None
            
            start_time = time.time()

            last_pareto = study.best_trials[-1].number if study.best_trials else 0

            n_done = len(study.trials)
            if (n_done - last_pareto) >= pareto_stop:
                print(f"Already reached Pareto front stop condition for {classifier_type}. No further optimization needed.")
                continue
            elif n_done >= max_iter:
                print(f"Already reached maximum iterations ({max_iter}) for {classifier_type}. No further optimization needed.")
                continue
            else:
                print(f"Running iteration of optimizer, aiming for {pareto_stop} tests after advancing.")

            while ((n_done - last_pareto) < pareto_stop) and (n_done < max_iter):
                if (n_done + pareto_stop) >= max_iter:
                    n_tests = max_iter - n_done
                    print(f"Approaching maximum iterations ({max_iter}). Running only {n_tests} more trials.")
                else:
                    n_tests = pareto_stop - (n_done - last_pareto)
                    print(f"Last Pareto front advance has been {n_done - last_pareto} trials ago. Test again after {n_tests} trials.")

                # Calculate number of studies per process
                N_studies_left = n_tests
                n_studies_process = round(N_studies_left / n_processes)

                for i in range(n_processes):  # Number of optimization trials
                    if i == n_processes - 1: # Last process takes remaining trials
                        n_studies_process = N_studies_left
                    else:
                        N_studies_left -= n_studies_process
                    p = mp.Process(target=study.optimize, args=(optimizer.objective,), kwargs={'n_trials': n_studies_process, 'n_jobs': 1})
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                last_pareto = study.best_trials[-1].number if study.best_trials else 0
                n_done = len(study.trials)

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
from scipy.stats import gmean
from .generator import RunInSemiSupervised
import numpy as np
import optuna
import multiprocessing as mp
from pathlib import Path

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
    _postgres_default_config = {
    'host': '100.107.250.125',
    'port': 5432,
    'database': 'optuna_db',
    'user': 'optuna_user',
    'password': 'optuna_password'
    }

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

    def __init__(self, classifier = "LogisticRegression", compressor_model="a", max_init_samples = 180, use_postgres = True):
        if classifier not in self.supported_classifiers:
            raise ValueError(f"Classifier '{classifier}' not supported. Choose from: {self.supported_classifiers}")
        self.classifier = classifier
        self.parameters = {}
        self.compressor_model = compressor_model
        self.max_init_samples = max_init_samples
        self.use_postgres = use_postgres
        self.postgres_config = self._postgres_default_config

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
            max_delay = (self.max_init_samples - (self.parameters['moving_average'] - 1))// (self.parameters['window_size'] - 1)
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
        
        
        if self.compressor_model == "a":
            test_compressor = ["a3"]
        elif self.compressor_model == "b":
            test_compressor = ["b10"]
        else:
            test_compressor = ["a3","b10"]

        # Initialize the semi-supervised learning pipeline with trial parameters
        self.model = RunInSemiSupervised(
            **(self.parameters),
            compressor_model=self.compressor_model,
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

        if (self.model.window_size - 1)*self.model.delay + (self.model.moving_average - 1) > self.max_init_samples:
            print(f"Skipping trial due to excessive window size: {self.model.window_size}, delay: {self.model.delay}, moving average: {self.model.moving_average}")
            return None, None

        # Train the model
        self.model.fit()

        result = self.model.cross_validate()

        labeled_percentage = gmean(result["percent_labeled"])

        joined_confusion_matrix = np.array(result["confusion_matrix"]).sum(axis=0)

        mcc = MCC_score_from_confusion_matrix(joined_confusion_matrix)
        
        return mcc, labeled_percentage

    def get_database_url(self):
        """
        Get the database URL for either SQLite or PostgreSQL.
        
        Args:
            classifier_type (str): Name of the classifier
        
        Returns:
            str: Database URL string
        """
        if self.use_postgres:
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
            return (f"postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}"
                    f"@{self.postgres_config['host']}:{self.postgres_config['port']}"
                    f"/{self.postgres_config['database']}")
        else:
            # SQLite URL format
            results_dir = Path(__file__).resolve().parent.parent / "Results"
            results_dir.mkdir(exist_ok=True)
            return f"sqlite:///{results_dir.as_posix()}/RunIn_{self.classifier}.db"

    def init_study(self):
        """
        Initialize an Optuna study with the given classifier type and compressor model.
        
        Parameters:
        classifier_type (str): Type of classifier to optimize.
        compressor_model (str): Compressor model to use.
        storage_name (str): Name of the storage for the study.
        
        Returns:
        optuna.Study: Configured Optuna study.
        """
        storage_name = self.get_database_url()  # Use PostgreSQL by default
        
        # Configure storage with timeout settings
        try:
            if self.use_postgres:
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
                    f"Database connection failed. "
                    f"If using PostgreSQL, install psycopg2-binary:\n"
                    f"pip install psycopg2-binary\n"
                    f"Original error: {e}"
                )
            else:
                raise e
        
        # Create Optuna study name
        study_name = f"RunIn_{self.classifier}_{self.compressor_model}"

        # Create Optuna study
        study = optuna.create_study(
            directions=["maximize", "maximize"], 
            study_name=study_name, 
            storage=storage, 
            load_if_exists=True
        )
        
        return study

    def run_optimization(self, n_trials: int, n_jobs:int = 1, study: optuna.Study|None = None):
        if study is None:
            study = self.init_study()

        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs)


    def multithreading_optimization(self, n_trials: int, n_processes: int = 2, n_jobs: int = 1, method: str = "spawn", connect_db_allproc: bool = False):
        
        if n_processes < 1:
            raise ValueError("Number of processes must be at least 1")
        elif (n_processes == 1) or connect_db_allproc:
            # One connection to all processes
            study = self.init_study()
            self.run_optimization(n_trials=n_trials, n_jobs=n_jobs, study=study)
        else:
            study = None

            # Calculate number of studies per process
            N_studies_left = n_trials
            n_studies_process = round(N_studies_left / n_processes)

            if method == "spawn":
                mp.set_start_method("spawn", force=True)
            else:
                mp.set_start_method("fork", force=True)

            processes = []

            try:
                for i in range(n_processes):  # Number of optimization trials
                    if i == n_processes - 1: # Last process takes remaining trials
                        n_studies_process = N_studies_left
                    else:
                        N_studies_left -= n_studies_process
                    p = mp.Process(target=self.run_optimization, args=(n_studies_process, n_jobs), kwargs={'study': study})
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

            except KeyboardInterrupt:
                print(f"\nKeyboard interrupt received. Cleaning up processes for {self.classifier}...")
                # Terminate all optimization processes
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            p.kill()  # Force kill if terminate didn't work
                raise  # Re-raise to exit the main loop

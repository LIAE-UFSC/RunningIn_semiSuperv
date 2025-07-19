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

warnings.filterwarnings("ignore", category=UserWarning)

compressor_model = "a"
n_processes = 2
n_tests = 500

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

def objective(trial):
    # Define hyperparameters to optimize
    parameters = {
        'window_size': trial.suggest_int('window_size', 1, 10),
        'delay': trial.suggest_int('delay', 1, 10),
        'moving_average': trial.suggest_int('moving_average', 1, 10),
        'scale': trial.suggest_categorical('scale', [True, False]),
        'balance': trial.suggest_categorical('balance', ['undersample', 'none'])
    }
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
    model = RunInSemiSupervised(
        **parameters,
        compressor_model=compressor_model,
        classifier="LogisticRegression",
        semisupervised_params={
            'threshold': threshold,
            'criterion': 'threshold',
            'max_iter': 100
        },
        classifier_params={},
        test_split=["a3"],
        features=[features],
    )
    
    # Train the model
    model.fit()

    result = model.cross_validate()

    labeled_percentage = gmean(result["percent_labeled"])

    joined_confusion_matrix = np.array(result["confusion_matrix"]).sum(axis=0)

    mcc = MCC_score_from_confusion_matrix(joined_confusion_matrix)
    
    return mcc, labeled_percentage

if __name__ == "__main__":
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"RunIn_LogisticRegression_{compressor_model}"  # Unique identifier of the study.
    results_dir = Path(__file__).resolve().parent.parent / "Results"
    results_dir.mkdir(exist_ok=True)
    storage_name = f"sqlite:///{results_dir.as_posix()}/RunIn_LogisticRegression.db"
    # Run optimization with two processes

    # Create Optuna study
    study = optuna.create_study(directions=["maximize", "maximize"], 
                                study_name=study_name, 
                                storage=storage_name, 
                                load_if_exists=True)
    
    start_time = time.time()
    processes = []

    for i in range(n_processes):  # Number of optimization trials
        p = mp.Process(target=study.optimize, args=(objective,), kwargs={'n_trials': int(n_tests/n_processes), 'n_jobs': 1})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start_time
    print(f"Tests finished. Elapsed time: {elapsed:.2f} seconds")
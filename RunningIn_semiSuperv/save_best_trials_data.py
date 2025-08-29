import optuna
import matplotlib.pyplot as plt
from utils import RunInSemiSupervised
from utils.optimizer import OptimizationRunIn
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import multiprocessing as mp
import os

def unpack_classifier_parameters(classifier: str,parameters: dict):
    if classifier == "LogisticRegression":
        classifier_parameters = {
            "C": parameters.pop("C"),
            "max_iter": 1000,
        }
    elif classifier == "DecisionTreeClassifier":
        classifier_parameters = {
            "max_depth": parameters.pop("max_depth"),
            "min_samples_split": parameters.pop("min_samples_split"),
            "min_samples_leaf": parameters.pop("min_samples_leaf"),
            "max_features": parameters.pop("max_features")
        }
    elif classifier == "PLSCanonical":
        classifier_parameters = {
            "n_components": parameters.pop("n_components"),
            "scale": parameters.pop("scale"),
            "max_iter": 1000
        }
    elif classifier == "KNeighborsClassifier":
        classifier_parameters = {
            "n_neighbors": parameters.pop("n_neighbors"),
            "weights": parameters.pop("weights"),
            "metric": parameters.pop("metric"),
            "p": parameters.pop("p") if "p" in parameters else 2
        }
    elif classifier == "LinearSVM":
        classifier_parameters = {
            "C": parameters.pop("C"),
            "max_iter": 5000
        }
    elif classifier == "RBFSVM":
        classifier_parameters = {
            "C": parameters.pop("C"),
            "gamma": parameters.pop("gamma"),
            "max_iter": 5000
        }
    elif classifier == "GaussianProcess":
        classifier_parameters = {
            # Left out kernel selection for now because of LinAlg errors
            # "kernel": trial.suggest_categorical('kernel', ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared"]),
            "n_restarts_optimizer": parameters.pop("n_restarts_optimizer"),
            "max_iter_predict": 100
        }
    elif classifier == "RandomForest":
        classifier_parameters = {
            "n_estimators": parameters.pop("n_estimators"),
            "max_depth": parameters.pop("max_depth"),
            "min_samples_split": parameters.pop("min_samples_split"),
            "min_samples_leaf": parameters.pop("min_samples_leaf"),
            "max_features": parameters.pop("max_features"),
            "bootstrap": parameters.pop("bootstrap")
        }
    elif classifier == "NeuralNet":
        hidden_layer_sizes = tuple([parameters.pop(f'n_units_l{i}') 
                                    for i in range(parameters.pop('n_layers'))])
        classifier_parameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": parameters.pop('activation'),
            "alpha": parameters.pop('alpha'),
            "learning_rate": parameters.pop('learning_rate'),
            "max_iter": 600
        }
    elif classifier == "AdaBoost":
        classifier_parameters = {
            "n_estimators": parameters.pop("n_estimators"),
            "learning_rate": parameters.pop("learning_rate"),
        }
    elif classifier == "NaiveBayes":
        classifier_parameters = {
            "var_smoothing": parameters.pop("var_smoothing")
        }
    elif classifier == "QDA":
        classifier_parameters = {
            "reg_param": parameters.pop("reg_param"),
            "store_covariance": parameters.pop("store_covariance")
        }
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return classifier_parameters

def join_score_data(X, Y, meta, Y_score):
    result = X.copy()
    result['Y_true'] = Y
    result['Y_score'] = Y_score
    result = result.join(meta)
    return result

def MCC_score(Y_test, Y_pred):
    if isinstance(Y_test, pd.Series) or isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.reset_index(drop=True)
        labeled_test_index = Y_test[Y_test != -1].index
        Y_true = Y_test.loc[labeled_test_index]
    else: # Y_test is np.ndarray
        labeled_test_index = np.where(Y_test != -1)[0]
        Y_true = Y_test[labeled_test_index]
    if isinstance(Y_pred, pd.Series) or isinstance(Y_pred, pd.DataFrame):
        Y_pred = Y_pred.reset_index(drop=True)
        Y_pred_labeled = Y_pred.loc[labeled_test_index]
    else:
        Y_pred_labeled = Y_pred[labeled_test_index]

    # Compute MCC
    mcc = matthews_corrcoef(Y_true, Y_pred_labeled)
    return mcc

def get_unlabeled_data(X_train, Y_train, meta_train):

    # Identify unlabeled instances
    unlabeled_indices = Y_train[Y_train == -1].index
    if isinstance(X_train, np.ndarray):
        X_unlabeled = X_train[unlabeled_indices]
    else:
        X_unlabeled = X_train.loc[unlabeled_indices]
    Y_unlabeled = Y_train.loc[unlabeled_indices]
    meta_unlabeled = meta_train.loc[unlabeled_indices]

    return X_unlabeled, Y_unlabeled, meta_unlabeled

def get_labeled_data(X_train, Y_train, meta_train):
    Y_train = Y_train.reset_index(drop=True)
    meta_train = meta_train.reset_index(drop=True)
    # Identify labeled instances
    labeled_indices = Y_train[Y_train != -1].index
    if isinstance(X_train, np.ndarray):
        X_labeled = X_train[labeled_indices]
    else:
        X_train = X_train.reset_index(drop=True)
        X_labeled = X_train.loc[labeled_indices]
    Y_labeled = Y_train.loc[labeled_indices]
    meta_labeled = meta_train.loc[labeled_indices]

    return X_labeled, Y_labeled, meta_labeled

def save_single_study(study_name,storage_name, thr_labeled=0):
    _, classifier, compressor_model = study_name.split("_")
    if compressor_model == "a":
        test_compressor = ["a3"]
    elif compressor_model == "b":
        test_compressor = ["b10"]
    else:
        test_compressor = ["a3", "b10"]

    best_trials = optuna.load_study(study_name=study_name, storage=storage_name).best_trials

    # Pick the trial with the best MCC which has at least a certain % of labeled
    valid_trials = [trial for trial in best_trials if trial.values[1] >= thr_labeled]

    if not valid_trials:
        print(f"No valid trials found for study {study_name} with compressor model {compressor_model}.")
        return

    best_trial = max(valid_trials, key=lambda t: t.values[0])
    trial_number = best_trial.number
    expected_MCC = best_trial.values[0]
    expected_percentage_labeled = best_trial.values[1]

    parameters = best_trial.params

    selflearning_threshold = parameters.pop('threshold')
    features = parameters.pop('features')
    classifier_parameters = unpack_classifier_parameters(classifier,parameters)
    pca = parameters.pop('n_pca',0)
    del parameters['pca']
    # Plot the best trial
    model = RunInSemiSupervised(
        **(parameters),
        compressor_model=compressor_model,
        classifier=classifier,
        pca=pca,
        semisupervised_params={
            'threshold': selflearning_threshold,
            'criterion': 'threshold',
            'max_iter': 1000
        },
        classifier_params=classifier_parameters,
        test_split=test_compressor,
        features=[features],
    )

    model.fit()
    X_train,Y_train = model.get_train_data()
    meta_train = model._preprocessor.get_train_metadata()
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    meta_train = meta_train.reset_index(drop=True)

    Y_predict_train = model.predict(X_train, already_transformed=True)
    Y_predict_train_labeled, Y_train_labeled, _ = get_labeled_data(Y_predict_train, Y_train, meta_train)

    MCC_labeled_train = MCC_score(Y_train_labeled, Y_predict_train_labeled)

    X_test,Y_test = model.get_test_data()
    meta_test = model._preprocessor.get_test_metadata()
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    meta_test = meta_test.reset_index(drop=True)

    Y_test_predict = model.predict(X_test, already_transformed=True)
    Y_test_predict_labeled, Y_test_labeled, _ = get_labeled_data(Y_test_predict, Y_test, meta_test)
    MCC_test = MCC_score(Y_test_labeled, Y_test_predict_labeled)

    ammount_labeled = np.sum(model._model.labeled_iter_ == 1)
    percent_labeled = ammount_labeled/len(model._model.labeled_iter_)

    Y_score_test = model.predict_proba(X_test, already_transformed=True)
    Y_score_train = model.predict_proba(X_train, already_transformed=True)

    # Select the positive class probability
    pos_index = list(model._model.classes_).index(1)
    Y_score_test = Y_score_test[:, pos_index]
    Y_score_train = Y_score_train[:, pos_index]

    data_train = join_score_data(X_train, Y_train, meta_train, Y_score_train)
    data_test = join_score_data(X_test, Y_test, meta_test, Y_score_test)

    # Save the dataframes to CSV files
    results_dir = os.path.join("Results", "data")
    os.makedirs(results_dir, exist_ok=True)
    data_train.to_csv(os.path.join(results_dir, f"data_train_{study_name}.csv"), index=False)
    data_test.to_csv(os.path.join(results_dir, f"data_test_{study_name}.csv"), index=False)

    print(f"\n===== Study: {study_name} | Classifier: {classifier} | Compressor Model: {compressor_model} | Test Compressor: {test_compressor} =====")
    print(f"Selected trial: ")
    print(f"Expected MCC: {expected_MCC:.4f}")
    print(f"Expected Percentage Labeled: {expected_percentage_labeled:.2%}")
    print(f"MCC (Train): {MCC_labeled_train:.4f}")
    print(f"Percentage Labeled (train): {percent_labeled:.2%}")
    print(f"MCC (Test): {MCC_test:.4f}")

    result = {
        "study": study_name,
        "classifier": classifier,
        "compressor_model": compressor_model,
        "trial_number": trial_number,
        "test_compressor": test_compressor,
        "expected_MCC": expected_MCC,
        "expected_percentage_labeled": expected_percentage_labeled,
        "MCC_train": MCC_labeled_train,
        "MCC_test": MCC_test,
        "percent_labeled": percent_labeled
    }

    return result

if __name__ == "__main__":
    thr_labeled = 0

    # Load PostgreSQL configuration
    POSTGRES_CONFIG = OptimizationRunIn.load_postgres_config()

    compressor_models = ["a", "b", "all"]
    test_compressors = [["a3"],["b10"], ["a3","b10"]]

    storage_name = (f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}"
                    f"@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}"
                    f"/{POSTGRES_CONFIG['database']}")
    
    studies = optuna.study.get_all_study_names(storage_name)
    
    with mp.Pool(len(studies)) as pool:
        studies_results = pool.starmap(save_single_study, [(study, storage_name) for study in studies])

    # Save the results to a CSV file
    results_df = pd.DataFrame(studies_results)
    results_data_dir = os.path.join("Results", "data")
    os.makedirs(results_data_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_data_dir, "studies_results.csv"), index=False)
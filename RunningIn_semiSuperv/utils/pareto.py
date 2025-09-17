import optuna
from matplotlib import pyplot as plt
from typing import Tuple
import itertools
import multiprocessing as mp

def get_valid_studies(storage_name, compressor_model = None):
    studies = optuna.get_all_study_names(storage=storage_name)
    valid_studies = []
    for study in studies:
        if compressor_model == None:
            valid_studies.append(study)
        elif study.startswith(f"RunIn_") and study.endswith(f"_{compressor_model}"):
            valid_studies.append(study)
    return valid_studies

def get_pareto_front_single_study(study_name, storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study_summary = {}
    pareto_trials = study.best_trials
    study_summary["pareto_trials"] = sorted(pareto_trials, key=lambda t: t.values[0])
    study_summary["total_studies"] = len(study.trials)
    study_summary["study_name"] = study_name
    return study_summary

def get_pareto_front_multiple_studies(studies, storage_name):
    with mp.Pool(len(studies)) as pool:
        study_summaries = pool.starmap(get_pareto_front_single_study, [(study, storage_name) for study in studies])
    return study_summaries

def get_labels(study_names):
    classifiers = []
    compressor_models = []
    for study_name in study_names:
        _, classifier, compressor_model = study_name.split("_")
        classifiers.append(classifier)
        compressor_models.append(compressor_model)

    if len(set(compressor_models)) == 1:
        labels = classifiers
    if len(set(classifiers)) == 1:
        labels = compressor_models
    else:
        labels = [f"{c}_{cm}" for c, cm in zip(classifiers, compressor_models)]
    return labels

def plot_studies(pareto_studies):
    fig, ax = plt.subplots()
    colors = plt.get_cmap('tab10', len(pareto_studies))
    color_cycle = itertools.cycle(colors.colors)

    study_names = [study["study_name"] for study in pareto_studies]
    pareto_trials = [study["pareto_trials"] for study in pareto_studies]

    legend_labels = get_labels(study_names)
    legend_plots = []
    for study_name, trials in zip(study_names, pareto_trials):
        color = next(color_cycle)
        lin, = ax.step(
            [t.values[0] for t in trials],
            [t.values[1] for t in trials],
            where="pre",
            linestyle="--",
            color=color,
            alpha=0.5  # Make the line slightly faded
        )
        mark, = ax.plot(
            [t.values[0] for t in trials],
            [t.values[1] for t in trials],
            'o',
            markersize=4,
            color=color
        )
        _,classifier,_ = study_name.split("_")

        legend_plots.append((lin, mark))

    ax.set_xlabel("MCC")
    ax.set_ylabel("% labeled")
    ax.legend(legend_plots, legend_labels)
    
    return fig, ax

def plot_pareto_per_compressor(storage_name = None, compressor_model = None, summaries = None):
    if summaries is None:
        if storage_name is None:
            raise ValueError("Either summaries or storage_name must be provided.")
        summaries = get_pareto_front_multiple_studies(get_valid_studies(storage_name, compressor_model), storage_name)

    summaries = [summary for summary in summaries if summary["study_name"].endswith(f"_{compressor_model}")]
    fig, ax = plot_studies(summaries)
    if compressor_model is None:
        ax.set_title("Pareto Front")
    elif compressor_model == "all":
        ax.set_title("Pareto Front for all models")
    else:
        ax.set_title(f"Pareto Front for model {compressor_model.upper()}")
    return fig, ax

def get_summary_studies(storage_name = None, compressor_model = "a", summaries = None) -> Tuple[list, str]:
    if summaries is None:
        if storage_name is None:
            raise ValueError("Either summaries or storage_name must be provided.")
        summaries = get_pareto_front_multiple_studies(get_valid_studies(storage_name, compressor_model), storage_name)

    available_studies = [summary["study_name"] for summary in summaries]

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

    
    str_summary = ""
    header = "=" * 10
    title = f"Summary of tests for compressor: {compressor_model}"
    
    str_summary += header + "\n"
    str_summary += title + "\n"
    str_summary += header + "\n"
    
    print(header)
    print(title)
    print(header)

    result_summary = []
    
    for classifier in supported_classifiers:
        study_name = f"RunIn_{classifier}_{compressor_model}"
        if study_name in available_studies:
            idx = available_studies.index(study_name)
            trial = summaries[idx]
            trial["pareto_trials"] = sorted(trial["pareto_trials"], key=lambda t: t.number)
            last_pareto = trial["pareto_trials"][-1].number if trial["pareto_trials"] else 0
            n_done = trial["total_studies"]
            line = f"{classifier}::: Trials Done: {n_done}       |  Last Pareto advance: {last_pareto}"
            str_summary += line + "\n"
            result = {"classifier": classifier, "trials_done": n_done, "last_pareto": last_pareto, "compressor_model": compressor_model,
                      "pareto_values": [t.values for t in trial["pareto_trials"]]}
            result_summary.append(result)
            print(line)
        else:
            line = f"{classifier}::: Not found."
            str_summary += line + "\n"
            print(line)
    
    return result_summary, str_summary
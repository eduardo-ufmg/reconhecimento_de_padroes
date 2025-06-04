import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).parent.parent / "CustomSVC"))

from CustomSVC.CustomSVC import CustomSVC


class EquivalenceResults:
    pvalue: float
    equivalent: bool

    def __init__(self, pvalue: float, equivalent: bool):
        self.pvalue = pvalue
        self.equivalent = equivalent


class AccuracyResults:
    mean: float
    std: float

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std


class DatasetResults:
    accuracy_results: dict[str, AccuracyResults]
    time: dict[str, float]
    equivalence: dict[tuple[str, str], EquivalenceResults]

    def __init__(
        self,
        accuracy_results: dict[str, AccuracyResults],
        time: dict[str, float],
        equivalence: dict[tuple[str, str], EquivalenceResults],
    ):
        self.accuracy_results = accuracy_results
        self.time = time
        self.equivalence = equivalence


ExperimentResults = dict[str, DatasetResults]


def load_dataset(dataset_name: str):
    sets_dir = Path(__file__).parent.parent / "sets"
    file_path = sets_dir / f"{dataset_name}.npz"
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
    return X, y


def compute_pvalue(results1: AccuracyResults, results2: AccuracyResults) -> float:
    # Using Wilcoxon signed-rank test for paired samples
    return wilcoxon([results1.mean], [results2.mean]).pvalue


def run_experiment(dataset: str) -> DatasetResults:
    X, y = load_dataset(dataset)

    sklearns_svc_pipeline = Pipeline([("pca", PCA(n_components="mle")), ("svc", SVC())])

    murilos_svc_pipeline = Pipeline(
        [
            ("pca", PCA(n_components="mle")),
            ("svc", CustomSVC(optimization_metric="dissimilarity")),
        ]
    )

    my_svc_pipeline = Pipeline(
        [
            ("pca", PCA(n_components="mle")),
            ("svc", CustomSVC(optimization_metric="spatial_spread")),
        ]
    )

    pipelines = {
        "sklearns_svc_pipeline": sklearns_svc_pipeline,
        "murilos_svc_pipeline": murilos_svc_pipeline,
        "my_svc_pipeline": my_svc_pipeline,
    }

    accuracy_results = {}
    time_results = {}

    for name, pipeline in pipelines.items():
        start_time = time.time()
        scores = cross_val_score(pipeline, X, y)
        elapsed_time = time.time() - start_time

        accuracy_results[name] = AccuracyResults(mean=scores.mean(), std=scores.std())
        time_results[name] = elapsed_time

    equivalence_results = {}
    pairs = [
        ("sklearns_svc_pipeline", "murilos_svc_pipeline"),
        ("sklearns_svc_pipeline", "my_svc_pipeline"),
        ("murilos_svc_pipeline", "my_svc_pipeline"),
    ]

    for name1, name2 in pairs:
        pvalue = compute_pvalue(accuracy_results[name1], accuracy_results[name2])
        equivalent = pvalue > 0.05
        equivalence_results[(name1, name2)] = EquivalenceResults(
            pvalue=pvalue, equivalent=equivalent
        )

    dataset_results = DatasetResults(
        accuracy_results=accuracy_results,
        time=time_results,
        equivalence=equivalence_results,
    )

    return dataset_results


if __name__ == "__main__":
    DATASETS = [
        "adult",
        "banknote-authentication",
        "blood-transfusion-service-center",
        "breast_cancer",
        "diabetes",
        "digits_binary_0_vs_1",
        "digits_binary_5_vs_rest",
        "german_credit_g",
        "ionosphere",
        "iris_binary_setosa_vs_rest",
        "iris_binary_setosa_vs_versicolor",
        "kc1",
        "mushroom",
        "qsar-biodeg",
        "sonar",
        "spambase",
        "sylvine",
        "titanic",
        "vote",
        "wpbc",
    ]

    experiment_results: ExperimentResults = {}

    for dataset in DATASETS:
        experiment_results[dataset] = run_experiment(dataset)

    output_dir = Path(__file__).resolve() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "experiment_results.json"

    with open(output_file, "w") as f:
        json.dump(experiment_results, f)

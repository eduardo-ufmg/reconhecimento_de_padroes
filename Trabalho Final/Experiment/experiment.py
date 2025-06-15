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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).parent.parent))

from CustomSVC.customSVC import CSVC


class EquivalenceResults:
    pvalue: float
    equivalent: bool

    def __init__(self, pvalue: float, equivalent: bool):
        self.pvalue = pvalue
        self.equivalent = equivalent


class AccuracyResults:
    scores: np.ndarray
    mean: float
    std: float

    def __init__(self, scores: np.ndarray):
        self.scores = scores
        self.mean = scores.mean()
        self.std = scores.std()


class DatasetResults:
    accuracy_results: dict[str, AccuracyResults]
    time: dict[str, float]
    equivalence: dict[str, EquivalenceResults]

    def __init__(
        self,
        accuracy_results: dict[str, AccuracyResults],
        time: dict[str, float],
        equivalence: dict[str, EquivalenceResults],
    ):
        self.accuracy_results = accuracy_results
        self.time = time
        self.equivalence = equivalence


ExperimentResults = dict[str, DatasetResults]


class Preprocessor(Pipeline):
    """Pipeline for preprocessing the dataset with StandardScaler and PCA."""

    n_components: int | None = None

    def __init__(self, n_components=None):
        steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
        ]
        super().__init__(steps)
        self.n_components = n_components


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (AccuracyResults, EquivalenceResults, DatasetResults)):
            return o.__dict__
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def load_dataset(dataset_name: str):
    sets_dir = Path(__file__).parent.parent / "sets"
    file_path = sets_dir / f"{dataset_name}.npz"
    with np.load(file_path) as data:
        X = data["X"]
        y = data["y"]
    return X, y


def compute_pvalue(results1: AccuracyResults, results2: AccuracyResults) -> float:
    # Using Wilcoxon signed-rank test for paired samples
    return wilcoxon(results1.scores, results2.scores).pvalue  # type: ignore


def run_experiment(dataset: str) -> DatasetResults:

    try:
        X, y = load_dataset(dataset)
    except FileNotFoundError:
        print(f"Dataset {dataset} not found. Skipping...")
        return DatasetResults({}, {}, {})

    sklearns_svc_pipeline = Pipeline([("preprocessor", Preprocessor()), ("svc", SVC())])

    murilos_svc_pipeline = Pipeline(
        [
            ("preprocessor", Preprocessor()),
            ("svc", CSVC(optimization_metric="dissimilarity")),
        ]
    )

    my_svc_pipeline = Pipeline(
        [
            ("preprocessor", Preprocessor()),
            ("svc", CSVC(optimization_metric="spatial_spread")),
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
        scores = cross_val_score(pipeline, X, y, error_score="raise")
        elapsed_time = time.time() - start_time

        accuracy_results[name] = AccuracyResults(scores=scores)
        time_results[name] = elapsed_time

    equivalence_results = {}
    pairs = [
        ("sklearns_svc_pipeline", "murilos_svc_pipeline"),
        ("sklearns_svc_pipeline", "my_svc_pipeline"),
        ("murilos_svc_pipeline", "my_svc_pipeline"),
    ]

    for name1, name2 in pairs:
        if np.isnan(accuracy_results[name1].mean) or np.isnan(
            accuracy_results[name2].mean
        ):
            pvalue = np.nan
            equivalent = False
        else:
            pvalue = compute_pvalue(accuracy_results[name1], accuracy_results[name2])
            equivalent = bool(pvalue > 0.05)

        key = str((name1, name2))
        equivalence_results[key] = EquivalenceResults(
            pvalue=pvalue, equivalent=equivalent
        )

    dataset_results = DatasetResults(
        accuracy_results=accuracy_results,
        time=time_results,
        equivalence=equivalence_results,
    )

    return dataset_results


if __name__ == "__main__":
    # Dynamically find datasets in the "sets" directory
    sets_dir = Path(__file__).parent.parent / "sets"
    if not sets_dir.exists():
        print(f"Error: The directory '{sets_dir}' was not found. Exiting.")
        sys.exit(1)

    # Get all files ending with .npz and extract their names without the extension
    DATASETS = sorted([f.stem for f in sets_dir.glob("*.npz")])

    if not DATASETS:
        print(f"No datasets (.npz files) found in '{sets_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(DATASETS)} datasets to process: {DATASETS}")

    output_dir = Path(__file__).parent.resolve() / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "experiment_results.json"

    experiment_results: ExperimentResults = {}
    if output_file.exists():
        with open(output_file, "r") as f:
            try:
                experiment_results = json.load(f)
            except json.JSONDecodeError:
                print("Results file is empty or corrupted. Starting from scratch.")
                experiment_results = {}

    for dataset in DATASETS:
        if dataset in experiment_results and experiment_results[dataset].get(
            "accuracy_results"
        ):
            print(f"Dataset {dataset} already tested. Skipping...")
            continue

        print(f"Running experiment on: {dataset}")
        try:
            dataset_results = run_experiment(dataset)
            experiment_results[dataset] = dataset_results

            with open(output_file, "w") as f:
                json.dump(experiment_results, f, cls=CustomEncoder, indent=4)
            print(f"Results for {dataset} saved.")

        except Exception as e:
            print(f"An error occurred while running the experiment for {dataset}: {e}")
            continue

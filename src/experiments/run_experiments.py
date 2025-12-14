import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Dataset loaders
from src.datasets.adult import load_data as load_adult
from src.datasets.bank import load_data as load_bank
from src.datasets.breast_cancer import load_data as load_breast_cancer
from src.datasets.heart import load_data as load_heart
from src.datasets.wine import load_data as load_wine

# Models
from src.models.logistic import get_model_and_params as logistic_model
from src.models.knn import get_model_and_params as knn_model
from src.models.svm import get_model_and_params as svm_model
from src.models.random_forest import get_model_and_params as rf_model
from src.models.adaboost import get_model_and_params as adaboost_model


# ==========================
# Configuration
# ==========================

TRAIN_FRACTIONS = [0.2, 0.5, 0.8]
RANDOM_SEEDS = [0, 1, 2]
CV_FOLDS = 5
N_JOBS = -1

RESULTS_DIR = "results/tables"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")


# ==========================
# Dataset registry
# ==========================

DATASETS = {
    "Adult": load_adult,
    "Bank": load_bank,
    "BreastCancer": load_breast_cancer,
    "HeartDisease": load_heart,
    "WineQuality": load_wine,
}


# ==========================
# Model registry
# ==========================

MODELS = {
    "LogisticRegression": logistic_model,
    "kNN": knn_model,
    "SVM_RBF": svm_model,
    "RandomForest": rf_model,
    "AdaBoost": adaboost_model,
}


# ==========================
# Helper function
# ==========================

def evaluate_model(
    X, y,
    model_name,
    model_fn,
    train_frac,
    seed
):
    """
    Train, tune, and evaluate a single model on a single split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_frac,
        random_state=seed,
        stratify=y
    )

    model, param_grid = model_fn(X_train)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=CV_FOLDS,
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    # Best estimator
    best_model = grid.best_estimator_

    # Accuracies
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    val_acc = grid.best_score_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    return {
        "classifier": model_name,
        "train_fraction": train_frac,
        "seed": seed,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "best_params": json.dumps(grid.best_params_)
    }


# ==========================
# Main experiment loop
# ==========================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []

    for dataset_name, dataset_loader in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        X, y = dataset_loader()

        for model_name, model_fn in MODELS.items():
            print(f"\n--- Classifier: {model_name} ---")

            for train_frac in TRAIN_FRACTIONS:
                for seed in RANDOM_SEEDS:
                    print(
                        f"Running | Train={train_frac:.1f} | Seed={seed}"
                    )

                    result = evaluate_model(
                        X=X,
                        y=y,
                        model_name=model_name,
                        model_fn=model_fn,
                        train_frac=train_frac,
                        seed=seed
                    )

                    result["dataset"] = dataset_name
                    all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)

    df = df[
        [
            "dataset",
            "classifier",
            "train_fraction",
            "seed",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "best_params",
        ]
    ]

    df.to_csv(RESULTS_FILE, index=False)

    print("\n===================================")
    print("All experiments completed.")
    print(f"Results saved to: {RESULTS_FILE}")
    print("===================================")


if __name__ == "__main__":
    main()

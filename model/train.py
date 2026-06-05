"""Train and select the diabetes risk model.

Pipeline upgrades over the v1 script:
- Biologically impossible zeros (Glucose, BloodPressure, SkinThickness,
  Insulin, BMI) are treated as missing and median-imputed inside the
  pipeline, so no information leaks from test folds.
- Three candidate models are compared with stratified 5-fold
  cross-validation on the training split and selected by mean ROC-AUC.
- The winner is refit on the full training split and reported against an
  untouched 30% hold-out set, alongside the majority-class baseline.
- Metrics and feature importance are exported to model/metrics.json,
  which the dashboard reads.
  
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "diabetes.csv"
MODEL_DIR = REPO_ROOT / "model"
MODEL_PATH = MODEL_DIR / "diabetes_model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COLUMN = "Outcome"

# Zero is biologically impossible for these measurements; in the Pima
# dataset it encodes a missing value.
IMPOSSIBLE_ZERO_COLUMNS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
]

TEST_SIZE = 0.3
RANDOM_STATE = 42
CV_FOLDS = 5


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load the dataset and convert impossible zeros to missing values."""
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Dataset not found at {path}. "
                 "Expected data/diabetes.csv in the repository root.")
    except pd.errors.ParserError as error:
        sys.exit(f"Dataset at {path} could not be parsed: {error}")

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN]
               if c not in data.columns]
    if missing:
        sys.exit(f"Dataset is missing required columns: {missing}")

    zero_counts = {
        column: int((data[column] == 0).sum())
        for column in IMPOSSIBLE_ZERO_COLUMNS
    }
    data[IMPOSSIBLE_ZERO_COLUMNS] = (
        data[IMPOSSIBLE_ZERO_COLUMNS].replace(0, np.nan)
    )
    return data, zero_counts


def summarize_dataset(data: pd.DataFrame, zero_counts: dict) -> dict:
    """Print dataset overview and return class distribution counts."""
    total = len(data)
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Shape: {data.shape}")
    print("\nImpossible zeros converted to missing (median-imputed in-pipeline):")
    for column, count in zero_counts.items():
        print(f"  {column:<14}: {count:>3} ({count / total * 100:.1f}%)")

    counts = data[TARGET_COLUMN].value_counts()
    no_diabetes, has_diabetes = int(counts[0]), int(counts[1])

    print("\nCLASS DISTRIBUTION")
    print(f"No Diabetes  : {no_diabetes} ({no_diabetes / total * 100:.1f}%)")
    print(f"Has Diabetes : {has_diabetes} ({has_diabetes / total * 100:.1f}%)")
    print(f"Majority-class baseline accuracy: {no_diabetes / total * 100:.1f}%")

    return {
        "total_patients": total,
        "no_diabetes": no_diabetes,
        "has_diabetes": has_diabetes,
    }


def candidate_pipelines() -> dict:
    """Three candidates sharing the same leakage-safe preprocessing."""
    imputer = SimpleImputer(strategy="median")
    return {
        "Logistic Regression": Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE,
                                         max_iter=1000)),
        ]),
        "Logistic Regression (balanced)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=RANDOM_STATE,
                                         max_iter=1000,
                                         class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300,
                                             random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }


def select_model(X_train, y_train) -> tuple[str, Pipeline, dict]:
    """Cross-validate all candidates on the training split, pick by ROC-AUC."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    print("\n" + "=" * 70)
    print(f"MODEL SELECTION: stratified {CV_FOLDS}-fold CV on training split")
    print("=" * 70)

    cv_results = {}
    for name, pipeline in candidate_pipelines().items():
        scores = cross_validate(pipeline, X_train, y_train, cv=folds,
                                scoring=["accuracy", "roc_auc", "recall"])
        cv_results[name] = {
            "cv_accuracy_mean": round(scores["test_accuracy"].mean() * 100, 1),
            "cv_accuracy_std": round(scores["test_accuracy"].std() * 100, 1),
            "cv_roc_auc_mean": round(scores["test_roc_auc"].mean(), 3),
            "cv_roc_auc_std": round(scores["test_roc_auc"].std(), 3),
            "cv_recall_diabetic_mean": round(scores["test_recall"].mean() * 100, 1),
            "_raw_auc": scores["test_roc_auc"].mean(),
            "_raw_recall": scores["test_recall"].mean(),
        }
        r = cv_results[name]
        print(f"{name:<32} accuracy {r['cv_accuracy_mean']}% "
              f"(+/- {r['cv_accuracy_std']})  "
              f"ROC-AUC {r['cv_roc_auc_mean']}  "
              f"recall {r['cv_recall_diabetic_mean']}%")

    # Selection: ROC-AUC first; diabetic recall breaks ties because a
    # screening tool should prioritise catching true positives.
    winner_name = max(cv_results, key=lambda n: (
        round(cv_results[n]["_raw_auc"], 3), cv_results[n]["_raw_recall"]))
    for result in cv_results.values():
        result.pop("_raw_auc"), result.pop("_raw_recall")
    winner = candidate_pipelines()[winner_name]
    print(f"\nSelected by mean ROC-AUC: {winner_name}")
    return winner_name, winner, cv_results


def evaluate_holdout(pipeline, X_test, y_test, baseline_pct: float) -> dict:
    """Report hold-out performance for the selected, refit model."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    recall_diabetic = recall_score(y_test, predictions, pos_label=1)
    recall_healthy = recall_score(y_test, predictions, pos_label=0)

    print("\n" + "=" * 70)
    print("HOLD-OUT PERFORMANCE (untouched 30% test split)")
    print("=" * 70)
    print(f"Accuracy            : {accuracy * 100:.2f}%  "
          f"(majority baseline: {baseline_pct:.1f}%)")
    print(f"ROC-AUC             : {auc:.3f}")
    print(f"Recall, diabetic    : {recall_diabetic * 100:.1f}%")
    print(f"Recall, non-diabetic: {recall_healthy * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions,
                                target_names=["No Diabetes", "Has Diabetes"]))
    return {
        "accuracy_pct": round(accuracy * 100, 1),
        "roc_auc": round(float(auc), 3),
        "recall_diabetic_pct": round(recall_diabetic * 100, 1),
        "recall_healthy_pct": round(recall_healthy * 100, 1),
    }


def feature_importance(pipeline, winner_name: str) -> list:
    """Model-appropriate importance: signed coefficients or impurity scores."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        values, signed = model.coef_[0], True
    else:
        values, signed = model.feature_importances_, False

    ranked = sorted(zip(FEATURE_COLUMNS, values),
                    key=lambda pair: abs(pair[1]), reverse=True)
    print(f"\nFeature importance ({winner_name}):")
    for feature, value in ranked:
        print(f"  {feature}: {value:.4f}")
    return [{"feature": f, "value": round(float(v), 4), "signed": signed}
            for f, v in ranked]


def save_artifacts(pipeline, metrics: dict) -> None:
    """Persist the fitted pipeline and the metrics the dashboard displays."""
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(pipeline, model_file)
    with open(METRICS_PATH, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"\nModel saved to   : {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"Metrics saved to : {METRICS_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    data, zero_counts = load_dataset(DATA_PATH)
    class_counts = summarize_dataset(data, zero_counts)
    baseline_pct = round(
        class_counts["no_diabetes"] / class_counts["total_patients"] * 100, 1)

    features, target = data[FEATURE_COLUMNS], data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target)

    winner_name, winner, cv_results = select_model(X_train, y_train)
    winner.fit(X_train, y_train)
    holdout = evaluate_holdout(winner, X_test, y_test, baseline_pct)
    importance = feature_importance(winner, winner_name)

    metrics = {
        "model_name": winner_name,
        **holdout,
        "baseline_accuracy_pct": baseline_pct,
        "cv_results": cv_results,
        "zero_values_imputed": zero_counts,
        "feature_importance": importance,
        **class_counts,
        "no_diabetes_pct": round(
            class_counts["no_diabetes"] / class_counts["total_patients"] * 100, 1),
        "has_diabetes_pct": round(
            class_counts["has_diabetes"] / class_counts["total_patients"] * 100, 1),
    }
    save_artifacts(winner, metrics)


if __name__ == "__main__":
    main()
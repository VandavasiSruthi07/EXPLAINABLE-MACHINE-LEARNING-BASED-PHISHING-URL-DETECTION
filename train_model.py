
"""Phishing URL Detection - Model Training"""

from pathlib import Path
import json
import warnings
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from feature_extraction import FEATURE_NAMES, feature_vector


# Project paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_PATH = DATA_DIR / "urlset.csv"


def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip"
    )

    df.columns = df.columns.str.strip().str.lower()

    # Convert domain column into url
    if "domain" in df.columns and "url" not in df.columns:
        df.rename(columns={"domain": "url"}, inplace=True)

    if not {"url", "label"}.issubset(df.columns):
        raise ValueError(
            f"Required columns missing. Found: {list(df.columns)}"
        )

    # Clean dataset
    df = df.dropna(subset=["url", "label"]).copy()

    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df[df["label"].isin([0, 1])]
    df = df[df["url"] != ""]
    df = df.drop_duplicates(subset=["url"])

    print("\nDataset loaded successfully!")
    print("Total URLs:", len(df))
    print("Legitimate URLs:", (df["label"] == 0).sum())
    print("Phishing URLs:", (df["label"] == 1).sum())

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both labels 0 and 1.")

    return df


def train_models(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression": make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        ),

        "Random Forest": make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(
                n_estimators=120,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    }

    # Add XGBoost if installed
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = make_pipeline(
            SimpleImputer(),
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )

    except Exception as error:
        warnings.warn(f"XGBoost unavailable: {error}")

    results = []
    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": round(
                accuracy_score(y_test, predictions), 4
            ),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0), 4
            ),
            "f1": round(
                f1_score(y_test, predictions, zero_division=0), 4
            )
        }

        results.append(metrics)
        trained_models[name] = model

        print(metrics)

    return results, trained_models


def main():

    MODEL_DIR.mkdir(exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = load_dataset()

    print("\nExtracting URL features...")

    X = pd.DataFrame(
        [feature_vector(url) for url in df["url"]],
        columns=FEATURE_NAMES
    )

    y = df["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTraining models...")

    results, trained_models = train_models(
        X_train, X_test, y_train, y_test
    )

    # Select best model using F1 score
    best_result = max(
        results,
        key=lambda item: (item["f1"], item["recall"])
    )

    best_name = best_result["model"]

    # Save best model
    joblib.dump(
        {
            "model": trained_models[best_name],
            "model_name": best_name,
            "feature_names": FEATURE_NAMES
        },
        MODEL_DIR / "best_model.joblib"
    )

    # Save evaluation metrics
    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\n================================")
    print("       TRAINING COMPLETED")
    print("================================")

    print("Best Model:", best_name)
    print("Accuracy:", best_result["accuracy"])
    print("F1 Score:", best_result["f1"])

    print("\nModels saved successfully!")


if __name__ == "__main__":
    main()

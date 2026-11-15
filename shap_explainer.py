
from pathlib import Path
import numpy as np
import pandas as pd

from feature_extraction import FEATURE_NAMES


def explain_prediction(pipeline, feature_names, feature_values):

    try:
        import shap

        root = Path(__file__).resolve().parent

        # Load background dataset
        data_path = root / "data" / "urls.csv"

        if not data_path.exists():
            data_path = root / "data" / "sample_urls.csv"

        df = pd.read_csv(data_path)

        background = pd.DataFrame(
            [
                __import__("feature_extraction").feature_vector(str(url))
                for url in df["url"]
            ],
            columns=feature_names
        )

        # Separate preprocessing and ML model
        preprocessing = pipeline[:-1]
        estimator = pipeline.steps[-1][1]

        background_transformed = preprocessing.transform(background)

        current = pd.DataFrame(
            [feature_values],
            columns=feature_names
        )

        current_transformed = preprocessing.transform(current)

        model_name = estimator.__class__.__name__

        # SHAP for Logistic Regression
        if model_name == "LogisticRegression":

            explainer = shap.LinearExplainer(
                estimator,
                background_transformed
            )

            shap_values = explainer.shap_values(
                current_transformed
            )

        # SHAP for Random Forest and XGBoost
        elif model_name in ("RandomForestClassifier", "XGBClassifier"):

            explainer = shap.TreeExplainer(estimator)

            shap_values = explainer.shap_values(
                current_transformed
            )

        else:
            raise ValueError("Unsupported model for this SHAP implementation")

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])[0]
        else:
            values = np.asarray(shap_values)

            if values.ndim == 3:
                values = values[0, :, -1]
            elif values.ndim == 2:
                values = values[0]

        ranked = sorted(
            zip(feature_names, values),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )[:5]

        explanations = [
            {
                "feature": name,
                "impact": round(float(value), 4)
            }
            for name, value in ranked
        ]

        return explanations, "SHAP explanation"

    except Exception as error:

        print("SHAP Error:", error)

        return [], "SHAP unavailable"

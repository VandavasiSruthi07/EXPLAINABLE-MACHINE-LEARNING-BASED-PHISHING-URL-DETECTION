from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from feature_extraction import FEATURE_NAMES, extract_features, feature_vector
from shap_explainer import explain_prediction

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
app = Flask(__name__)

def load_artifacts():
    if not MODEL_PATH.exists():
        return None, []
    bundle = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else []
    return bundle, metrics

@app.route("/")
def index():
    bundle, metrics = load_artifacts()
    return render_template("index.html", ready=bundle is not None)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL is too long (maximum 2048 characters)."}), 400
    bundle, _ = load_artifacts()
    if bundle is None:
        return jsonify({"error": "Model not trained yet. Run: python train_model.py"}), 503

    model = bundle["model"]
    values = feature_vector(url)
    X = pd.DataFrame([values], columns=FEATURE_NAMES)
    prediction = int(model.predict(X)[0])
    try:
        probabilities = model.predict_proba(X)[0]
        phishing_probability = float(probabilities[list(model.classes_).index(1)])
    except Exception:
        phishing_probability = float(prediction)

    explanations, explanation_type = explain_prediction(model, FEATURE_NAMES, values)
    return jsonify({
        "prediction": "Phishing" if prediction == 1 else "Legitimate",
        "phishing_probability": round(phishing_probability * 100, 2),
        "model": bundle["model_name"],
        "features": extract_features(url),
        "explanations": explanations,
        "explanation_type": explanation_type,
        "notice": "Educational prototype only. This result is not a guarantee of website safety."
    })

@app.route("/api/metrics")
def metrics():
    _, results = load_artifacts()
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)

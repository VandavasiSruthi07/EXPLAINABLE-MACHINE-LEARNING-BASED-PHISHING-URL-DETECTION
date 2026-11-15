# Explainable Machine Learning-Based Phishing URL Detection

A runnable educational prototype with a Flask web interface, URL feature extraction,
four classifiers, model comparison, and SHAP explanations.

## Important safety / accuracy note
The included `data/sample_urls.csv` is a tiny illustrative dataset so the project can
run immediately. It is NOT suitable for real-world security decisions and will not
produce trustworthy phishing detection. Replace it with a large, verified, labelled
dataset before evaluating or presenting performance as meaningful. Never visit
suspicious URLs to test them. HTTPS, URL length, and keywords alone do not prove a
site is safe or malicious.

## Requirements
- Python 3.10–3.12 recommended
- VS Code
- Internet connection for installing packages

## Windows setup (PowerShell terminal in VS Code)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```
If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```
Then run the same `pip install`, training, and app commands.

## Open the application
Open http://127.0.0.1:5000 in your browser.

## Project files
- `app.py` – Flask backend and prediction API
- `feature_extraction.py` – extracts URL-only lexical features (does not visit URLs)
- `train_model.py` – trains and compares Logistic Regression, Random Forest, SVM, XGBoost
- `shap_explainer.py` – generates per-URL SHAP feature contributions
- `templates/index.html` – web page
- `static/style.css` – styling
- `static/script.js` – browser interactions
- `data/sample_urls.csv` – tiny demonstration dataset

## Dataset format
Create a CSV with columns:
- `url`: URL string
- `label`: 0 for legitimate, 1 for phishing

Put your verified dataset in `data/urls.csv`, then run `python train_model.py`.
The training script uses stratified splitting and reports accuracy, precision,
recall, and F1. Do not train/test on duplicated URLs; ideally split by domain/source
and use a separate, newer external test set to reduce data leakage.

## Troubleshooting
- `python` not found: try `py` instead.
- `ModuleNotFoundError`: activate `.venv`, then `pip install -r requirements.txt`.
- XGBoost installation issues: use Python 3.11/3.12 and update pip.
- SHAP is optional at runtime: if unavailable, the app shows a simple feature summary.

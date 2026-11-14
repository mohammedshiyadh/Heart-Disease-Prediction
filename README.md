
# HeartSense AI

HeartSense AI is an educational and research demonstration built around an existing heart disease machine learning project. It wraps two saved LightGBM pipelines in a Streamlit dashboard for prediction exploration, experiment comparison, and model-specific SHAP summaries. The app passes the original named feature columns to each pipeline so its fitted preprocessing remains in use.

## Project objectives

- Explore structured heart disease datasets and their differences.
- Compare five classification algorithms using the recorded evaluation settings.
- Examine generalization from the synthetic Playground dataset to the UCI dataset.
- Present model outputs and the notebook's SHAP summary in an interactive interface.

## Datasets

- **UCI Heart Disease (Cleveland)**: 303 records. The holdout experiment used a stratified 80/20 split, leaving 61 records for testing.
- **Kaggle Playground Series S6E2**: 630,000 training records in a synthetic dataset. The experiment notebook reports five-fold cross-validation.

## Algorithms evaluated

Logistic Regression, Random Forest, XGBoost, Linear SVM, and LightGBM. The Playground-trained LightGBM pipeline is the primary interactive model. The UCI-trained LightGBM pipeline is available as a secondary experimental option and was fitted on 242 training records.

## Recorded evaluation results

Scores below are copied from `results/model_comparison.csv`. Each group belongs to its own evaluation setting; scores across groups are not a shared test-set ranking.

### A. Playground five-fold cross-validation

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.51% | 88.12% | 85.96% | 87.03% | 95.27% |
| Random Forest | 88.31% | 87.48% | 86.28% | 86.88% | 94.98% |
| XGBoost | 88.78% | 88.14% | 86.64% | 87.38% | 95.48% |
| Linear SVM | 88.51% | 88.21% | 85.86% | 87.02% | 95.27% |
| LightGBM | 88.80% | 88.16% | 86.65% | 87.40% | 95.48% |

### B. UCI holdout evaluation

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 88.52% | 81.82% | 96.43% | 88.52% | 94.26% |
| LightGBM | 93.44% | 90.00% | 96.43% | 93.10% | 96.43% |

This is the stratified UCI holdout experiment: 242 records for training and 61 separate records for evaluation.

### C. Cross-dataset evaluation

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 81.85% | 79.17% | 82.01% | 80.57% | 91.73% |
| LightGBM | 82.18% | 79.72% | 82.01% | 80.85% | 91.82% |

The cross-dataset score comes from evaluating the **Playground-trained LightGBM model on all 303 UCI records**. It is a separate transfer evaluation from the UCI holdout experiment above, and must not be interpreted as another result on the 61-record holdout.

## SHAP explainability

The explainability view uses the grouped mean absolute SHAP values recorded in the notebook and displays the original notebook beeswarm image, extracted without alteration to `assets/shap_beeswarm.png`. The values were calculated on a random sample of 1,000 Playground validation rows, then summed across one-hot encoded columns for each original feature. In the beeswarm, positive values increase the Playground model's raw output toward the Presence label; negative values move it toward the Absence label. These are model-specific patterns, not causal or medical effects.

The prediction workspace also calculates a local SHAP explanation for the reviewed input using the selected saved LightGBM pipeline. It verifies the transformed features and checks additivity against that model's raw output before displaying contributions. If that check fails, the prediction can still be shown, but contribution values are withheld.

## Prediction workflow

The form uses readable category choices and measurement units instead of model category codes. Category names and encodings were checked against the [UCI Heart Disease dataset dictionary](https://archive.ics.uci.edu/dataset/45/heart+disease), the original training notebook, and the saved pipelines' fitted category lists. A private mapping layer converts selections to the original model input values; feature names, order, preprocessing, and encodings remain unchanged. Numeric ranges are bounded by values observed in the training data.

Users can type exact measurements or adjust sliders, confirm the readings match their report, review a readable summary, and explicitly start prediction. Slider starting points are only controls, not normal values or medical advice. The optional demonstration profile is illustrative only. Results include the model's Presence and Absence probabilities, the submitted-information summary, a CSV download, and (when validated) per-input feature contributions. These outputs are educational and are not clinical risk estimates.

The appearance controls are available in the sidebar across all pages. Changing the accent, page background, or card background updates the interface while retaining readable foreground text.

## Technologies

Python, Pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, Joblib, Streamlit, and Plotly. The prediction interface reuses the saved preprocessing: median imputation and standard scaling for numerical features; most-frequent imputation and one-hot encoding for categorical features.

## Project structure

```text
heart-disease-prediction/
├── app.py
├── views/
│   ├── about.py
│   ├── explainability.py
│   ├── home.py
│   ├── model_comparison.py
│   └── prediction.py
├── assets/shap_beeswarm.png
├── utils/
│   ├── model_loader.py
│   ├── explainability.py
│   ├── prediction.py
│   └── visualizations.py
├── models/
│   ├── lightgbm_playground.pkl
│   └── lightgbm_uci.pkl
├── results/model_comparison.csv
├── screenshots/
├── heart-disease-prediction.ipynb
├── requirements.txt
└── README.md
```

## Windows installation and launch

Use Python 3.10–3.13 in a clean virtual environment. This range is suitable for the pinned scikit-learn 1.6.1 pipeline artifacts and the application's Python syntax. From PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run the app using the environment's executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app expects the two model artifacts at the paths shown above. It reports a loading error if a requested artifact is missing or incompatible; it does not switch models silently. Numeric inputs are limited to the ranges observed in the training data. The UCI category dictionary is linked from the prediction workspace and from the section above.

## Application screenshots

Place genuine screenshots captured from a running application in `screenshots/` and link them here. No screenshots are included yet; none have been fabricated.

Suggested filenames:

- `screenshots/dashboard.png`
- `screenshots/prediction.png`
- `screenshots/model-comparison.png`
- `screenshots/explainability.png`

## Limitations

- The Playground dataset is synthetic and may not represent real-world patient populations.
- The UCI dataset has only 303 records; its holdout set contains 61 records.
- The cross-dataset evaluation uses all 303 UCI records and is not a held-out UCI test.
- Performance estimates depend on the data and evaluation design in the original notebook.
- The saved model probabilities have not been clinically validated or calibrated for care decisions.
- SHAP importance is model-specific and does not show causal or clinical relationships.

## Medical disclaimer

This application is for educational and research purposes only. It has not been clinically validated and must not be used for medical diagnosis, treatment decisions, or emergency assessment. Predictions and probabilities are outputs of experimental models, not clinically validated heart disease risks.
#   H e a r t - D i s e a s e - P r e d i c t i o n  
 
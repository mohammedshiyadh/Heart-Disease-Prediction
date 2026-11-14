# HeartSense AI --- Project Recap

## Project overview

HeartSense AI is an educational machine-learning research application
that explores heart disease classification using health-related input
features. It compares machine-learning models, evaluates generalisation
across datasets, and provides an interactive Streamlit interface.

> **Important:** This is an educational research prototype. It has not
> been clinically validated and must not be used for diagnosis,
> treatment decisions, or emergency assessment.

## 1. Datasets

### Kaggle Playground Series S6E2

-   630,000 training records and 270,000 test records.
-   15 training columns, including an ID and target.
-   Target labels: `Absence` and `Presence`.
-   Used as the primary training dataset.
-   This is synthetic competition data, not 630,000 real clinical
    records.

### UCI Heart Disease

-   303 records.
-   Original target converted into `Absence` and `Presence`.
-   Some features contain missing values, including `ca` and `thal`.
-   Used for a separate holdout experiment and cross-dataset evaluation.

## 2. Input features

The model uses these 13 features in the training order: 1. Age 2. Sex 3.
Chest pain type 4. BP 5. Cholesterol 6. FBS over 120 7. EKG results 8.
Max HR 9. Exercise angina 10. ST depression 11. Slope of ST 12. Number
of vessels fluro 13. Thallium

The application should display understandable labels and units. It must
map those selections internally to the exact category encodings used
during training. Verify mappings against the original notebook and
dataset; do not guess category descriptions.

## 3. Exploratory Data Analysis

EDA means Exploratory Data Analysis. It helps understand dataset
structure, quality, distributions, and relationships before training.

Typical checks include: - Dataset dimensions (`shape`). - Column types
and non-null counts (`info()`). - Statistical summaries
(`describe()`). - Missing values and duplicates. - Target-class
distribution. - Feature distributions and correlations.

Only describe plots or findings as completed if they are present in the
actual notebook.

## 4. Data preprocessing

### Numerical features

-   Age
-   BP
-   Cholesterol
-   Max HR
-   ST depression

Preprocessing: 1. Median imputation replaces missing numerical values
with the training median. 2. StandardScaler transforms values using
`z = (x - mean) / standard deviation`.

### Categorical features

-   Sex
-   Chest pain type
-   FBS over 120
-   EKG results
-   Exercise angina
-   Slope of ST
-   Number of vessels fluro
-   Thallium

Preprocessing: 1. Most-frequent imputation. 2. OneHotEncoder with
`handle_unknown="ignore"`.

### ColumnTransformer and Pipeline

A `ColumnTransformer` applies different preprocessing to numerical and
categorical columns. A Scikit-learn `Pipeline` combines the preprocessor
and classifier so the same fitted transformations are used during
prediction. Preprocessing must be fitted on training data only, not on
held-out test data.

## 5. Machine-learning algorithms

### Logistic Regression

A classification algorithm that estimates class probability using the
logistic function. It provides a useful baseline.

### Random Forest

An ensemble of decision trees. For classification, trees aggregate their
predictions. It can model nonlinear relationships.

### XGBoost

A gradient-boosting framework that builds trees sequentially, with later
trees improving the ensemble.

### Linear Support Vector Machine

Finds a separating hyperplane with a large margin between classes.
Feature scaling is important for this model.

### LightGBM

A gradient-boosting decision-tree framework designed for efficient
training. It was selected as the primary interactive model based on the
reported experiments.

### LightGBM configuration

-   `n_estimators=300`
-   `learning_rate=0.05`
-   `max_depth=6`
-   `num_leaves=31`
-   `subsample=0.8`
-   `colsample_bytree=0.8`
-   `random_state=42`
-   `n_jobs=-1`
-   `verbosity=-1`

Parameter meanings: - `n_estimators`: number of boosting
iterations/trees. - `learning_rate`: contribution size of each boosting
iteration. - `max_depth`: maximum tree depth. - `num_leaves`: maximum
leaves per tree. - `subsample`: row sampling fraction. -
`colsample_bytree`: feature sampling fraction. - `random_state`:
reproducibility control. - `n_jobs=-1`: use available CPU cores.

## 6. Model evaluation

### Confusion-matrix terms

-   True Positive (TP): predicted Presence and actual Presence.
-   True Negative (TN): predicted Absence and actual Absence.
-   False Positive (FP): predicted Presence but actual Absence.
-   False Negative (FN): predicted Absence but actual Presence.

### Metrics

-   Accuracy = `(TP + TN) / (TP + TN + FP + FN)`
-   Precision = `TP / (TP + FP)`
-   Recall = `TP / (TP + FN)`
-   F1 = `2 × (Precision × Recall) / (Precision + Recall)`
-   ROC-AUC measures how well the model ranks positive examples above
    negative examples across thresholds.

### Playground five-fold cross-validation results

  -----------------------------------------------------------------------------
  Model            Accuracy    Precision       Recall           F1      ROC-AUC
  ------------ ------------ ------------ ------------ ------------ ------------
  XGBoost            88.78%       88.14%       86.64%       87.38%       95.48%

  LightGBM       **88.80%**   **88.16%**   **86.65%**   **87.40%**   **95.48%**

  Logistic           88.51%       88.12%       85.96%       87.03%       95.27%
  Regression                                                       

  Linear SVM         88.51%       88.21%       85.86%       87.02%       95.27%

  Random             88.31%       87.48%       86.28%       86.88%       94.98%
  Forest                                                           
  -----------------------------------------------------------------------------

LightGBM's result is only marginally higher than XGBoost's. Do not claim
a large performance advantage.

### UCI holdout experiment

-   Accuracy: 93.44%
-   Precision: 90.00%
-   Recall: 96.43%
-   F1-score: 93.10%
-   ROC-AUC: 96.43%
-   Test set size: 61 records.

The test set is small, so these estimates have substantial uncertainty.

### Cross-dataset experiment

The Playground-trained LightGBM model was evaluated on all 303 UCI
records: - Accuracy: 82.18% - Precision: 79.72% - Recall: 82.01% -
F1-score: 80.85% - ROC-AUC: 91.82%

The UCI holdout and cross-dataset results are different experiments and
must not be combined.

## 7. Cross-validation and generalisation

Five-fold cross-validation divides data into five folds. Each fold is
used once for validation while the other four folds are used for
training.

Cross-dataset evaluation tests whether a model trained on one dataset
transfers to another. Performance can decrease because of differences in
data distributions, feature definitions, populations, or collection
processes.

## 8. Explainable AI --- SHAP

SHAP means **SHapley Additive exPlanations**. It uses ideas from
cooperative game theory to estimate how features contribute to a model
output.

-   Global explanation: summarises feature contributions across many
    examples.
-   Local explanation: examines contributions for one prediction.
-   SHAP values describe model behaviour, not medical causation.

Project notes indicate grouped SHAP importance was recorded using a
1,000-row validation sample. Verify the actual explainer, grouping
method, and implementation in the notebook and
`views/explainability.py`.

## 9. Saving and loading models

The fitted pipelines were saved using Joblib: -
`models/lightgbm_playground.pkl` - `models/lightgbm_uci.pkl`

Saving the fitted pipeline avoids retraining every time the app starts.
The saved object includes preprocessing and the classifier. Pickled
Scikit-learn pipelines are version-sensitive, so use compatible package
versions between training and inference environments.

## 10. Prediction workflow

1.  User enters the 13 input features.
2.  The app collects them into a Pandas DataFrame.
3.  The DataFrame uses the exact expected feature names and order.
4.  The saved pipeline performs fitted preprocessing.
5.  LightGBM predicts the class.
6.  `predict_proba()` returns model class probabilities.
7.  Streamlit displays the result.

The probability is a model output, not a clinically validated personal
disease-risk estimate.

## 11. Streamlit application

Expected project structure:

``` text
heart-disease-prediction/
├── app.py
├── views/
│   ├── __init__.py
│   ├── home.py
│   ├── prediction.py
│   ├── model_comparison.py
│   ├── explainability.py
│   └── about.py
└── models/
    ├── lightgbm_playground.pkl
    └── lightgbm_uci.pkl
```

Responsibilities: - `app.py`: application setup, navigation, and shared
theme. - `views/home.py`: project overview and quick prediction. -
`views/prediction.py`: input form and prediction. -
`views/model_comparison.py`: algorithm metrics. -
`views/explainability.py`: SHAP and feature explanation. -
`views/about.py`: project information. - `models/`: saved pipelines.

Streamlit concepts used or planned: - `st.set_page_config()`: page
configuration. - `st.sidebar`: sidebar layout. - `st.radio()`:
navigation. - `st.columns()`: column layout. - `st.metric()`: metric
cards. - `st.form()`: grouped input form. - `st.selectbox()`:
categorical inputs. - `st.number_input()`: numerical inputs. -
`st.form_submit_button()`: form submission. - `st.cache_resource`: cache
the loaded model. - `st.progress()`: visual probability display. -
`st.dataframe()`: display submitted inputs. - `st.color_picker()`:
interactive theme colors. - `st.session_state`: preserve theme settings
across reruns.

## 12. Human-friendly prediction form requirements

Do not show raw category codes such as `0`, `1`, `2`, `3`, `6`, or `7`
as user-facing categorical choices.

Use readable labels and units, then map selections internally to the
exact training encodings. Verify every mapping against the source
dataset and notebook. Do not invent medical category descriptions.

Numerical fields should show units: - Age: years. - Resting blood
pressure: mmHg. - Cholesterol: mg/dL. - Maximum heart rate: beats per
minute. - ST depression: dataset's numeric measurement.

Show a review of the entered information before prediction. Explain
unfamiliar terms in plain language. Validate inputs and ensure the final
DataFrame has the exact feature names and order expected by the saved
pipeline.

## 13. Presentation script

Good morning everyone.

Today, I am presenting HeartSense AI, a machine-learning-based heart
disease prediction and research platform.

The main objective is to explore how machine-learning algorithms can
analyse health-related features, identify patterns in datasets, and
generate heart disease predictions.

The project also compares different algorithms, evaluates their
performance, studies cross-dataset generalisation, and uses explainable
AI to investigate model behaviour.

For this project, I used two datasets: the Kaggle Playground Series S6E2
dataset, containing 630,000 training records, and the UCI Heart Disease
dataset, containing 303 records. The Playground dataset is synthetic
competition data, while the UCI dataset is a smaller traditional
research dataset.

After collecting the data, I performed exploratory data analysis to
understand the dataset structure, data types, missing values,
distributions, and target classes.

Next, I performed preprocessing. For numerical features, I used median
imputation and StandardScaler. For categorical features, I used
most-frequent imputation and OneHotEncoder. I combined these
transformations using ColumnTransformer and a Scikit-learn Pipeline.

I evaluated five algorithms: Logistic Regression, Random Forest,
XGBoost, Linear SVM, and LightGBM.

For model evaluation, I used accuracy, precision, recall, F1-score, and
ROC-AUC. I also used five-fold cross-validation for the Playground model
comparison.

LightGBM achieved the highest reported Playground cross-validation
accuracy of 88.80%, with an F1-score of 87.40% and ROC-AUC of 95.48%.
XGBoost performed very closely.

I also conducted a UCI holdout experiment and a separate cross-dataset
experiment. The UCI holdout accuracy was 93.44% on 61 test records. The
Playground-trained model achieved 82.18% accuracy when evaluated on all
303 UCI records. These are separate experiments.

For explainability, I used SHAP to estimate how input features
contribute to model outputs. SHAP helps explain model behaviour, but it
does not establish medical causation.

After training, I saved the fitted pipelines using Joblib. I then built
the interactive application using Streamlit. The application provides an
overview, prediction interface, model comparison, explainability
section, and project information.

During prediction, the app collects user inputs in a DataFrame, passes
them through the saved preprocessing pipeline, and uses LightGBM to
produce a predicted class and class probabilities.

To conclude, HeartSense AI demonstrates a machine-learning workflow,
from dataset exploration and preprocessing to model evaluation,
explainability, and an interactive application.

The project is an educational research prototype and has not been
clinically validated. Its predictions must not be used for medical
diagnosis or treatment decisions.

Thank you.

## 14. Viva questions to practise

1.  What is supervised learning?
2.  Why is this a classification problem?
3.  What is the difference between numerical and categorical features?
4.  Why did you use median imputation?
5.  What does StandardScaler do?
6.  What is one-hot encoding?
7.  Why did you use ColumnTransformer?
8.  What is the purpose of a Pipeline?
9.  What is data leakage?
10. How does Logistic Regression work?
11. What is an ensemble model?
12. How does Random Forest work?
13. What is gradient boosting?
14. What is the difference between XGBoost and LightGBM?
15. What is a hyperparameter?
16. What is five-fold cross-validation?
17. What is the difference between precision and recall?
18. What does ROC-AUC measure?
19. What is cross-dataset generalisation?
20. What is SHAP?
21. Does SHAP prove causation?
22. Why did you save the pipeline using Joblib?
23. What is the difference between `predict()` and `predict_proba()`?
24. Why should category encodings be hidden from users?
25. Why is this application not a medical diagnostic tool?

## 15. Final verification checklist

Before presenting, verify these details in the actual notebook and
application: - Exact UCI target conversion. - Exact train/validation
split and random states. - Whether preprocessing was inside each
cross-validation fold. - Exact UCI holdout procedure. - Exact SHAP
explainer and feature-grouping method. - Exact categorical mappings. -
Actual class order in the saved model. - Whether every page and theme
control works. - Whether the prediction page loads the intended saved
model.

Do not claim a feature is complete unless you can demonstrate it in the
code.

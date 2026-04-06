# ADC Toxicity Classifier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train ML models to predict ADC toxicity (regression: %G≥3 observed; classification: binary G≥3 >10%) from structural ADC features, with a Streamlit app for interactive prediction.

**Architecture:** A `src/` package handles data loading, preprocessing, training, and evaluation. A root-level `notebook.ipynb` orchestrates the full pipeline using those modules. `app.py` loads the saved models and preprocessor to serve Streamlit predictions.

**Tech Stack:** Python 3.10+, pandas, scikit-learn, xgboost, shap, streamlit, joblib, openpyxl, pytest

---

## Key Data Facts (read before coding)

- Source file: `TADC_Complete_v3_avec_S.xlsx`, sheet `RÉSUMÉ_Régression`
- 106 valid rows (14 ADCs × 6–10 organ/EI rows each) + 8 metadata rows at bottom to drop
- Valid ADC names: `['Mylotarg','Adcetris','Kadcyla','Besponsa','Polivy','Padcev','Enhertu','Trodelvy','Blenrep','Zynlonta','Tivdak','Elahere','Datroway','Teliso-V']`
- Features: `P, D, H, B, L, E, V` (numeric) + `S(payload,organe)` (numeric) + `Payload class`, `Organe` (categorical)
- Targets: `%G≥3 observé` (regression float), `Y binaire (G≥3 >10%)` (binary 0/1)
- Baseline column: `T-ADC v3 = Σ×V×S` (formula-computed, use as comparison)
- Class imbalance: 84 negative / 22 positive (use `class_weight='balanced'` or `scale_pos_weight`)
- LOAOCV: 14 folds (one per ADC), group = `ADC` column

---

## File Structure

```
adc_toxicity_classifier/
├── data/
│   └── TADC_Complete_v3_avec_S.xlsx       (existing)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                      (load + clean Excel)
│   ├── preprocessor.py                     (encode + scale features)
│   └── train.py                            (LOAOCV + model training)
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_train.py
├── models/
│   ├── model_regression.pkl
│   ├── model_classification.pkl
│   └── preprocessor.pkl
├── notebook.ipynb
├── app.py
└── requirements.txt
```

---

## Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `models/.gitkeep`

- [ ] **Step 1: Create `requirements.txt`**

```
pandas==2.2.3
openpyxl==3.1.5
scikit-learn==1.5.2
xgboost==2.1.3
shap==0.46.0
streamlit==1.40.2
joblib==1.4.2
matplotlib==3.9.3
pytest==8.3.4
numpy==1.26.4
```

- [ ] **Step 2: Install dependencies**

Run: `pip3 install -r requirements.txt`
Expected: All packages install without errors.

- [ ] **Step 3: Create empty init files and models dir**

```bash
mkdir -p src tests models data
touch src/__init__.py tests/__init__.py models/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git init
git add requirements.txt src/__init__.py tests/__init__.py models/.gitkeep
git commit -m "chore: project scaffold"
```

---

## Task 2: Data loader

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_loader.py`:

```python
import pandas as pd
import pytest
from src.data_loader import load_data

VALID_ADCS = [
    'Mylotarg','Adcetris','Kadcyla','Besponsa','Polivy','Padcev',
    'Enhertu','Trodelvy','Blenrep','Zynlonta','Tivdak','Elahere',
    'Datroway','Teliso-V'
]

def test_load_returns_dataframe():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    assert isinstance(df, pd.DataFrame)

def test_load_drops_metadata_rows():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    assert df['ADC'].isin(VALID_ADCS).all()

def test_load_shape():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    assert df.shape == (106, 26)

def test_required_columns_present():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    required = ['ADC', 'P', 'D', 'H', 'B', 'L', 'E', 'V',
                'S(payload,organe)', 'Payload class', 'Organe',
                '%G≥3 observé', 'Y binaire (G≥3 >10%)',
                'T-ADC v3 = Σ×V×S']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_no_nulls_in_features():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    feature_cols = ['P', 'D', 'H', 'B', 'L', 'E', 'V',
                    'S(payload,organe)', 'Payload class', 'Organe']
    assert df[feature_cols].isnull().sum().sum() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_loader.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_data'`

- [ ] **Step 3: Implement `src/data_loader.py`**

```python
import pandas as pd
import openpyxl

VALID_ADCS = [
    'Mylotarg', 'Adcetris', 'Kadcyla', 'Besponsa', 'Polivy', 'Padcev',
    'Enhertu', 'Trodelvy', 'Blenrep', 'Zynlonta', 'Tivdak', 'Elahere',
    'Datroway', 'Teliso-V'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean the RÉSUMÉ_Régression sheet.

    Uses data_only=True to read cached formula values.
    Drops metadata rows at the bottom (non-ADC rows).
    """
    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb['RÉSUMÉ_Régression']

    headers = [cell.value for cell in ws[1]]
    rows = [
        [cell for cell in row]
        for row in ws.iter_rows(min_row=2, values_only=True)
    ]
    df = pd.DataFrame(rows, columns=headers)

    # Keep only valid ADC rows
    df = df[df['ADC'].isin(VALID_ADCS)].reset_index(drop=True)

    # Cast numeric columns
    numeric_cols = ['P', 'D', 'H', 'B', 'L', 'E', 'V',
                    'S(payload,organe)', '%G≥3 observé',
                    'Y binaire (G≥3 >10%)', 'T-ADC v3 = Σ×V×S']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_loader.py -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: data loader with metadata row filtering"
```

---

## Task 3: Preprocessor

**Files:**
- Create: `src/preprocessor.py`
- Create: `tests/test_preprocessor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preprocessor.py`:

```python
import pandas as pd
import numpy as np
import pytest
from src.data_loader import load_data
from src.preprocessor import build_features, FEATURE_COLS, TARGET_REG, TARGET_CLF

def test_build_features_returns_array():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    X, y_reg, y_clf = build_features(df)
    assert hasattr(X, 'shape')
    assert len(X) == 106

def test_no_nulls_after_preprocessing():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    X, y_reg, y_clf = build_features(df)
    assert not np.isnan(X).any()

def test_targets_correct_length():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    X, y_reg, y_clf = build_features(df)
    assert len(y_reg) == 106
    assert len(y_clf) == 106

def test_clf_target_is_binary():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    X, y_reg, y_clf = build_features(df)
    assert set(y_clf).issubset({0, 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessor.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_features'`

- [ ] **Step 3: Implement `src/preprocessor.py`**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = ['P', 'D', 'H', 'B', 'L', 'E', 'V', 'S(payload,organe)']
CATEGORICAL_FEATURES = ['Payload class', 'Organe']
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REG = '%G≥3 observé'
TARGET_CLF = 'Y binaire (G≥3 >10%)'


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for feature encoding."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )


def build_features(df: pd.DataFrame):
    """Fit-transform features and return (X_array, y_reg, y_clf).

    For use in the notebook only (fits on full data).
    For LOAOCV, use build_preprocessor() inside each fold.
    """
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df[FEATURE_COLS])
    y_reg = df[TARGET_REG].values.astype(float)
    y_clf = df[TARGET_CLF].values.astype(int)
    return X, y_reg, y_clf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocessor.py -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/preprocessor.py tests/test_preprocessor.py
git commit -m "feat: preprocessor with StandardScaler + OneHotEncoder"
```

---

## Task 4: LOAOCV training pipeline

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import numpy as np
import pytest
from src.data_loader import load_data
from src.train import run_loaocv

def test_loaocv_returns_expected_keys():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    results = run_loaocv(df, task='regression')
    assert 'mae' in results
    assert 'rmse' in results
    assert 'r2' in results
    assert 'y_true' in results
    assert 'y_pred' in results

def test_loaocv_clf_returns_expected_keys():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    results = run_loaocv(df, task='classification')
    assert 'auc' in results
    assert 'f1' in results
    assert 'accuracy' in results

def test_loaocv_pred_length_matches_data():
    df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
    results = run_loaocv(df, task='regression')
    assert len(results['y_pred']) == len(df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_loaocv'`

- [ ] **Step 3: Implement `src/train.py`**

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                              r2_score, roc_auc_score, f1_score, accuracy_score)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, XGBClassifier
from src.preprocessor import build_preprocessor, FEATURE_COLS, TARGET_REG, TARGET_CLF

RF_REG_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 5, None]}
RF_CLF_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 5, None],
                 'class_weight': ['balanced']}
XGB_REG_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 4], 'learning_rate': [0.05, 0.1]}
XGB_CLF_PARAMS = {'n_estimators': [100, 200], 'max_depth': [3, 4], 'learning_rate': [0.05, 0.1],
                  'scale_pos_weight': [3.8]}  # 84/22 ≈ 3.8


def _get_models(task: str):
    if task == 'regression':
        return [
            ('RandomForest', RandomForestRegressor(random_state=42), RF_REG_PARAMS),
            ('XGBoost', XGBRegressor(random_state=42, verbosity=0), XGB_REG_PARAMS),
        ]
    return [
        ('RandomForest', RandomForestClassifier(random_state=42), RF_CLF_PARAMS),
        ('XGBoost', XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False,
                                   eval_metric='logloss'), XGB_CLF_PARAMS),
    ]


def run_loaocv(df: pd.DataFrame, task: str = 'regression') -> dict:
    """Leave-One-ADC-Out cross-validation.

    Args:
        df: cleaned dataframe from load_data()
        task: 'regression' or 'classification'

    Returns:
        dict with aggregate metrics and y_true/y_pred arrays.
    """
    adcs = df['ADC'].unique()
    target_col = TARGET_REG if task == 'regression' else TARGET_CLF

    y_true_all, y_pred_all, idx_all = [], [], []

    for adc in adcs:
        train_mask = df['ADC'] != adc
        test_mask = df['ADC'] == adc

        X_train_raw = df.loc[train_mask, FEATURE_COLS]
        X_test_raw = df.loc[test_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, target_col].values
        y_test = df.loc[test_mask, target_col].values

        preprocessor = build_preprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        best_score = -np.inf
        best_pred = None

        for name, model, params in _get_models(task):
            scoring = 'neg_mean_absolute_error' if task == 'regression' else 'roc_auc'
            cv_folds = min(3, len(np.unique(df.loc[train_mask, 'ADC'])))
            gs = GridSearchCV(model, params, cv=cv_folds, scoring=scoring, n_jobs=-1)
            gs.fit(X_train, y_train)
            score = gs.best_score_
            if score > best_score:
                best_score = score
                if task == 'regression':
                    best_pred = gs.best_estimator_.predict(X_test)
                else:
                    best_pred = gs.best_estimator_.predict_proba(X_test)[:, 1]

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(best_pred.tolist())
        idx_all.extend(df.index[test_mask].tolist())

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    if task == 'regression':
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': float(root_mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred,
        }
    else:
        y_pred_binary = (y_pred >= 0.5).astype(int)
        return {
            'auc': roc_auc_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'y_true': y_true,
            'y_pred': y_pred,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: 3 tests PASS (may take ~30s due to GridSearchCV)

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: LOAOCV pipeline with RandomForest and XGBoost"
```

---

## Task 5: Full model training & export

**Files:**
- Create: `notebook.ipynb`
- Modify: `models/` (write pkl files)

- [ ] **Step 1: Create `notebook.ipynb` with cells below**

Cell 1 — Imports & load:
```python
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error,
                              r2_score, roc_auc_score, f1_score)
from src.data_loader import load_data
from src.preprocessor import build_preprocessor, FEATURE_COLS, TARGET_REG, TARGET_CLF
from src.train import run_loaocv, RF_REG_PARAMS, RF_CLF_PARAMS, XGB_REG_PARAMS, XGB_CLF_PARAMS

df = load_data('data/TADC_Complete_v3_avec_S.xlsx')
print(df.shape)
df[['P','D','H','B','L','E','V','S(payload,organe)','%G≥3 observé','Y binaire (G≥3 >10%)']].describe()
```

Cell 2 — Class distribution:
```python
print("Class distribution (Y binaire):")
print(df['Y binaire (G≥3 >10%)'].value_counts())
print("\nADC row counts:")
print(df['ADC'].value_counts())
```

Cell 3 — LOAOCV regression:
```python
reg_results = run_loaocv(df, task='regression')
print(f"Regression LOAOCV — MAE: {reg_results['mae']:.2f}  RMSE: {reg_results['rmse']:.2f}  R²: {reg_results['r2']:.3f}")
```

Cell 4 — LOAOCV classification:
```python
clf_results = run_loaocv(df, task='classification')
print(f"Classification LOAOCV — AUC: {clf_results['auc']:.3f}  F1: {clf_results['f1']:.3f}  Acc: {clf_results['accuracy']:.3f}")
```

Cell 5 — Baseline comparison (T-ADC v3 formula):
```python
from sklearn.metrics import mean_absolute_error, roc_auc_score
baseline_pred_reg = df['T-ADC v3 = Σ×V×S'].values
baseline_true_reg = df[TARGET_REG].values
baseline_mae = mean_absolute_error(baseline_true_reg, baseline_pred_reg)
print(f"Baseline (T-ADC v3) MAE: {baseline_mae:.2f}")
print(f"ML LOAOCV MAE:          {reg_results['mae']:.2f}")
```

Cell 6 — Train final models on full data & export:
```python
preprocessor = build_preprocessor()
X_all = preprocessor.fit_transform(df[FEATURE_COLS])
y_reg_all = df[TARGET_REG].values.astype(float)
y_clf_all = df[TARGET_CLF].values.astype(int)

# Regression
gs_reg = GridSearchCV(
    RandomForestRegressor(random_state=42),
    RF_REG_PARAMS, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)
gs_reg.fit(X_all, y_reg_all)
best_reg = gs_reg.best_estimator_

# Classification
gs_clf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    RF_CLF_PARAMS, cv=5, scoring='roc_auc', n_jobs=-1
)
gs_clf.fit(X_all, y_clf_all)
best_clf = gs_clf.best_estimator_

joblib.dump(best_reg, 'models/model_regression.pkl')
joblib.dump(best_clf, 'models/model_classification.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Models saved.")
```

Cell 7 — SHAP summary (regression):
```python
explainer_reg = shap.TreeExplainer(best_reg)
shap_values_reg = explainer_reg.shap_values(X_all)

# Get feature names after one-hot encoding
feature_names = (
    list(preprocessor.named_transformers_['num'].get_feature_names_out())
    + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
)

shap.summary_plot(shap_values_reg, X_all, feature_names=feature_names, show=True)
```

Cell 8 — SHAP summary (classification):
```python
explainer_clf = shap.TreeExplainer(best_clf)
shap_values_clf = explainer_clf.shap_values(X_all)
if isinstance(shap_values_clf, list):
    shap_values_clf = shap_values_clf[1]  # positive class

shap.summary_plot(shap_values_clf, X_all, feature_names=feature_names, show=True)
```

- [ ] **Step 2: Run all cells**

Run via Jupyter or: `jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb`
Expected: All cells complete without errors, `models/` now contains 3 pkl files.

- [ ] **Step 3: Commit**

```bash
git add notebook.ipynb models/model_regression.pkl models/model_classification.pkl models/preprocessor.pkl
git commit -m "feat: train final models and export to models/"
```

---

## Task 6: Streamlit app

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py`**

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.preprocessor import FEATURE_COLS

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg = joblib.load('models/model_regression.pkl')
    clf = joblib.load('models/model_classification.pkl')
    pre = joblib.load('models/preprocessor.pkl')
    return reg, clf, pre

@st.cache_data
def load_dataset():
    return load_data('data/TADC_Complete_v3_avec_S.xlsx')

reg_model, clf_model, preprocessor = load_models()
df = load_dataset()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.title("ADC Toxicity Classifier")
st.sidebar.header("Paramètres ADC")

P = st.sidebar.slider("P — Puissance payload", 1.0, 3.0, 1.5, 0.5)
D = st.sidebar.slider("D — DAR", 0.5, 3.0, 1.5, 0.5)
H = st.sidebar.slider("H — Hydrophobie", 0.5, 2.0, 1.0, 0.5)
B = st.sidebar.slider("B — Effet bystander", 0.5, 2.0, 1.0, 0.5)
L = st.sidebar.slider("L — Stabilité linker", 0.5, 3.0, 1.5, 0.5)
E = st.sidebar.slider("E — Exposition", 0.5, 2.5, 1.0, 0.5)
V = st.sidebar.slider("V — Valeur organe", 1.0, 2.0, 1.3, 0.1)
S = st.sidebar.slider("S — Sensibilité patient", 0.2, 2.81, 1.0, 0.1)

payload_options = sorted(df['Payload class'].dropna().unique().tolist())
organe_options = sorted(df['Organe'].dropna().unique().tolist())
payload = st.sidebar.selectbox("Payload class", payload_options)
organe = st.sidebar.selectbox("Organe", organe_options)

# ── Prediction ────────────────────────────────────────────────────────────────
if st.sidebar.button("Prédire la toxicité"):
    input_df = pd.DataFrame([{
        'P': P, 'D': D, 'H': H, 'B': B, 'L': L, 'E': E,
        'V': V, 'S(payload,organe)': S,
        'Payload class': payload, 'Organe': organe,
    }])

    X_input = preprocessor.transform(input_df[FEATURE_COLS])

    pred_reg = float(reg_model.predict(X_input)[0])
    pred_prob = float(clf_model.predict_proba(X_input)[0][1])
    pred_binary = int(pred_prob >= 0.5)

    # T-ADC v3 formula baseline
    weighted_sum = 5*P + 1*D + 1*H + 1.5*B + 0.5*L + 0.5*E
    tadc_v3 = weighted_sum * V * S

    # ── Display results ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("% G≥3 prédit (ML)", f"{pred_reg:.1f}%")
    col2.metric("T-ADC v3 (formule)", f"{tadc_v3:.1f}")

    if pred_binary == 1:
        col3.markdown("### :red[Risque ÉLEVÉ]")
    else:
        col3.markdown("### :green[Risque FAIBLE]")

    st.progress(min(pred_prob, 1.0), text=f"Probabilité G≥3 >10% : {pred_prob:.0%}")

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    st.subheader("Interprétabilité (SHAP)")
    feature_names = (
        list(preprocessor.named_transformers_['num'].get_feature_names_out())
        + list(preprocessor.named_transformers_['cat'].get_feature_names_out())
    )
    explainer = shap.TreeExplainer(reg_model)
    shap_vals = explainer.shap_values(X_input)
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_input[0],
            feature_names=feature_names,
        ),
        show=False,
    )
    st.pyplot(fig)
    plt.close()
```

- [ ] **Step 2: Run the app locally**

Run: `streamlit run app.py`
Expected: Browser opens at `http://localhost:8501`. Sliders visible in sidebar. After clicking "Prédire la toxicité", three metrics appear + SHAP waterfall plot.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit app with SHAP waterfall and risk badge"
```

---

## Task 7: Run all tests

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected:
```
tests/test_data_loader.py::test_load_returns_dataframe PASSED
tests/test_data_loader.py::test_load_drops_metadata_rows PASSED
tests/test_data_loader.py::test_load_shape PASSED
tests/test_data_loader.py::test_required_columns_present PASSED
tests/test_data_loader.py::test_no_nulls_in_features PASSED
tests/test_preprocessor.py::test_build_features_returns_array PASSED
tests/test_preprocessor.py::test_no_nulls_after_preprocessing PASSED
tests/test_preprocessor.py::test_targets_correct_length PASSED
tests/test_preprocessor.py::test_clf_target_is_binary PASSED
tests/test_train.py::test_loaocv_returns_expected_keys PASSED
tests/test_train.py::test_loaocv_clf_returns_expected_keys PASSED
tests/test_train.py::test_loaocv_pred_length_matches_data PASSED
```

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: all tests passing"
```

---

## Known Limitations

- With only 14 ADCs in LOAOCV, R² and AUC confidence intervals are very wide — report raw metrics but don't over-interpret.
- The class imbalance (84/22) means F1 on minority class is the most meaningful metric, not accuracy.
- `root_mean_squared_error` requires scikit-learn ≥ 1.4. If you get an ImportError, use `np.sqrt(mean_squared_error(...))` instead.

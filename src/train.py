import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
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
        ('XGBoost', XGBClassifier(random_state=42, verbosity=0,
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

    y_true_all, y_pred_all = [], []

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

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    if task == 'regression':
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
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

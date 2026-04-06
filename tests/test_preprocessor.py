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

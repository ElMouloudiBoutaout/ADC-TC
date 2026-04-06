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

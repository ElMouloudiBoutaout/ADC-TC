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

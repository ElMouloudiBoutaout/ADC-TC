import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

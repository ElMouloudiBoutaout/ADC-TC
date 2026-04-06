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

    Adds interaction features: P*D, V*S, and E/L (with safety epsilon).
    """
    df = df.copy()
    
    # Feature Engineering (Synergy)
    df['P_D'] = df['P'] * df['D']
    df['V_S'] = df['V'] * df['S(payload,organe)']
    df['E_L'] = df['E'] / (df['L'] + 1e-6) # Higher E and lower L = high toxicity
    
    # Update feature lists
    numeric_features = NUMERIC_FEATURES + ['P_D', 'V_S', 'E_L']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )
    
    X = preprocessor.fit_transform(df[FEATURE_COLS + ['P_D', 'V_S', 'E_L']])
    y_reg = df[TARGET_REG].values.astype(float)
    y_clf = df[TARGET_CLF].values.astype(int)
    return X, y_reg, y_clf

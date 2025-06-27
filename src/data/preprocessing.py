import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_features(df: pd.DataFrame, feature_cols: list):
    """Standardize features & drop NA rows."""
    df = df.dropna(subset=feature_cols)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def validate_features(df: pd.DataFrame, feature_cols: list):
    """Ensure features are numeric and have no missing values."""
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"
    assert not df[feature_cols].isna().any().any(), "Features contain NaN"
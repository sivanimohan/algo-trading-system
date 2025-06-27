import numpy as np

def predict_next(model, scaler, df, feature_cols):
    """
    Predict the next day's movement (1=UP, 0=DOWN) using the most recent row of df.
    """
    latest = df[feature_cols].iloc[[-1]]
    X_latest = scaler.transform(latest)
    pred = model.predict(X_latest)
    return int(pred[0])  # 1=UP, 0=DOWN
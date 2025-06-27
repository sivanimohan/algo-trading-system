from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd

def train_ml_pipeline(df, model_type="decision_tree"):
    """
    Train a classification model on historical data to predict next-day movement.
    Supports: decision_tree, logistic_regression, random_forest, svm, xgboost, lightgbm, mlp
    Returns: fitted model, scaler, accuracy, feature columns, test X/y, prediction, probas, report, confusion matrix
    """
    exclude_cols = ['Date', 'date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Symbol']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    if not feature_cols or 'Close' not in df:
        raise ValueError("Not enough features or missing 'Close' column for ML pipeline.")

    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna(subset=feature_cols+['target'])

    X = df[feature_cols].values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection
    if model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "svm":
        model = SVC(probability=True)
    elif model_type == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_type == "lightgbm":
        model = LGBMClassifier()
    elif model_type == "mlp":
        model = MLPClassifier(max_iter=1000)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    # Probability of predicting UP
    try:
        y_proba = model.predict_proba(X_test_scaled)[:,1]
    except Exception:
        y_proba = None
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return model, scaler, acc, feature_cols, X_test_scaled, y_test, y_pred, y_proba, report, cm
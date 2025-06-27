from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def get_model(model_type='decision_tree'):
    """Return a scikit-learn classifier by type."""
    if model_type == 'decision_tree':
        return DecisionTreeClassifier(max_depth=4, random_state=42)
    elif model_type == 'logistic_regression':
        return LogisticRegression(max_iter=500)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
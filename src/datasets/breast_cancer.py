import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    """
    Breast Cancer Wisconsin (Diagnostic)

    Task:
        Classify tumors as malignant or benign.

    Positive class:
        Malignant

    Returns:
        X : pandas.DataFrame
        y : pandas.Series (binary, 0/1)
    """
    data = load_breast_cancer(as_frame=True)

    X = data.data.copy()

    # In sklearn: 0 = malignant, 1 = benign
    # We convert malignant to positive class (1)
    y = (data.target == 0).astype(int)

    return X, y

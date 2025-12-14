import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Heart Disease Dataset (Cleveland)

    Task:
        Predict presence of heart disease.

    Positive class:
        disease present (target > 0)

    Returns:
        X : pandas.DataFrame
        y : pandas.Series (binary, 0/1)
    """
    data = fetch_openml(name="heart-disease", version=1, as_frame=True)

    df = data.frame.copy()

    # Target column is usually named 'target'
    # Values: 0 = no disease, 1-4 = disease present
    y = (df["target"].astype(int) > 0).astype(int)

    X = df.drop(columns=["target"])

    return X, y

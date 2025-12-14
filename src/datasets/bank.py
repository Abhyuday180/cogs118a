import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Bank Marketing Dataset (UCI)

    Task:
        Predict whether a client subscribes to a term deposit.

    Positive class:
        y == 'yes'

    Returns:
        X : pandas.DataFrame
        y : pandas.Series (binary, 0/1)
    """
    data = fetch_openml(name="bank-marketing", version=1, as_frame=True)

    df = data.frame.copy()

    # Target
    y = (df["y"] == "yes").astype(int)

    # Features
    X = df.drop(columns=["y"])

    return X, y

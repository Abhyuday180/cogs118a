import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Adult Income Dataset (UCI)

    Task:
        Predict whether income > $50K per year.

    Positive class:
        income == '>50K'

    Returns:
        X : pandas.DataFrame
        y : pandas.Series (binary, 0/1)
    """
    data = fetch_openml(name="adult", version=2, as_frame=True)

    df = data.frame.copy()

    # Target
    y = (df["class"] == ">50K").astype(int)

    # Features
    X = df.drop(columns=["class"])

    return X, y

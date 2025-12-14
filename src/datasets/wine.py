import pandas as pd
from sklearn.datasets import fetch_openml


def load_data():
    """
    Wine Quality (Red) Dataset (UCI)

    Task:
        Predict wine quality (binary version).

    Positive class:
        quality >= 6 (good wine)

    Returns:
        X : pandas.DataFrame
        y : pandas.Series (binary, 0/1)
    """
    data = fetch_openml(name="wine-quality-red", version=1, as_frame=True)

    df = data.frame.copy()

    # Target
    y = (df["quality"].astype(int) >= 6).astype(int)

    X = df.drop(columns=["quality"])

    return X, y

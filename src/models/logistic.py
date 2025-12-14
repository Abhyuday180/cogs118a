from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.models.preprocessing import make_preprocessor


def get_model_and_params(X=None):
    preprocessor = make_preprocessor(X)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=500
        ))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    return model, param_grid

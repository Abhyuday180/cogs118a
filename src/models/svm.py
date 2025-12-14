from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.models.preprocessing import make_preprocessor


def get_model_and_params(X=None):
    preprocessor = make_preprocessor(X)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", SVC(kernel="rbf"))
    ])

    param_grid = {
        "clf__C": [1.0, 10.0],
        "clf__gamma": [0.01, 0.1]
    }

    return model, param_grid

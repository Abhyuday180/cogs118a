from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from src.models.preprocessing import make_preprocessor


def get_model_and_params(X=None):
    preprocessor = make_preprocessor(X)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", KNeighborsClassifier())
    ])

    param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 15],
        "clf__weights": ["uniform", "distance"]
    }

    return model, param_grid

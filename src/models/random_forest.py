from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.models.preprocessing import make_preprocessor


def get_model_and_params(X=None):
    preprocessor = make_preprocessor(X)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(
            random_state=0,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 20],
        "clf__max_features": ["sqrt", "log2"]
    }

    return model, param_grid

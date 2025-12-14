from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.models.preprocessing import make_preprocessor


def get_model_and_params(X=None):
    preprocessor = make_preprocessor(X)

    base_estimator = DecisionTreeClassifier(random_state=0)

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", AdaBoostClassifier(
            estimator=base_estimator,
            algorithm="SAMME",
            random_state=0
        ))
    ])

    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.1, 1.0],
        "clf__estimator__max_depth": [1, 2, 3]
    }

    return model, param_grid

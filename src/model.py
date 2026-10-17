"""Model construction: the model alone (build_model) and preprocessing + model (build_pipeline)."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def adaboost(max_depth: int = 1, **params):
    """AdaBoost on decision trees. max_depth=1 means stumps. The depth is tuned as model__estimator__max_depth."""
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    Forest of oblique trees. Every tree first projects the features onto n_components random
    linear combinations (new ones per tree) and then splits on those, so its splits are
    slanted lines instead of axis-parallel ones. Built from a bagging of (projection + tree).
    Tuned as model__estimator__tree__<param> and model__estimator__project__<param>.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
}


def build_model(model_config: dict):
    """Builds the model named in model_config["type"] with model_config["params"]."""
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    Preprocessing + model as one Pipeline, so everything that learns from data is re-fit on
    the training part of each fold. Hyperparameters are addressed by step name, e.g.
    model__max_depth. A new learned step (feature selection etc.) goes between "prep" and "model".
    """
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])

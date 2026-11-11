"""Predicting new rows with the final refit pipeline (rows that have no label)."""
import os

import pandas as pd

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target


def predict_new_rows(pipeline, predict_config: dict, diagnostics_config: dict, data_config: dict,
                     mnar_indicator_sources: list) -> pd.DataFrame:
    """
    Cleans and predicts the rows in `predict_config["path"]` with the (already refit) pipeline and
    writes `predict_config["output"]`: the id, the probability of class 1 and the 0/1 prediction
    (class 1 if the probability >= `threshold`, default 0.5). The rows get the same cleaning as the
    training data, but no de-duplication: every row needs a prediction, so rows out == rows in.
    """
    id_column = predict_config["id_column"]
    threshold = predict_config.get("threshold", 0.5)

    df_raw = load_data(predict_config["path"])
    df_clean = clean_dataset(df_raw, diagnostics_config)
    X_new, _, _ = split_features_target(df_clean, data_config, mnar_indicator_sources)

    probability = pipeline.predict_proba(X_new)[:, 1]
    predictions = pd.DataFrame({id_column: df_clean[id_column].to_numpy(), "probability": probability,
                                "prediction": (probability >= threshold).astype(int)})
    if len(predictions) != len(df_raw):
        raise RuntimeError(f"{len(df_raw)} rows in, {len(predictions)} predictions out: no row may be dropped")

    os.makedirs(os.path.dirname(predict_config["output"]) or ".", exist_ok=True)
    predictions.to_csv(predict_config["output"], index=False)
    return predictions

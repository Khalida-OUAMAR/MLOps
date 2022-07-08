import pandas as pd
from typing import Any, Dict
from xgboost import XGBRegressor
import mlflow
import os

# import pickle
# from datetime import datetime


def train_model(
    train_x: pd.DataFrame,
    train_y: pd.DataFrame,
    test_x: pd.DataFrame,
    parameters: Dict[str, Any],
) -> None:
    # Create the column transformer for one-hot encoding
    _n_estimators = parameters["n_estimators"]
    _max_depth = parameters["max_depth"]
    _verbosity = parameters["verbosity"]

    model = XGBRegressor(
        n_estimators=_n_estimators, max_depth=_max_depth, verbosity=_verbosity
    )
    model.fit(train_x, train_y)

    model.predict(train_x)
    model.predict(test_x)
    # date = datetime.today().strftime('%Y-%m-%d')
    # filename_save_model = f"data/06_models/model_{date}.pkl"
    # pickle.dump(model, open(filename_save_model, "wb"))
    return model


def auto_ml(
    model: XGBRegressor, log_to_mlflow: bool = False, experiment_id: int = -1
) -> str:

    run_id = ""
    if log_to_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))
        run = mlflow.start_run(experiment_id=experiment_id)
        run_id = run.info.run_id
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

    return run_id

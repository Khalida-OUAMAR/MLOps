import pandas as pd
from typing import Any, Dict
from xgboost import XGBRegressor


def train_model(train_x: pd.DataFrame, train_y: pd.DataFrame, test_x: pd.DataFrame, parameters: Dict[str, Any]) -> None :
    # Create the column transformer for one-hot encoding    
    _n_estimators = parameters["n_estimators"]
    _max_depth = parameters["max_depth"]
    _verbosity = parameters["verbosity"]
    

    model = XGBRegressor(n_estimators=_n_estimators, max_depth=_max_depth, verbosity=_verbosity)
    model.fit(train_x, train_y)

    model.predict(train_x)
    model.predict(test_x)
    return model
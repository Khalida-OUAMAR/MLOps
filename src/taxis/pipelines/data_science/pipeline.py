from kedro.pipeline import Pipeline, node

from .nodes import train_model, auto_ml  # , evaluate,predict


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["train_x", "train_y", "test_x", "parameters"],
                "model",
                name="train",
            ),
            node(
                auto_ml,
                ["model", "params:mlflow_enabled", "params:mlflow_experiment_id"],
                "mlflow_run_id",
                name="automl",
            ),
        ]
    )

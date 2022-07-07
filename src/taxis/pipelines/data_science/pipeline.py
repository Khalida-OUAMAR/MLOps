from kedro.pipeline import Pipeline, node 

from .nodes import train_model#, evaluate,predict


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["train_x", "train_y", "test_x", "parameters"],
                "model",
                name="train"
            )
            # node(
            #     evaluate,
            #     ["model", "test_x", "test_y"],
            #     ["test_acc", "test_loss"],
            #     name="evaluate"
            # ),
            # node(
            #     predict,
            #     ["model", "test_x"],
            #     "predictions",
            #     name="predict"
            # )
        ]
    )
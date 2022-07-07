from kedro.pipeline import Pipeline, node 

from .nodes import load_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                load_data,
                inputs='dataset_taxis',
                outputs=dict(
                    train_x="train_x",
                    train_y="train_y",
                    test_x="test_x",
                    test_y="test_y"
                ),
                name='load'
            )
        ]
    )
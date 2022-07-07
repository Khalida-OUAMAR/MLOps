"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from taxis.pipelines.data_preparation import pipeline as dp
from taxis.pipelines.data_science import pipeline as ds


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_preparation_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    return {"__default__": data_preparation_pipeline + data_science_pipeline,
            "dp": data_preparation_pipeline,
            "ds": data_science_pipeline}

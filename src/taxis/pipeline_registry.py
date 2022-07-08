"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from taxis.pipelines.data_preparation import pipeline as dp
from taxis.pipelines.data_science import pipeline as ds
from taxis.pipelines.deploy_model import pipeline as dm


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_preparation_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()
    deploy_model_pipeline = dm.create_pipeline()
    return {
        "__default__": data_preparation_pipeline + data_science_pipeline,
        "dp": data_preparation_pipeline,
        "ds": data_science_pipeline,
        "dm": deploy_model_pipeline
    }

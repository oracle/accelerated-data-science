import json
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from functools import lru_cache

model_name = "model.json"


"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""


@lru_cache(maxsize=10)
def load_model(model_file_name=model_name):
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    model_xgb = xgb.Booster()
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    if model_file_name in contents:
        model_xgb.load_model(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file_name)
        )
        return model_xgb
    else:
        raise Exception(
            "{0} is not found in model directory {1}".format(model_file_name, model_dir)
        )


def deserialize(data):
    """
    Deserialize json serialization data to data in original type when sent to predict.

    Parameters
    ----------
    data: serialized input data.

    Returns
    -------
    data: deserialized input data.

    """
    json_data = data["data"]
    data_type = data["data_type"]

    if "numpy.ndarray" in data_type:
        return np.array(json_data)
    if "pandas.core.series.Series" in data_type:
        return pd.Series(json_data)
    if "pandas.core.frame.DataFrame" in data_type:
        return pd.read_json(json_data)

    return json_data


def pre_inference(data):
    """
    Preprocess data

    Parameters
    ----------
    data: Data format as expected by the predict API of the core estimator.

    Returns
    -------
    data: Data format after any processing.

    """
    data = deserialize(data)
    return data


def post_inference(yhat):
    """
    Post-process the model results

    Parameters
    ----------
    yhat: Data format after calling model.predict.

    Returns
    -------
    yhat: Data format after any processing.

    """
    return yhat.tolist()


def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case xgboost models it could be dict/string of Numpy Array/Pandas DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction': output from model.predict method}

    """
    data = pre_inference(data)
    data = xgb.DMatrix(data)
    return {"prediction": post_inference(model.predict(data))}

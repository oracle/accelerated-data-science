#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import sys
from functools import lru_cache
from cloudpickle import cloudpickle

model_name = "model.pkl"


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
    model_dir = os.path.dirname(os.path.realpath(__file__))
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    contents = os.listdir(model_dir)
    if model_file_name in contents:
        with open(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file_name),
            "rb",
        ) as file:
            return cloudpickle.load(file)
    else:
        raise Exception(
            "{0} is not found in model directory {1}".format(model_file_name, model_dir)
        )


def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data has to be json serialiable, hence, accepted data type includes
        only Dict and Str. You need to process the data in order to transform
        the data from Dict or str format to the type that the model accepts.

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction': output from model.predict method}

    """
    from pandas import read_json, DataFrame
    from io import StringIO
    import pandas as pd

    y_update = (
        read_json(StringIO(data["y_update"]))["infl"]
        if isinstance(data["y_update"], str)
        else DataFrame.from_dict(data["y_update"])["infl"]
    )
    y_update.index = pd.period_range(data["start_period"], data["end_period"], freq="Q")
    res_post = model.append(y_update)
    yhat = pd.concat([y_update, res_post.forecast(data["x_test"])]).to_list()

    return {"prediction": yhat}

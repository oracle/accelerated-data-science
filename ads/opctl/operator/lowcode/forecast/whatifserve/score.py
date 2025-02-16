#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from joblib import load
import pandas as pd
import numpy as np
from functools import lru_cache
import logging
import ads
from ads.opctl.operator.lowcode.common.utils import load_data
from ads.opctl.operator.common.operator_config import InputData
from ads.opctl.operator.lowcode.forecast.const import SupportedModels

ads.set_auth("resource_principal")

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger_pred = logging.getLogger('model-prediction')
logger_pred.setLevel(logging.INFO)
logger_feat = logging.getLogger('input-features')
logger_feat.setLevel(logging.INFO)

"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""


@lru_cache(maxsize=10)
def load_model():
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    model_file_name = "models.pickle"
    if model_file_name in contents:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file_name), "rb") as file:
            model = load(file)
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_file_name, model_dir))
    return model


@lru_cache(maxsize=1)
def fetch_data_type_from_schema(
        input_schema_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_schema.json")):
    """
    Returns data type information fetch from input_schema.json.

    Parameters
    ----------
    input_schema_path: path of input schema.

    Returns
    -------
    data_type: data type fetch from input_schema.json.

    """
    data_type = {}
    if os.path.exists(input_schema_path):
        schema = json.load(open(input_schema_path))
        for col in schema['schema']:
            data_type[col['name']] = col['dtype']
    else:
        print(
            "input_schema has to be passed in in order to recover the same data type. pass `X_sample` in `ads.model.framework.sklearn_model.SklearnModel.prepare` function to generate the input_schema. Otherwise, the data type might be changed after serialization/deserialization.")
    return data_type


def deserialize(data, input_schema_path):
    """
    Deserialize json serialization data to data in original type when sent to predict.

    Parameters
    ----------
    data: serialized input data.
    input_schema_path: path of input schema.

    Returns
    -------
    data: deserialized input data.

    """

    # Add further data deserialization if needed
    return data


def pre_inference(data, input_schema_path):
    """
    Preprocess data

    Parameters
    ----------
    data: Data format as expected by the predict API of the core estimator.
    input_schema_path: path of input schema.

    Returns
    -------
    data: Data format after any processing.

    """
    return deserialize(data, input_schema_path)


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
    if isinstance(yhat, pd.core.frame.DataFrame):
        yhat = yhat.values
    if isinstance(yhat, np.ndarray):
        yhat = yhat.tolist()
    return yhat


def get_forecast(future_df, model_name, series_id, model_object, date_col, target_column, target_cat_col, horizon):
    date_col_name = date_col["name"]
    date_col_format = date_col["format"]
    future_df[target_cat_col] = future_df[target_cat_col].astype("str")
    future_df[date_col_name] = pd.to_datetime(
        future_df[date_col_name], format=date_col_format
    )
    if model_name == SupportedModels.AutoTS:
        series_id_col = "Series"
        full_data_indexed = future_df.rename(columns={target_cat_col: series_id_col})
        additional_regressors = list(
            set(full_data_indexed.columns) - {target_column, series_id_col, date_col_name}
        )
        future_reg = full_data_indexed.reset_index().pivot(
            index=date_col_name,
            columns=series_id_col,
            values=additional_regressors,
        )
        pred_obj = model_object.predict(future_regressor=future_reg)
        return pred_obj.forecast[series_id].tolist()
    elif model_name == SupportedModels.Prophet and series_id in model_object:
        model = model_object[series_id]['model']
        label_encoder = model_object[series_id]['le']
        processed = future_df.rename(columns={date_col_name: 'ds', target_column: 'y'})
        encoded_df = label_encoder.transform(processed)
        forecast = model.predict(encoded_df)
        return forecast['yhat'].tolist()
    elif model_name == SupportedModels.NeuralProphet and series_id in model_object:
        model = model_object[series_id]['model']
        label_encoder = model_object[series_id]['le']
        model.restore_trainer()
        accepted_regressors = list(model.config_regressors.regressors.keys())
        data = future_df.rename(columns={date_col_name: 'ds', target_column: 'y'})
        encoded_df = label_encoder.transform(data)
        future = encoded_df[accepted_regressors + ["ds"]].reset_index(drop=True)
        future["y"] = None
        forecast = model.predict(future)
        return forecast['yhat1'].tolist()
    elif model_name == SupportedModels.Arima and series_id in model_object:
        model = model_object[series_id]['model']
        label_encoder = model_object[series_id]['le']
        predict_cols = model_object[series_id]["predict_component_cols"]
        encoded_df = label_encoder.transform(future_df)
        x_pred = encoded_df.set_index(date_col_name)
        x_pred = x_pred.drop(target_cat_col, axis=1)
        yhat, conf_int = model.predict(
            n_periods=horizon,
            X=x_pred[predict_cols],
            return_conf_int=True
        )
        yhat_clean = pd.DataFrame(yhat, index=yhat.index, columns=["yhat"])
        return yhat_clean['yhat'].tolist()
    elif model_name == SupportedModels.AutoMLX and series_id in model_object:
        model = model_object[series_id]['model']
        label_encoder = model_object[series_id]['le']
        encoded_df = label_encoder.transform(future_df)
        x_pred = encoded_df.drop(target_cat_col, axis=1)
        x_pred = x_pred.set_index(date_col_name)
        forecast = model.forecast(
            X=x_pred,
            periods=horizon
        )
        return forecast[target_column].tolist()
    else:
        raise Exception(f"Invalid model object type: {type(model_object).__name__}.")


def predict(data, model=load_model()) -> dict:
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Panda DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: { 'prediction': output from `model.predict` method }

    """
    assert model is not None, "Model is not loaded"

    models = model["models"]
    specs = model["spec"]
    horizon = specs["horizon"]
    model_name = specs["model"]
    date_col = specs["datetime_column"]
    target_column = specs["target_column"]
    target_category_column = specs["target_category_columns"][0]

    try:
        input_data = InputData(**data["additional_data"])
    except TypeError as e:
        raise ValueError(f"Validation error: {e}")
    additional_data = load_data(input_data)

    unique_values = additional_data[target_category_column].unique()
    forecasts = {}
    for key in unique_values:
        try:
            s_id = str(key)
            filtered = additional_data[additional_data[target_category_column] == key]
            future = filtered.tail(horizon)
            forecast = get_forecast(future, model_name, s_id, models, date_col,
                                    target_column, target_category_column, horizon)
            forecasts[s_id] = json.dumps(forecast)
        except Exception as e:
            raise RuntimeError(
                f"An error occurred during prediction: {e}."
                f" Please ensure the input data matches the format and structure of the data used during training.")

    return {'prediction': json.dumps(forecasts)}

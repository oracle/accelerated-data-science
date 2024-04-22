#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from ads.model.framework.lightgbm_model import LightGBMModel
from ads.model.framework.xgboost_model import XGBoostModel
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tests.ads_unit_tests.utils import get_test_dataset_path
from tests.integration.config import secrets


def regression_data():
    df = pd.read_csv(get_test_dataset_path("vor_flights5k.csv"), index_col=0)
    df = df[~df["DepDelay"].isnull()]
    X_reg = df.drop(
        ["DepDelay", "UniqueCarrier", "TailNum", "Origin", "Dest", "CancellationCode"],
        axis=1,
    )
    y_reg = df["DepDelay"]
    (
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    ) = train_test_split(X_reg, y_reg, test_size=0.25)
    X_dict_reg = X_test_reg[:100].to_dict()
    for col in X_dict_reg.keys():
        for row in X_dict_reg[col].keys():
            if np.isnan(X_dict_reg[col][row]):
                X_dict_reg[col][row] = None
    return (X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_dict_reg)


regression_dataset = regression_data()


def classification_data():
    # lightgbm model with sklearn api, classification
    data = pd.read_csv(get_test_dataset_path("vor_iris.csv"))
    dataset = data.values
    # split data into X and y
    X_clas = dataset[:, 0:4]
    Y_clas = dataset[:, 4]
    # encode string class values as integers
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y_clas)
    label_encoded_y = label_encoder.transform(Y_clas)
    seed = 7
    test_size = 0.33
    (
        X_train_clas,
        X_test_clas,
        y_train_clas,
        y_test_clas,
    ) = model_selection.train_test_split(
        X_clas, label_encoded_y, test_size=test_size, random_state=seed
    )
    return (X_train_clas, X_test_clas, y_train_clas, y_test_clas)


classification_dataset = classification_data()


def lightgbm_learning_api():
    (X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_dict_reg) = regression_dataset
    train = lgb.Dataset(X_train_reg, label=y_train_reg)
    test = lgb.Dataset(X_test_reg, label=y_test_reg)
    num_round = 10
    param = {"num_leaves": 31, "objective": "binary"}
    param["metric"] = "auc"
    model_lgb = lgb.train(param, train, num_round)
    lgb_origin = model_lgb.predict(X_test_reg[:100])
    return {
        "framework": LightGBMModel,
        "estimator": model_lgb,
        "artifact_dir": "./artifact_folder/lightgbm_learning_api",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_reg[:100],
        "y_true": y_test_reg[:100],
        "onnx_data": X_dict_reg,
        "local_pred": lgb_origin,
        "score_py_path": None,
    }


def lightgbm_sklearn_api_regression():
    (X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_dict_reg) = regression_dataset
    model_sklearn_reg = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
    )
    model_sklearn_reg.fit(
        X_train_reg,
        y_train_reg,
        eval_set=[
            (X_train_reg, y_train_reg),
            (X_test_reg, y_test_reg),
        ],
        early_stopping_rounds=20,
    )
    sklearn_reg_origin = model_sklearn_reg.predict(X_test_reg[:100])
    return {
        "framework": LightGBMModel,
        "estimator": model_sklearn_reg,
        "artifact_dir": "./artifact_folder/lightgbm_sklearn_api_regression",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_reg[:100],
        "y_true": y_test_reg[:100],
        "onnx_data": X_dict_reg,
        "local_pred": sklearn_reg_origin,
        "score_py_path": None,
    }


def lightgbm_sklearn_api_classification():
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = classification_dataset
    # fit model to training data
    model_sklearn_clas = lgb.LGBMClassifier()
    model_sklearn_clas.fit(X_train_clas, y_train_clas)
    sklearn_clas_origin = model_sklearn_clas.predict(X_test_clas)
    return {
        "framework": LightGBMModel,
        "estimator": model_sklearn_clas,
        "artifact_dir": "./artifact_folder/lightgbm_sklearn_api_classification",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_clas,
        "y_true": y_test_clas,
        "onnx_data": X_test_clas.tolist(),
        "local_pred": sklearn_clas_origin,
        "score_py_path": None,
    }


def xgboost_learning_api():
    (X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_dict_reg) = regression_dataset
    train_reg = xgb.DMatrix(X_train_reg.values, y_train_reg.values)
    test_reg = xgb.DMatrix(X_test_reg.values, y_test_reg.values)
    params = {"learning_rate": 0.01, "max_depth": 3}
    model_xgb = xgb.train(
        params,
        train_reg,
        evals=[
            (train_reg, "train"),
            (test_reg, "validation"),
        ],
        num_boost_round=100,
        early_stopping_rounds=20,
    )
    model_xgb_origin = model_xgb.predict(test_reg)[:100]

    return {
        "framework": XGBoostModel,
        "estimator": model_xgb,
        "artifact_dir": "./artifact_folder/xgboost_learning_api",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env2/1.0/ads_env2v1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_reg[:100],
        "y_true": y_test_reg[:100],
        "onnx_data": X_dict_reg,
        "local_pred": model_xgb_origin,
        "score_py_path": None,
    }


def xgboost_sklearn_api_regression():
    (X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_dict_reg) = regression_dataset
    model_sklearn_reg = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
    )
    model_sklearn_reg.fit(
        X_train_reg.values,
        y_train_reg.values,
        eval_set=[
            (X_train_reg.values, y_train_reg.values),
            (X_test_reg.values, y_test_reg.values),
        ],
        early_stopping_rounds=20,
    )
    sklearn_reg_origin = model_sklearn_reg.predict(X_test_reg[:100])

    return {
        "framework": XGBoostModel,
        "estimator": model_sklearn_reg,
        "artifact_dir": "./artifact_folder/xgboost_sklearn_api_regression",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env2/1.0/ads_env2v1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_reg[:100],
        "y_true": y_test_reg[:100],
        "onnx_data": X_dict_reg,
        "local_pred": sklearn_reg_origin,
        "score_py_path": None,
    }


def xgboost_sklearn_api_classification():
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = classification_dataset
    # fit model to training data
    model_sklearn_clas = xgb.XGBClassifier()
    model_sklearn_clas.fit(X_train_clas, y_train_clas)
    sklearn_clas_origin = model_sklearn_clas.predict(X_test_clas)
    return {
        "framework": XGBoostModel,
        "estimator": model_sklearn_clas,
        "artifact_dir": "./artifact_folder/xgboost_sklearn_api_classification",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env2/1.0/ads_env2v1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_clas,
        "y_true": y_test_clas,
        "onnx_data": X_test_clas.tolist(),
        "local_pred": sklearn_clas_origin,
        "score_py_path": None,
        "prepare_args": {"use_case_type": "multinomial_classification"},
    }

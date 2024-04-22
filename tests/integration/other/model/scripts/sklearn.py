#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.model.framework.sklearn_model import SklearnModel
from tests.ads_unit_tests.utils import get_test_dataset_path
from tests.integration.config import secrets
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def flights_data():
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
    imp_mean_X = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean_X.fit(X_reg)
    X_train_imp = imp_mean_X.transform(X_train_reg)
    X_test_imp = imp_mean_X.transform(X_test_reg)
    return (X_train_imp, X_test_imp, y_train_reg, y_test_reg)


flights_dataset = flights_data()


def bank_data():
    df_clas = pd.read_csv(get_test_dataset_path("vor_bank.csv"))
    y_clas = df_clas["y"]
    X_clas = df_clas.drop(columns=["y"])
    for i, col in X_clas.items():
        col.replace("unknown", "", inplace=True)
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = train_test_split(
        X_clas, y_clas, test_size=0.1, random_state=42
    )
    le = LabelEncoder()
    y_train_clas = le.fit_transform(y_train_clas)
    y_test_clas = le.transform(y_test_clas)
    return (X_train_clas, X_test_clas, y_train_clas, y_test_clas)


bank_dataset = bank_data()


def sklearn_no_pipeline():
    X_train_imp, X_test_imp, y_train_reg, y_test_reg = flights_dataset
    model_reg = LogisticRegression()
    model_reg.fit(X_train_imp, y_train_reg)
    origin = model_reg.predict(X_test_imp[:100])
    return {
        "framework": SklearnModel,
        "estimator": model_reg,
        "artifact_dir": "./artifact_folder/sklearn_no_pipeline",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_imp[:100],
        "y_true": y_test_reg[:100],
        "onnx_data": X_test_imp[:100],
        "local_pred": origin,
        "score_py_path": None,
    }


def sklearn_pipeline_with_sklearn_model():
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = bank_dataset
    categorical_cols = []
    numerical_cols = []
    for i, col in X_train_clas.items():
        if col.dtypes == "object":
            categorical_cols.append(col.name)
        else:
            numerical_cols.append(col.name)
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent", fill_value="nan", missing_values=""
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model_sklearn = RandomForestClassifier(n_estimators=100, random_state=0)
    model_sklearn_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", model_sklearn)]
    )
    model_sklearn_pipeline.fit(X_train_clas, y_train_clas)
    sklearn_pipeline_origin = model_sklearn_pipeline.predict(X_test_clas)
    return {
        "framework": SklearnModel,
        "estimator": model_sklearn_pipeline,
        "artifact_dir": "./artifact_folder/sklearn_pipeline_with_sklearn_model",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_clas[:100],
        "y_true": y_test_clas[:100],
        "onnx_data": X_test_clas[:100],
        "local_pred": sklearn_pipeline_origin[:100],
        "score_py_path": None,
    }


def sklearn_pipeline_with_xgboost_model():
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = bank_dataset
    categorical_cols = []
    numerical_cols = []
    for i, col in X_train_clas.items():
        if col.dtypes == "object":
            categorical_cols.append(col.name)
        else:
            numerical_cols.append(col.name)
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent", fill_value="nan", missing_values=""
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model_sklearn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(n_estimators=100, random_state=0)),
        ]
    )
    model_sklearn_pipeline.fit(X_train_clas, y_train_clas)
    sklearn_pipeline_origin = model_sklearn_pipeline.predict(X_test_clas)
    return {
        "framework": SklearnModel,
        "estimator": model_sklearn_pipeline,
        "artifact_dir": "./artifact_folder/sklearn_pipeline_with_xgboost_model",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env2/1.0/ads_env2v1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_clas[:100],
        "y_true": y_test_clas[:100],
        "onnx_data": X_test_clas[:100],
        "local_pred": sklearn_pipeline_origin[:100],
        "score_py_path": None,
    }


def sklearn_pipeline_with_lightgbm_model():
    (X_train_clas, X_test_clas, y_train_clas, y_test_clas) = bank_dataset
    categorical_cols = []
    numerical_cols = []
    for i, col in X_train_clas.items():
        if col.dtypes == "object":
            categorical_cols.append(col.name)
        else:
            numerical_cols.append(col.name)
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent", fill_value="nan", missing_values=""
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model_sklearn_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(n_estimators=100, random_state=0)),
        ]
    )
    model_sklearn_pipeline.fit(X_train_clas, y_train_clas)
    sklearn_pipeline_origin = model_sklearn_pipeline.predict(X_test_clas)
    return {
        "framework": SklearnModel,
        "estimator": model_sklearn_pipeline,
        "artifact_dir": "./artifact_folder/sklearn_pipeline_with_lightgbm_model",
        "inference_conda_env": f"oci://{secrets.other.BUCKET_1}@{secrets.common.NAMESPACE}/published_conda_environments/cpu/ads_env/1.0/ads_envv1_0",
        "inference_python_version": "3.7",
        "model_file_name": None,
        "data": X_test_clas[:100],
        "y_true": y_test_clas[:100],
        "onnx_data": X_test_clas[:100],
        "local_pred": sklearn_pipeline_origin[:100],
        "score_py_path": None,
    }

#!/usr/bin/env python

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings
import logging
import pytest
from ads.common.data import ADSData
import numpy as np
import shutil
import pandas as pd
import os
from ads.model.transformer.onnx_transformer import ONNXTransformer

warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.ERROR)

tmp_model_dir = "/tmp/model"


@pytest.fixture
def df():
    # TODO: This will still fail for timestamps
    sample_classification_data = {
        "A": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 0},
        "B": {
            0: 0.1111111111111111,
            1: 0.2222222222222222,
            2: 0.0,
            3: 0.3333333333333333,
            4: 0.1111111111111111,
            5: 0.1111111111111111,
        },
        "C": {0: "cat1", 1: "cat1", 2: "cat1", 3: "cat1", 4: "cat1", 5: ""},
        "D": {0: "dog1", 1: "", 2: "dog1", 3: "dog2", 4: "dog3", 5: "dog1"},
        "E": {0: True, 1: False, 2: True, 3: False, 4: True, 5: False},
        "F": {0: 4000, 1: 5000, 2: 6000, 3: 7000, 4: 8000, 5: 9000},
        "target": {0: "yes", 1: "yes", 2: "no", 3: "maybe", 4: "maybe", 5: "no"},
    }
    # dtype_dict = {'A': np.int8, 'B': np.float64, 'C': str, 'D': str, 'E': bool, 'F': 'datetime64[ns]', 'target': str}
    dtype_dict = {
        "A": np.int8,
        "B": np.float64,
        "C": str,
        "D": str,
        "E": bool,
        "F": int,
        "target": str,
    }
    return pd.DataFrame(sample_classification_data).astype(dtype_dict)


@pytest.fixture
def train(df):
    return ADSData(X=df.drop(columns="target")[:4], y=df.target[:4])


@pytest.fixture
def test(df):
    return ADSData(X=df.drop(columns="target")[4:], y=df.target[4:])


def setup_module():
    os.makedirs(tmp_model_dir)


# Do a bunch of stuff to utilize all tools, NaNs, label encoding, adding a new category to test set, new category
# to y column, dtype trans -> float32, experiment with other dtypes
def test_serialization(train, test):
    data_transformer = ONNXTransformer()
    X_t = data_transformer.fit_transform(train.X, {"C": "cat2", "D": "dog1"})
    assert data_transformer._fitted, "Transformer should be fit"
    X_t_test = data_transformer.transform(test.X)
    assert any(X_t_test.C != test.X.C), "The transformer should have imputed the nan"
    assert any(
        X_t_test.C == test.X.C
    ), "The transformer should not have changed the first row of the test set"
    assert all(
        X_t_test.A == test.X.A
    ), "The transformer should not have changed the A column at all"
    assert all(
        X_t_test.B != test.X.B
    ), "The transformer should have cut off both of these values (from float64-32)"
    assert (
        test.X.D.iloc[0] == X_t_test.D.iloc[0]
    ), "The transformer should not impute the unknown category"

    data_transformer.save(os.path.join(tmp_model_dir, "data_transformer_sample.json"))
    data_transformer2 = ONNXTransformer.load(
        os.path.join(tmp_model_dir, "data_transformer_sample.json")
    )
    X_t_test2 = data_transformer2.transform(test.X)
    assert all(
        X_t_test == X_t_test2
    ), "The reloaded transformer on test.X should be the same as the original"


def test_serialization_not_hanlding_missing_value(df):
    data_transformer = ONNXTransformer()
    train2 = ADSData(X=df.drop(columns="target")[:4], y=np.arange(4))
    test2 = ADSData(X=df.drop(columns="target")[4:], y=np.arange(2))
    X_t = data_transformer.fit_transform(train2.X)
    assert data_transformer._fitted, "Transformer should be fit"
    X_t_test = data_transformer.transform(test2.X)

    assert all(
        X_t_test.C == test2.X.C
    ), "The transformer should not have changed the first row of the test set"
    assert all(
        X_t_test.A == test2.X.A
    ), "The transformer should not have changed the A column at all"
    assert any(
        X_t_test.B != test2.X.B
    ), "The transformer should have cut off both of these values (from float64-32)"
    assert (
        test2.X.D.iloc[0] == X_t_test.D.iloc[0]
    ), "The transformer should've imputed the unknown category"

    data_transformer.save(os.path.join(tmp_model_dir, "data_transformer_sample.json"))
    data_transformer2 = ONNXTransformer.load(
        os.path.join(tmp_model_dir, "data_transformer_sample.json")
    )
    X_t_test2 = data_transformer2.transform(test2.X)
    assert all(
        X_t_test == X_t_test2
    ), "The reloaded transformer on test.X should be the same as the original"


def test__handle_dtypes_numpy_array():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype="int")
    array_new = ONNXTransformer._handle_dtypes(array)
    assert "float32" in str(array_new.dtype), "converted from int to float32"
    assert isinstance(array_new, np.ndarray)
    assert array_new.shape == array.shape


def test__handle_dtypes_list():
    l = [[1, 2, 3], [4, 5, 6]]
    l_new = ONNXTransformer._handle_dtypes(l)
    assert "int" in str(np.array(l_new).dtype), "converted from int to float32"
    assert isinstance(l_new, list)


def test__handle_dtypes_dataframe(df):
    df_new = ONNXTransformer._handle_dtypes(df)
    assert "float32" in str(df_new.dtypes.A), "converted from int to float32"
    assert "float32" in str(df_new.dtypes.B), "converted from float64 to float32"
    assert "float32" in str(df_new.dtypes.F), "converted from float64 to float32"
    assert "object" in str(df_new.dtypes.C), "didnt change"
    assert "object" in str(df_new.dtypes.D), "didnt change"
    assert "float32" in str(df_new.dtypes.E), "didnt change"
    assert "object" in str(df_new.dtypes.target), "didnt change"


def test__handle_dtypes_series(df):
    df_new_A = ONNXTransformer._handle_dtypes(df["A"])
    assert "float32" in str(df_new_A.dtypes), "converted from int to float32"
    assert isinstance(df_new_A, pd.Series)


def test__handle_missing_value_dataframe(df):
    df_new = ONNXTransformer._handle_missing_value_dataframe(df, {"C": "cat2"})
    assert df_new["C"][5] == "cat2"
    assert df_new["D"][1] == ""
    df_new = ONNXTransformer._handle_missing_value_dataframe(df, {"D": "dog4"})
    assert df_new["D"][1] == "dog4"


def test__handle_missing_value_np_array(df):
    array_new = ONNXTransformer._handle_missing_value(df.values, {2: "cat2"})
    assert array_new[5, 2] == "cat2"
    assert array_new[1, 3] == ""
    assert isinstance(array_new, np.ndarray)
    assert array_new.shape == df.values.shape
    array_new1 = ONNXTransformer._handle_missing_value(df.values, {3: "dog4"})
    assert array_new1[1, 3] == "dog4"


def test__handle_missing_value_list(df):
    l_new = ONNXTransformer._handle_missing_value(df.values.tolist(), {2: "cat2"})
    assert np.array(l_new)[5, 2] == "cat2"
    assert np.array(l_new)[1, 3] == ""
    assert isinstance(l_new, list)
    array_new1 = ONNXTransformer._handle_missing_value(df.values, {3: "dog4"})
    assert array_new1[1, 3] == "dog4"


def test__handle_missing_value_series(df):
    df_new_C = ONNXTransformer._handle_missing_value(df["C"], {"C": "cat2"})
    assert df_new_C[5] == "cat2"


def teardown_module():
    shutil.rmtree(tmp_model_dir)

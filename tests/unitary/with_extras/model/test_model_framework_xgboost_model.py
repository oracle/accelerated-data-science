#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - XGBoostModel
"""
import base64
import os
import shutil
from io import BytesIO

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
import pytest, mock, sys
import xgboost as xgb
from ads.model.framework.xgboost_model import XGBoostModel
from ads.model.serde.model_serializer import XgboostOnnxModelSerializer
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

tmp_model_dir = "/tmp/model"


class TestXGBoostModel:
    """Unittests for the XGBoostModel class."""

    def setup_method(cls):
        os.makedirs(tmp_model_dir, exist_ok=True)

    def setup_class(cls):

        X, y = make_classification(n_samples=100, n_informative=5, n_classes=2)
        (
            X_train_classification,
            cls.X_test_classification,
            y_train_classification,
            cls.y_test_classification,
        ) = train_test_split(X, y, test_size=0.25)

        X, y = make_regression(n_samples=150, n_features=1, noise=0.2)
        (
            X_train_regression,
            cls.X_test_regression,
            y_train_regression,
            cls.y_test_regression,
        ) = train_test_split(X, y, test_size=0.25)

        # xgboost api
        cls.train_classification = xgb.DMatrix(
            X_train_classification, y_train_classification
        )
        cls.test_classification = xgb.DMatrix(
            cls.X_test_classification, cls.y_test_classification
        )
        params = {"learning_rate": 0.01, "max_depth": 3}
        cls.model_xgb_classification = xgb.train(
            params,
            cls.train_classification,
            evals=[
                (cls.train_classification, "train"),
                (cls.test_classification, "validation"),
            ],
            num_boost_round=100,
            early_stopping_rounds=20,
        )
        cls.xgboost_classification = XGBoostModel(
            cls.model_xgb_classification, tmp_model_dir
        )

        cls.train_regression = xgb.DMatrix(X_train_regression, y_train_regression)
        cls.test_regression = xgb.DMatrix(cls.X_test_regression, cls.y_test_regression)
        params = {"learning_rate": 0.01, "max_depth": 3}
        cls.model_xgb_regression = xgb.train(
            params,
            cls.train_regression,
            evals=[
                (cls.train_regression, "train"),
                (cls.test_regression, "validation"),
            ],
            num_boost_round=100,
            early_stopping_rounds=20,
        )
        cls.xgboost_regression = XGBoostModel(cls.model_xgb_regression, tmp_model_dir)

        # sklearn api
        cls.model_sklearn = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.01, random_state=42
        )
        cls.model_sklearn.fit(
            X_train_classification,
            y_train_classification,
            eval_set=[
                (X_train_classification, y_train_classification),
                (cls.X_test_classification, cls.y_test_classification),
            ],
            early_stopping_rounds=20,
        )
        cls.sklearn = XGBoostModel(cls.model_sklearn, tmp_model_dir)

    def test_serialize_and_load_model_as_json_xgboost_api_classification(self):
        """
        Test serialize and load model using Xgboost API with classification.
        """
        target_path = os.path.join(tmp_model_dir, "test_xgboost.json")
        self.xgboost_classification.model_file_name = "test_xgboost.json"
        self.xgboost_classification.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)
        loaded_model = xgb.Booster()
        loaded_model.load_model(target_path)
        pred_xgboost = self.model_xgb_classification.predict(self.test_classification)
        pred_json = loaded_model.predict(self.test_classification)
        assert all(pred_xgboost == pred_json)

    def test_serialize_and_load_model_as_json_xgboost_api_regression(self):
        """
        Test serialize and load model using Xgboost API with regression.
        """
        target_path = os.path.join(tmp_model_dir, "test_xgboost_reg.json")
        self.xgboost_regression.model_file_name = "test_xgboost_reg.json"
        self.xgboost_regression.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)
        loaded_model = xgb.Booster()
        loaded_model.load_model(target_path)
        pred_xgboost = self.model_xgb_regression.predict(self.test_regression)
        pred_json = loaded_model.predict(self.test_regression)
        assert all(pred_xgboost == pred_json)

    def test_serialize_and_load_model_as_onnx_xgboost_api(self):
        """
        Test serialize and load model to ONNX format using Xgboost API.
        """
        target_path = os.path.join(tmp_model_dir, "test_xgboost.onnx")
        self.xgboost_classification.model_file_name = "test_xgboost.onnx"
        self.xgboost_classification.serialize_model(
            as_onnx=True, X_sample=self.X_test_classification
        )
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run(
            [label_name], {input_name: self.X_test_classification.astype(np.float32)}
        )[0]
        pred_xgboost = self.model_xgb_classification.predict(self.test_classification)
        pred_onx = pred_onx.ravel()
        pred_xgboost = pred_xgboost.ravel()
        d = np.abs(pred_onx - pred_xgboost)
        assert d.max() <= 1

    def test_serialize_and_load_model_as_json_sklearn_api(self):
        """
        Test serialize and load model using Sklearn API.
        """
        target_path = os.path.join(tmp_model_dir, "test_sklearn.json")
        self.sklearn.model_file_name = "test_sklearn.json"
        self.sklearn.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(target_path)
        pred_sklearn = self.model_sklearn.predict(self.X_test_classification)
        pred_json = loaded_model.predict(self.X_test_classification)
        assert all(pred_sklearn == pred_json)

    def test_serialize_and_load_model_as_onnx_sklearn_api(self):
        """
        Test serialize and load model using Sklearn API.
        """
        target_path = os.path.join(tmp_model_dir, "test_sklearn.onnx")
        self.sklearn.model_file_name = "test_sklearn.onnx"
        self.sklearn.serialize_model(as_onnx=True)
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run(
            [label_name], {input_name: self.X_test_classification.astype(np.float32)}
        )[0]
        pred_xgboost = self.model_sklearn.predict(self.X_test_classification)
        pred_onx = pred_onx.ravel()
        pred_xgboost = pred_xgboost.ravel()
        d = np.abs(pred_onx - pred_xgboost)
        assert d.max() <= 1

    def test_serialize_with_model_file_name(self):
        """
        Test correct and wrong model_file_name format.
        """
        with pytest.raises(
            AssertionError,
            match="Wrong file extension. Expecting `.onnx` suffix.",
        ):
            self.sklearn.model_file_name = self.sklearn._handle_model_file_name(
                as_onnx=True, model_file_name="test.abc"
            )

        self.sklearn.model_file_name = "test.json"
        self.sklearn.serialize_model(as_onnx=False)
        assert os.path.exists(os.path.join(tmp_model_dir, "test.json"))

    def test_serialize_without_model_file_name(self):
        """
        Test no model_file_name.
        """
        self.sklearn.model_file_name = None
        assert (
            self.sklearn._handle_model_file_name(
                as_onnx=False, model_file_name="model.json"
            )
            == "model.json"
        )

    @pytest.mark.parametrize(
        "test_data",
        [pd.Series([1, 2, 3]), [1, 2, 3]],
    )
    def test_get_data_serializer_helper_with_convert_to_list(self, test_data):
        serialized_data = self.xgboost_classification.get_data_serializer().serialize(
            test_data
        )
        assert serialized_data["data"] == [1, 2, 3]
        assert serialized_data["data_type"] == str(type(test_data))

    def test_get_data_serializer_helper_numpy(self):
        test_data = np.array([1, 2, 3])
        serialized_data = self.xgboost_classification.get_data_serializer().serialize(
            test_data
        )
        load_bytes = BytesIO(base64.b64decode(serialized_data["data"].encode("utf-8")))
        deserialized_data = np.load(load_bytes, allow_pickle=True)
        assert (deserialized_data == test_data).any()

    @pytest.mark.parametrize(
        "test_data",
        [
            pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}),
        ],
    )
    def test_get_data_serializer_helper_with_pandasdf(self, test_data):
        serialized_data = self.xgboost_classification.get_data_serializer().serialize(
            test_data
        )
        assert (
            serialized_data["data"]
            == '{"a":{"0":1,"1":2},"b":{"0":2,"1":3},"c":{"0":3,"1":4}}'
        )
        assert serialized_data["data_type"] == "<class 'pandas.core.frame.DataFrame'>"

    @pytest.mark.parametrize(
        "test_data",
        ["I have an apple", {"a": [1], "b": [2], "c": [3]}],
    )
    def test_get_data_serializer_helper_with_no_change(self, test_data):
        serialized_data = self.xgboost_classification.get_data_serializer().serialize(
            test_data
        )
        assert serialized_data["data"] == test_data

    def test_get_data_serializer_helper_raise_error(self):
        class TestData:
            pass

        test_data = TestData()
        with pytest.raises(TypeError):
            serialized_data = (
                self.xgboost_classification.get_data_serializer().serialize(test_data)
            )

    def test_X_sample_related_for_to_onnx(self):
        """
        Test if X_sample works in to_onnx propertly.
        """
        wrong_format = [1, 2, 3, 4]
        onnx_serializer = XgboostOnnxModelSerializer()
        onnx_serializer.estimator = None
        with pytest.raises(
            ValueError,
            match="`initial_types` can not be detected. Please directly pass initial_types.",
        ):
            onnx_serializer._to_onnx(X_sample=wrong_format)

    def test_xgboost_to_onnx_with_xgboost_uninstalled(self):
        """
        negative test for xgboost
        """
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            with pytest.raises(ModuleNotFoundError):
                XGBoostModel(self.model_xgb_regression, tmp_model_dir)

    def teardown_class(cls):
        shutil.rmtree(tmp_model_dir, ignore_errors=True)

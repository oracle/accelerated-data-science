#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - LightGBMModel
"""
import base64
import os
import shutil
from io import BytesIO

import joblib
import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
import pytest, mock, sys
import tempfile
from ads.model.framework.lightgbm_model import LightGBMModel
from ads.model.serde.model_serializer import LightGBMOnnxModelSerializer
from sklearn.datasets import load_iris, make_regression

tmp_model_dir = tempfile.mkdtemp()


class TestLightGBMModel:
    """Unittests for the LightGBMModel class."""

    def setup_class(cls):
        os.makedirs(tmp_model_dir, exist_ok=True)

        # lightgbm.basic.Booster
        cls.data = np.random.rand(500, 10)
        label = np.random.randint(2, size=500)
        train_data = lgb.Dataset(cls.data, label=label)
        param = {"num_leaves": 31, "objective": "binary"}
        param["metric"] = "auc"
        num_round = 10
        cls.bst = lgb.train(param, train_data, num_round)
        cls.Booster_model = LightGBMModel(cls.bst, tmp_model_dir)

        # LGBMClassifier
        cls.X_LGBMClassifier, y_LGBMClassifier = load_iris(return_X_y=True)
        cls.LGBMClassifier = lgb.LGBMClassifier()
        cls.LGBMClassifier.fit(cls.X_LGBMClassifier, y_LGBMClassifier)
        cls.LGBMClassifier_model = LightGBMModel(cls.LGBMClassifier, tmp_model_dir)

        # LGBMRegressor
        cls.X_LGBMRegressor, y_LGBMRegressor = make_regression(
            n_samples=150, n_features=1, noise=0.2
        )
        cls.LGBMRegressor = lgb.LGBMRegressor()
        cls.LGBMRegressor.fit(cls.X_LGBMRegressor, y_LGBMRegressor)
        cls.LGBMRegressor_model = LightGBMModel(cls.LGBMRegressor, tmp_model_dir)

    def test_serialize_and_load_model_as_txt_Booster(self):
        """
        Test serialize and load model using TXT with Booster.
        """
        self.Booster_model.model_file_name = "test_Booster.txt"
        target_path = os.path.join(tmp_model_dir, "test_Booster.txt")
        self.Booster_model.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        loaded_model = lgb.Booster(model_file=target_path)
        assert all(loaded_model.predict(self.data) == self.bst.predict(self.data))

    def test_serialize_and_load_model_as_ONNX_Booster(self):
        """
        Test serialize and load model using ONNX with Booster.
        """
        self.Booster_model.model_file_name = "test_Booster.onnx"
        target_path = os.path.join(tmp_model_dir, "test_Booster.onnx")
        self.Booster_model.serialize_model(as_onnx=True)
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        pred_onx = sess.run(None, {"input": self.data.astype(np.float32)})[1]
        pred_lgbm = self.bst.predict(self.data)
        for i in range(len(pred_onx)):
            assert abs(pred_onx[i][1] - pred_lgbm[i]) <= 0.0000001

    def test_serialize_and_load_model_as_ONNX_LGBMClassifier(self):
        """
        Test serialize and load model using ONNX with LGBMClassifier.
        """
        target_path = os.path.join(tmp_model_dir, "test_LGBMClassifier.onnx")
        self.LGBMClassifier_model.model_file_name = "test_LGBMClassifier.onnx"
        self.LGBMClassifier_model.serialize_model(as_onnx=True)
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        prob_onx = sess.run(None, {"input": self.X_LGBMClassifier.astype(np.float32)})[
            1
        ]
        pred_lgbm = self.LGBMClassifier.predict(self.X_LGBMClassifier)
        pred_onx = []
        for pred in prob_onx:
            max_pred = max(pred.values())
            for key, val in pred.items():
                if val == max_pred:
                    pred_onx.append(key)
                    break
        assert pred_onx == list(pred_lgbm)

    def test_serialize_and_load_model_as_joblib_LGBMClassifier(self):
        """
        Test serialize and load model using joblib with LGBMClassifier.
        """
        LGBMClassifier_model = LightGBMModel(self.LGBMClassifier, tmp_model_dir)
        target_path = os.path.join(tmp_model_dir, "test_LGBMClassifier.joblib")
        LGBMClassifier_model.model_file_name = "test_LGBMClassifier.joblib"
        LGBMClassifier_model.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        with open(target_path, "rb") as f:
            loaded_model = joblib.load(f)
        assert all(
            loaded_model.predict(self.X_LGBMClassifier)
            == self.LGBMClassifier.predict(self.X_LGBMClassifier)
        )

    def test_serialize_and_load_model_as_joblib_LGBMRegressor(self):
        """
        Test serialize and load model using joblib with LGBMRegressor.
        """
        LGBMRegressor_model = LightGBMModel(self.LGBMRegressor, tmp_model_dir)
        target_path = os.path.join(tmp_model_dir, "test_LGBMRegressor.joblib")
        LGBMRegressor_model.model_file_name = "test_LGBMRegressor.joblib"
        LGBMRegressor_model.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        with open(target_path, "rb") as f:
            loaded_model = joblib.load(f)
        assert all(
            loaded_model.predict(self.X_LGBMRegressor)
            == self.LGBMRegressor.predict(self.X_LGBMRegressor)
        )

    def test_serialize_with_model_file_name(self):
        """
        Test correct and wrong model_file_name format.
        """
        self.LGBMClassifier_model.model_file_name = "test.abc"
        with pytest.raises(
            AssertionError,
            match="Wrong file extension. Expecting `.onnx` suffix.",
        ):
            self.LGBMClassifier_model._handle_model_file_name(
                as_onnx=True, model_file_name="test.abc"
            )

        self.LGBMClassifier_model.model_file_name = "test.joblib"
        assert (
            self.LGBMClassifier_model._handle_model_file_name(
                as_onnx=False, model_file_name="test.joblib"
            )
            == "test.joblib"
        )

    def test_serialize_without_model_file_name(self):
        self.LGBMClassifier_model.model_file_name = None
        assert (
            self.LGBMClassifier_model._handle_model_file_name(
                as_onnx=False, model_file_name="test.joblib"
            )
            == "test.joblib"
        )

    @pytest.mark.parametrize(
        "test_data",
        [pd.Series([1, 2, 3]), [1, 2, 3]],
    )
    def test_get_data_serializer_helper_with_convert_to_list(self, test_data):
        serialized_data = self.LGBMClassifier_model.get_data_serializer().serialize(
            test_data
        )
        assert serialized_data["data"] == [1, 2, 3]
        assert serialized_data["data_type"] == str(type(test_data))

    def test_get_data_serializer_helper_numpy(self):
        test_data = np.array([1, 2, 3])
        serialized_data = self.LGBMClassifier_model.get_data_serializer().serialize(
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
        serialized_data = self.LGBMClassifier_model.get_data_serializer().serialize(
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
        serialized_data = self.LGBMClassifier_model.get_data_serializer().serialize(
            test_data
        )
        assert serialized_data["data"] == test_data

    def test_get_data_serializer_helper_raise_error(self):
        class TestData:
            pass

        test_data = TestData()
        with pytest.raises(TypeError):
            serialized_data = self.LGBMClassifier_model.get_data_serializer().serialize(
                test_data
            )

    def test_X_sample_related_for_to_onnx(self):
        """
        Test if X_sample works in to_onnx propertly.
        """
        wrong_format = [1, 2, 3, 4]
        onnx_serializer = LightGBMOnnxModelSerializer()
        onnx_serializer.estimator = self.Booster_model.estimator
        assert isinstance(
            onnx_serializer._to_onnx(X_sample=wrong_format),
            onnx.onnx_ml_pb2.ModelProto,
        )

        onnx_serializer.estimator = None
        with pytest.raises(
            ValueError,
            match="`initial_types` can not be detected. Please directly pass initial_types.",
        ):
            onnx_serializer._to_onnx(X_sample=wrong_format)

    def test_lightgbm_to_onnx_with_lightgbm_uninstalled(self):
        """
        negative test for lightgbm
        """
        with mock.patch.dict(sys.modules, {"lightgbm": None}):
            with pytest.raises(ModuleNotFoundError):
                LightGBMModel(self.bst, tmp_model_dir)

    def teardown_class(cls):
        shutil.rmtree(tmp_model_dir, ignore_errors=True)

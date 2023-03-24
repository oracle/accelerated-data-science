#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit tests for model frameworks. Includes tests for:
 - SklearnModel
"""
import base64
import os
import shutil
from io import BytesIO

import numpy as np
import onnx
import onnxruntime as rt
import pandas as pd
import pytest
from ads.model.framework.sklearn_model import SklearnModel
from ads.model.serde.model_serializer import SklearnOnnxModelSerializer
from joblib import load
from lightgbm import LGBMClassifier, LGBMRegressor
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    StringTensorType,
)
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import sys, mock

tmp_model_dir = "/tmp/model"


class TestSklearnModel:
    """Unittests for the SklearnModel class."""

    def setup_class(cls):
        os.makedirs(tmp_model_dir, exist_ok=True)

        # Generate a Sklearn model using Iris dataset
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        (
            cls.X_train_iris,
            cls.X_test_iris,
            cls.y_train_iris,
            cls.y_test_iris,
        ) = train_test_split(X_iris, y_iris)
        model = LogisticRegression()
        model.fit(cls.X_train_iris, cls.y_train_iris)
        cls.sklearn_model = SklearnModel(model, tmp_model_dir)

        # Generate a pipeline with ColumnTransformer
        data = {
            "survived": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
            "pass_class": [1, 3, 1, 1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
            "sex": [
                "x",
                "y",
                "y",
                "x",
                "y",
                "x",
                "x",
                "x",
                "y",
                "y",
                "x",
                "x",
                "x",
                "x",
                "x",
                "y",
                "x",
                "x",
                "x",
                "y",
            ],
            "age": [
                15,
                63,
                32,
                np.nan,
                56,
                82,
                28,
                38,
                40,
                34,
                32,
                22,
                9,
                37,
                48,
                52,
                50,
                44,
                3,
                28,
            ],
            "fare": 100 * np.random.rand(20) + 10,
            "embarked": [
                "s",
                "s",
                "q",
                "s",
                "c",
                "q",
                "s",
                "c",
                "q",
                "s",
                "c",
                "s",
                "q",
                "s",
                "s",
                "c",
                "c",
                "q",
                "s",
                "q",
            ],
        }
        df = pd.DataFrame(data=data)
        X = df.drop(
            columns=[
                "survived",
            ]
        )
        y = df["survived"]
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2
        )
        cls.numeric_features = ["age", "fare"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_features = ["embarked", "sex", "pass_class"]
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, cls.numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(solver="lbfgs")),
            ]
        )
        clf.fit(cls.X_train, cls.y_train)
        cls.sklearn_pipeline_model = SklearnModel(clf, tmp_model_dir)

        # xgboost classifier pipeline
        data = load_iris()
        cls.X_iris = data.data[:, :2]
        cls.y_iris = data.target
        ind = np.arange(cls.X_iris.shape[0])
        np.random.shuffle(ind)
        cls.X_iris = cls.X_iris[ind, :].copy()
        cls.y_iris = cls.y_iris[ind].copy()
        cls.xgb = Pipeline(
            [("scaler", StandardScaler()), ("lgbm", XGBClassifier(n_estimators=3))]
        )
        cls.xgb.fit(cls.X_iris, cls.y_iris)
        cls.xgb_pipe = SklearnModel(cls.xgb, tmp_model_dir)

        # lightgbm classifier pipeline
        cls.lgb = Pipeline(
            [("scaler", StandardScaler()), ("lgbm", LGBMClassifier(n_estimators=3))]
        )
        cls.lgb.fit(cls.X_iris, cls.y_iris)
        cls.lgb_pipe = SklearnModel(cls.lgb, tmp_model_dir)

        # lightgbm regressor pipeline
        N = 100
        cls.X_reg = np.random.randn(N, 2)
        y = np.random.randn(N) + np.random.randn(N) * 100 * np.random.randint(0, 1, 100)

        cls.lgb_reg = Pipeline([("lgbm", LGBMRegressor(n_estimators=100))])
        cls.lgb_reg.fit(cls.X_reg, y)
        cls.lgb_reg_pipe = SklearnModel(cls.lgb_reg, tmp_model_dir)

    def test_to_onnx(self):
        """
        Test if SklearnOnnxModelSerializer.to_onnx generate onnx formate result.
        """
        onnx_serializer = SklearnOnnxModelSerializer()
        onnx_serializer.estimator = self.sklearn_model.estimator
        sklearn_onnx = onnx_serializer._to_onnx(
            X_sample=np.array([[1, 2, 3], [1, 2, 3]])
        )
        assert isinstance(sklearn_onnx, onnx.onnx_ml_pb2.ModelProto)

    def test_serialize_and_load_model_as_onnx(self):
        """
        Test if SklearnModel.serialize_model generate onnx file.
        Load serialzed onnx model from file and check prediction result.
        """
        self.sklearn_model.model_file_name = self.sklearn_model._handle_model_file_name(
            as_onnx=True, model_file_name=None
        )
        self.sklearn_model.serialize_model(as_onnx=True, X_sample=self.X_test_iris)
        target_path = os.path.join(tmp_model_dir, "model.onnx")
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run(
            [label_name], {input_name: self.X_test_iris.astype(np.float32)}
        )[0]
        pred_skl = self.sklearn_model.estimator.predict(
            self.X_test_iris.astype(np.float32)
        )
        pred_onx = pred_onx.ravel()
        pred_skl = pred_skl.ravel()
        d = np.abs(pred_onx - pred_skl)
        assert d.max() == 0

    def test_serialize_and_load_model_as_joblib(self):
        """
        Test if SklearnModel.serialize_model generate joblib file.
        Load serialzed joblib model from file and check prediction result.
        """
        self.sklearn_model.set_model_save_serializer(
            self.sklearn_model.model_save_serializer_type.JOBLIB
        )
        self.sklearn_model.model_file_name = "test.joblib"
        self.sklearn_model.serialize_model(as_onnx=False)
        target_path = os.path.join(tmp_model_dir, "test.joblib")
        assert os.path.exists(target_path)
        with open(target_path, "rb") as file:
            loaded_model = load(file)
        pred_joblib = loaded_model.predict(self.X_test_iris.astype(np.float32))
        pred_skl = self.sklearn_model.estimator.predict(
            self.X_test_iris.astype(np.float32)
        )
        pred_joblib = pred_joblib.ravel()
        pred_skl = pred_skl.ravel()
        d = np.abs(pred_joblib - pred_skl)
        assert d.max() == 0

    def test_serialize_with_model_file_name(self):
        """
        Test correct and wrong model_file_name format.
        """
        self.sklearn_model.model_file_name = self.sklearn_model._handle_model_file_name(
            as_onnx=True, model_file_name="test.onnx"
        )
        self.sklearn_model.serialize_model(as_onnx=True, X_sample=self.X_test_iris)
        assert os.path.exists(os.path.join(tmp_model_dir, "test.onnx"))

        with pytest.raises(
            AssertionError,
            match="Wrong file extension. Expecting `.onnx` suffix.",
        ):
            self.sklearn_model.model_file_name = (
                self.sklearn_model._handle_model_file_name(
                    as_onnx=True, model_file_name="test.abc"
                )
            )

    def test_serialize_to_onnx_with_initial_types(self):
        """
        Test generate onnx model with specific initial_types.
        """
        initial_type = [("float_input", FloatTensorType([None, 4]))]
        self.sklearn_model.model_file_name = "test.onnx"
        self.sklearn_model.serialize_model(
            as_onnx=True, initial_types=initial_type, force_overwrite=True
        )
        assert os.path.exists(os.path.join(tmp_model_dir, "test.onnx"))

    def test_serialize_using_pipeline_onnx(self):
        self.sklearn_pipeline_model.model_file_name = "test_pipeline.onnx"
        initial_inputs = []
        for k, v in zip(self.X_train.columns, self.X_train.dtypes):
            if v == "int64":
                t = Int64TensorType([None, 1])
            elif v == "float64":
                t = FloatTensorType([None, 1])
            else:
                t = StringTensorType([None, 1])
            initial_inputs.append((k, t))
        self.sklearn_pipeline_model.serialize_model(
            as_onnx=True, initial_types=initial_inputs
        )
        target_path = os.path.join(tmp_model_dir, "test_pipeline.onnx")
        assert os.path.exists(target_path)
        sess = rt.InferenceSession(target_path)

        inputs = {}
        for c in self.X_test.columns:
            inputs[c] = self.X_test[c].values
        for c in self.numeric_features:
            inputs[c] = inputs[c].astype(np.float32)
        for k in inputs:
            inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))
        assert sess.run(None, inputs) != None

    def test_serialize_using_pipeline_joblib(self):
        self.sklearn_pipeline_model.set_model_save_serializer(
            self.sklearn_pipeline_model.model_save_serializer_type.JOBLIB
        )
        self.sklearn_pipeline_model.model_file_name = "test_pipeline.joblib"
        self.sklearn_pipeline_model.serialize_model(as_onnx=False)
        target_path = os.path.join(tmp_model_dir, "test_pipeline.joblib")
        assert os.path.exists(target_path)
        with open(target_path, "rb") as file:
            loaded_model = load(file)
        assert len(loaded_model.predict(self.X_test)) != 0

    def test_serialize_and_load_model_as_onnx_xgboost_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with xgboost model.
        """

        target_dir = os.path.join(tmp_model_dir, "xgboost_pipeline_onnx")
        self.xgb_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.onnx")
        self.xgb_pipe.model_file_name = "test_pipeline.onnx"
        self.xgb_pipe.serialize_model(as_onnx=True, X_sample=self.X_iris[:10])
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: self.X_iris.astype(np.float32)})[
            0
        ]
        pred_xgboost = self.xgb.predict(self.X_iris)
        pred_onx = pred_onx.ravel()
        pred_xgboost = pred_xgboost.ravel()
        d = np.abs(pred_onx - pred_xgboost)
        assert d.max() <= 1

    def test_serialize_and_load_model_as_joblib_xgboost_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with xgboost model.
        """
        xgb_pipe = SklearnModel(self.xgb, tmp_model_dir)
        target_dir = os.path.join(tmp_model_dir, "xgboost_pipeline")
        xgb_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.joblib")
        xgb_pipe.model_file_name = "test_pipeline.joblib"
        xgb_pipe.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        with open(target_path, "rb") as file:
            loaded_model = load(file)
        assert len(loaded_model.predict(self.X_iris)) != 0

    def test_serialize_and_load_model_as_onnx_lgb_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with lightgbm model.
        """

        target_dir = os.path.join(tmp_model_dir, "lgb_pipeline_onnx")
        self.lgb_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.onnx")
        self.lgb_pipe.model_file_name = "test_pipeline.onnx"
        self.lgb_pipe.serialize_model(as_onnx=True, X_sample=self.X_iris)
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: self.X_iris.astype(np.float32)})[
            0
        ]
        pred_lgb = self.lgb.predict(self.X_iris)
        pred_onx = pred_onx.ravel()
        pred_lgb = pred_lgb.ravel()
        d = np.abs(pred_onx - pred_lgb)
        assert d.max() <= 1

    def test_serialize_and_load_model_as_joblib_lgb_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with lightgbm model.
        """
        lgb_pipe = SklearnModel(self.lgb, tmp_model_dir)
        target_dir = os.path.join(tmp_model_dir, "lgb_pipeline")
        lgb_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.joblib")
        lgb_pipe.model_file_name = "test_pipeline.joblib"
        lgb_pipe.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        with open(target_path, "rb") as file:
            loaded_model = load(file)
        assert len(loaded_model.predict(self.X_iris)) != 0

    def test_serialize_and_load_model_as_onnx_lgb_reg_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with lightgbm regressor model.
        """

        target_dir = os.path.join(tmp_model_dir, "lgb_reg_pipeline_onnx")
        self.lgb_reg_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.onnx")
        self.lgb_reg_pipe.model_file_name = "test_pipeline.onnx"
        self.lgb_reg_pipe.serialize_model(as_onnx=True, X_sample=self.X_reg)
        assert os.path.exists(target_path)

        sess = rt.InferenceSession(target_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onx = sess.run([label_name], {input_name: self.X_reg.astype(np.float32)})[
            0
        ]
        pred_lgb = self.lgb_reg.predict(self.X_reg)
        pred_onx = pred_onx.ravel()
        pred_lgb = pred_lgb.ravel()
        d = np.abs(pred_onx - pred_lgb)
        assert d.max() <= 1

    def test_serialize_and_load_model_as_joblib_lgb_reg_pipeline(self):
        """
        Test serialize and load pipeline using Sklearn API with lightgbm regressor model.
        """
        lgb_reg_pipe = SklearnModel(self.lgb_reg, tmp_model_dir)
        target_dir = os.path.join(tmp_model_dir, "lgb_reg_pipeline")
        lgb_reg_pipe.artifact_dir = target_dir
        target_path = os.path.join(target_dir, "test_pipeline.joblib")
        lgb_reg_pipe.model_file_name = "test_pipeline.joblib"
        lgb_reg_pipe.serialize_model(as_onnx=False)
        assert os.path.exists(target_path)

        with open(target_path, "rb") as file:
            loaded_model = load(file)
        assert len(loaded_model.predict(self.X_reg)) != 0

    def test_X_sample_related_for_to_onnx(self):
        """
        Test if X_sample works in to_onnx propertly.
        """
        wrong_format = {"key": [1, 2, 3, 4]}
        self.sklearn_model.estimator = None
        with pytest.raises(
            ValueError,
            match="`initial_types` can not be detected. Please directly pass initial_types.",
        ):
            sklearn_onnx_serializer = SklearnOnnxModelSerializer()
            sklearn_onnx_serializer.estimator = self.sklearn_model.estimator
            sklearn_onnx_serializer._to_onnx(X_sample=wrong_format)

    @pytest.mark.parametrize(
        "test_data",
        [pd.Series([1, 2, 3]), [1, 2, 3]],
    )
    def test_get_data_serializer_helper_with_convert_to_list(self, test_data):
        serialized_data = self.sklearn_model.get_data_serializer().serialize(test_data)
        assert serialized_data["data"] == [1, 2, 3]
        assert serialized_data["data_type"] == str(type(test_data))

    def test_get_data_serializer_helper_numpy(self):
        test_data = np.array([1, 2, 3])
        serialized_data = self.sklearn_model.get_data_serializer().serialize(test_data)
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
        serialized_data = self.sklearn_model.get_data_serializer().serialize(test_data)
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
        serialized_data = self.sklearn_model.get_data_serializer().serialize(test_data)
        assert serialized_data["data"] == test_data

    def test_get_data_serializer_helper_raise_error(self):
        class TestData:
            pass

        test_data = TestData()
        with pytest.raises(TypeError):
            serialized_data = self.sklearn_model.get_data_serializer().serialize(
                test_data
            )

    def test_to_onnx_with_onnx_uninstalled(self):
        """
        negative test for onnx
        """
        with mock.patch.dict(sys.modules, {"onnx": None}):
            with pytest.raises(ModuleNotFoundError):
                sklearn_onnx = SklearnOnnxModelSerializer()._to_onnx(
                    X_sample=np.array([[1, 2, 3], [1, 2, 3]])
                )

    def test_to_onnx_with_onnxmltools_uninstalled(self):
        """
        negative test for onnxmltools
        """
        with mock.patch.dict(sys.modules, {"onnxmltools": None}):
            with pytest.raises(ModuleNotFoundError):
                sklearn_onnx = SklearnOnnxModelSerializer()._to_onnx(
                    X_sample=np.array([[1, 2, 3], [1, 2, 3]])
                )

    def test_to_onnx_with_skl2onnx_uninstalled(self):
        """
        negative test for skl2onnx
        """
        with mock.patch.dict(sys.modules, {"skl2onnx": None}):
            with pytest.raises(ModuleNotFoundError):
                sklearn_onnx = SklearnOnnxModelSerializer()._to_onnx(
                    X_sample=np.array([[1, 2, 3], [1, 2, 3]])
                )

    def teardown_class(cls):
        shutil.rmtree(tmp_model_dir, ignore_errors=True)

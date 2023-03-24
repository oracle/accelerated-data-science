#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pytest
from sklearn import datasets, linear_model

from ads.model.generic_model import GenericModel
from ads.model.framework.sklearn_model import SklearnModel
from ads.model.framework.xgboost_model import XGBoostModel
from ads.feature_engineering.schema import Schema
import sklearn
import os
import shutil
import xgboost


class TestMetadataMixin:
    def setup_method(cls):
        # Load the diabetes dataset
        diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

        # Use only one feature
        diabetes_X = diabetes_X[:, np.newaxis, 2]

        # Split the data into training/testing sets
        cls.diabetes_X_train = diabetes_X[:-20]
        cls.diabetes_X_test = diabetes_X[-20:]

        # Split the targets into training/testing sets
        cls.diabetes_y_train = diabetes_y[:-20]
        cls.diabetes_y_test = diabetes_y[-20:]

        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Create Xgboost regression object
        from xgboost.sklearn import XGBRegressor

        xgb_regr = XGBRegressor()
        # Train the model using the training sets
        cls.rgr = regr.fit(cls.diabetes_X_train, cls.diabetes_y_train)
        cls.xgb_rgr = xgb_regr.fit(cls.diabetes_X_train, cls.diabetes_y_train)

    def test_metadata_generic_model(self):
        model = GenericModel(self.rgr, artifact_dir="~/test_generic")
        model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            namespace="ociodscdev",
            model_file_name="model.joblib",
            force_overwrite=True,
        )
        model.populate_metadata(ignore_pending_changes=True)
        assert model.metadata_custom.get("ModelSerializationFormat").value == "joblib"
        assert "score.py" in model.metadata_custom.get("ModelArtifacts").value
        assert "runtime.yaml" in model.metadata_custom.get("ModelArtifacts").value
        assert model.metadata_custom.get("SlugName").value == ""
        assert model.metadata_custom.get("CondaEnvironmentPath").value == ""
        assert model.metadata_custom.get("EnvironmentType").value == ""

        assert model.metadata_taxonomy.get("Algorithm").value == "None"
        assert model.metadata_taxonomy.get("Framework").value is None
        assert model.metadata_taxonomy.get("Hyperparameters").value is None
        assert model.metadata_taxonomy.get("FrameworkVersion").value is None

        assert model.schema_input == Schema()
        assert model.schema_output == Schema()

        assert model.metadata_provenance.artifact_dir == os.path.abspath(
            os.path.expanduser("~/test_generic")
        )
        assert model.metadata_provenance.training_id is None

    def test_metadata_generic_model_container_runtime(self):
        model = GenericModel(self.rgr, artifact_dir="~/test_generic")
        model.prepare(
            model_file_name="model.joblib",
            force_overwrite=True,
            ignore_conda_error=True,
        )
        model.populate_metadata(ignore_pending_changes=True, ignore_conda_error=True)

        for key in [
            "ModelSerializationFormat",
            "ModelArtifacts",
            "SlugName",
            "CondaEnvironmentPath",
            "EnvironmentType",
        ]:
            with pytest.raises(ValueError, match=f"The metadata with {key} not found."):
                model.metadata_custom.get(key)

        assert model.metadata_taxonomy.get("Algorithm").value == "None"
        assert model.metadata_taxonomy.get("Framework").value is None
        assert model.metadata_taxonomy.get("Hyperparameters").value is None
        assert model.metadata_taxonomy.get("FrameworkVersion").value is None

        assert model.schema_input == Schema()
        assert model.schema_output == Schema()

        assert model.metadata_provenance.artifact_dir == os.path.abspath(
            os.path.expanduser("~/test_generic")
        )
        assert model.metadata_provenance.training_id is None

    def test_metadata_sklearn_model(self):
        model = SklearnModel(self.rgr, artifact_dir="./test_sklearn")
        model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            inference_python_version="3.7",
            training_conda_env="dataexpl_p37_cpu_v3",
            training_python_version="3.7",
            namespace="ociodscdev",
            model_file_name="model.joblib",
        )
        model.populate_metadata(
            use_case_type="other",
            X_sample=self.diabetes_X_test,
            y_sample=self.diabetes_y_test,
        )

        assert model.metadata_custom.get("ModelSerializationFormat").value == "joblib"
        assert "score.py" in model.metadata_custom.get("ModelArtifacts").value
        assert "runtime.yaml" in model.metadata_custom.get("ModelArtifacts").value
        assert "model.joblib" in model.metadata_custom.get("ModelArtifacts").value
        assert model.metadata_custom.get("SlugName").value == "dataexpl_p37_cpu_v3"
        assert (
            model.metadata_custom.get("CondaEnvironmentPath").value
            == "oci://service-conda-packs@ociodscdev/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3"
        )
        assert model.metadata_custom.get("EnvironmentType").value == "data_science"

        assert model.metadata_taxonomy.get("Algorithm").value == "LinearRegression"
        assert model.metadata_taxonomy.get("Framework").value == "scikit-learn"
        assert model.metadata_taxonomy.get("Hyperparameters").value is not None
        assert (
            model.metadata_taxonomy.get("FrameworkVersion").value == sklearn.__version__
        )

        assert model.schema_input[0].description == "0"
        assert model.schema_output[0].description == "0"

        assert model.metadata_provenance.artifact_dir == os.path.abspath(
            os.path.expanduser("./test_sklearn")
        )
        assert model.metadata_provenance.training_id is None

    def test_metadata_xgboost_model(self):
        model = XGBoostModel(self.xgb_rgr, artifact_dir="./test_xgboost")
        model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            inference_python_version="3.7",
            training_conda_env="oci://service-conda-packs@ociodscdev/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
            training_python_version="3.7",
            model_file_name="model.json",
        )
        model.populate_metadata(
            use_case_type="binary_classification",
            X_sample=self.diabetes_X_test,
            y_sample=self.diabetes_y_test,
        )
        assert (
            model.metadata_custom.get("CondaEnvironment").value
            == "oci://service-conda-packs@ociodscdev/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3"
        )
        assert model.metadata_custom.get("SlugName").value == "dataexpl_p37_cpu_v3"
        assert model.metadata_custom.get("ModelSerializationFormat").value == "json"
        assert "score.py" in model.metadata_custom.get("ModelArtifacts").value
        assert "runtime.yaml" in model.metadata_custom.get("ModelArtifacts").value
        assert "model.json" in model.metadata_custom.get("ModelArtifacts").value
        assert model.metadata_custom.get("SlugName").value == "dataexpl_p37_cpu_v3"

        assert model.metadata_custom.get("EnvironmentType").value == "data_science"

        assert (
            model.metadata_taxonomy.get("UseCaseType").value == "binary_classification"
        )
        assert model.metadata_taxonomy.get("Algorithm").value == "XGBRegressor"
        assert model.metadata_taxonomy.get("Framework").value == "xgboost"
        assert model.metadata_taxonomy.get("Hyperparameters").value is not None
        assert (
            model.metadata_taxonomy.get("FrameworkVersion").value == xgboost.__version__
        )

        assert model.schema_input[0].description == "0"
        assert model.schema_output[0].description == "0"

        assert model.metadata_provenance.artifact_dir == os.path.abspath(
            os.path.expanduser("./test_xgboost")
        )
        assert model.metadata_provenance.training_id is None
        assert (
            model.runtime_info.model_deployment.inference_conda_env.inference_env_path
            == "oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3"
        )

    def teardown_method(self):
        for dir in ["~/test_generic", "./test_sklearn", "./test_xgboost"]:
            if os.path.exists(dir):
                shutil.rmtree(dir)

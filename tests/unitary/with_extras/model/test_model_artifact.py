#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pickle
import sys

import mock
import pytest
import yaml
from ads.common.model import ADSModel
from ads.common.model_artifact import MODEL_ARTIFACT_VERSION
from ads.common.model_export_util import prepare_generic_model
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TestModelArtifact:
    """Contains test cases for ads/common/model_artifact.py."""

    compartment_id = "9898989898"  # os.environ['NB_SESSION_COMPARTMENT_OCID']
    project_id = "89898989898989"  # os.environ["PROJECT_OCID"]

    clf = None
    X_sample = None
    y_sample = None

    iris = datasets.load_iris(as_frame=True)
    X, y = iris["data"], iris["target"]
    X, y = iris["data"], iris["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = RandomForestClassifier().fit(X_train, y_train)
    X_sample = X_train.head(3)
    y_sample = y_train.head(3)

    @pytest.fixture(autouse=True)
    def conda_file(self, tmpdir_factory):
        conda_file = tmpdir_factory.mktemp("conda")
        manifest = {
            "manifest": {
                "pack_path": "pack_path: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/pyspark/1.0/pyspv10",
                "python": "3.6",
                "slug": "pyspv10",
                "type": "data_science",
                "version": "1.0",
                "arch_type": "CPU",
                "manifest_version": "1.0",
                "name": "pyspark",
            }
        }
        with open(os.path.join(conda_file.strpath, "test_manifest.yaml"), "w") as mfile:
            yaml.dump(manifest, mfile)

        conda_prefix = os.environ["CONDA_PREFIX"]
        os.environ["CONDA_PREFIX"] = conda_file.strpath
        yield conda_file
        os.environ["CONDA_PREFIX"] = conda_prefix

    @pytest.fixture(autouse=True, scope="module")
    def model(self):
        # build model
        model = ADSModel.from_estimator(self.clf)
        return model

    def test_prepare_artifact(self, tmpdir):
        path = os.path.join(tmpdir, "model")
        os.makedirs(path)
        with open(os.path.join(path, "model.pkl"), "wb") as mfile:
            pickle.dump(self.clf, mfile)
        value = os.environ.pop("CONDA_PREFIX", None)
        prepare_generic_model(path, force_overwrite=True, ignore_deployment_error=True)
        expected_output = f"""
MODEL_ARTIFACT_VERSION: '{MODEL_ARTIFACT_VERSION}'
MODEL_DEPLOYMENT:
    INFERENCE_CONDA_ENV:
        INFERENCE_ENV_SLUG: <slug of the conda environment>
        INFERENCE_ENV_TYPE: 'published'
        INFERENCE_ENV_PATH: oci://<bucket-name>@<namespace>/<prefix>/<env>.tar.gz
        INFERENCE_PYTHON_VERSION: <python version>
"""
        assert yaml.load(expected_output, Loader=yaml.FullLoader) == yaml.load(
            open(os.path.join(path, "runtime.yaml")).read(), Loader=yaml.FullLoader
        )
        if value:
            os.environ["CONDA_PREFIX"] = value

    def test_prepare_artifact_conda_info(self, tmpdir):
        path = os.path.join(tmpdir, "model")
        os.makedirs(path)
        with open(os.path.join(path, "model.pkl"), "wb") as mfile:
            pickle.dump(self.clf, mfile)
        value = os.environ.pop("CONDA_PREFIX", None)
        inference_conda_env = "oci://mybucket@mynamespace/test/condapackv1"
        inference_python_version = "3.7"
        prepare_generic_model(
            path,
            force_overwrite=True,
            ignore_deployment_error=True,
            inference_conda_env=inference_conda_env,
            inference_python_version=inference_python_version,
        )
        expected_output = f"""
MODEL_ARTIFACT_VERSION: '{MODEL_ARTIFACT_VERSION}'
MODEL_DEPLOYMENT:
    INFERENCE_CONDA_ENV:
        INFERENCE_ENV_SLUG: ''
        INFERENCE_ENV_TYPE: 'published'
        INFERENCE_ENV_PATH: {inference_conda_env}
        INFERENCE_PYTHON_VERSION: '{inference_python_version}'
"""
        assert yaml.load(expected_output, Loader=yaml.FullLoader) == yaml.load(
            open(os.path.join(path, "runtime.yaml")).read(), Loader=yaml.FullLoader
        )
        if value:
            os.environ["CONDA_PREFIX"] = value

    @pytest.mark.skip(reason="Test case seem to be invalid")
    def test_prepare_with_schema_with_exception(self, model, conda_file):
        with pytest.raises(
            Exception,
            match="The inference environment pyspv10 may have undergone changes over the course of development. You can choose to publish the current environment or set data_science_env to True in the prepare api",
        ):
            model.prepare(
                conda_file.strpath,
                X_sample=self.X_sample,
                y_sample=self.y_sample,
                force_overwrite=True,
            )

    def test_prepare_with_schema(self, model, conda_file):
        model.prepare(
            conda_file.strpath,
            X_sample=self.X_sample,
            y_sample=self.y_sample,
            force_overwrite=True,
            data_science_env=True,
        )
        assert os.path.exists(
            os.path.join(conda_file.strpath, "score.py")
        ), "score.py does not exist"
        assert os.path.exists(
            os.path.join(conda_file.strpath, "schema_input.json")
        ), "schema_input.json does not exist"
        assert os.path.exists(
            os.path.join(conda_file.strpath, "schema_output.json")
        ), "schema_output.json does not exist"

    def test_prepare_with_schema(self, model, conda_file):
        model.prepare(
            conda_file.strpath,
            X_sample=self.X_sample,
            y_sample=self.y_sample,
            force_overwrite=True,
            data_science_env=True,
        )
        assert os.path.exists(
            os.path.join(conda_file.strpath, "score.py")
        ), "score.py does not exist"
        assert os.path.exists(
            os.path.join(conda_file, "input_schema.json")
        ), "schema_input.json does not exist"
        assert os.path.exists(
            os.path.join(conda_file, "output_schema.json")
        ), "schema_output.json does not exist"

    def test_prepare_with_no_schema(self, model, conda_file):
        with pytest.raises(
            AssertionError,
            match="You must provide a data sample to infer the input and output data types which are used when converting the the model to an equivalent onnx model. This can be done as an ADSData object with the parameter `data_sample`, or as X and y samples to X_sample and y_sample respectively.",
        ):
            model.prepare(
                conda_file.strpath, force_overwrite=True, data_science_env=True
            )

    def test_script_in_artifact_dir(self, model, conda_file):
        model_artifact = model.prepare(
            conda_file.strpath,
            X_sample=self.X_sample,
            y_sample=self.y_sample,
            force_overwrite=True,
            data_science_env=True,
        )
        self._test_predict(model_artifact, model)

    def test_check_featurenames(self, model, conda_file):
        names = model.feature_names(self.X_sample)
        import numpy as np

        assert np.array_equal(names.values, self.X_sample.columns.values)

    def _test_predict(self, model_artifact, model):
        model_artifact.reload()
        est_pred = model.predict(self.X_sample)
        art_pred = model_artifact.predict(self.X_sample)["prediction"]
        # TODO: this line block tests/unitary/test_text_dataset_dataloader.py
        # fn_pred = model_artifact.verify({'input': self.X_sample.to_dict()})['prediction']
        assert est_pred is not None
        assert all(
            est_pred == art_pred
        ), "the score.py prediction is not aligned with the estimators prediction"
        # assert art_pred == fn_pred, "The func.py script is miss-handling the invoking of score.py (score.py is " \
        #                                  "consistent with the est.predict output). "

    def test_prepare_without_force(self, model, conda_file):
        with pytest.raises(
            ValueError, match="Directory already exists, set force to overwrite"
        ):
            model.prepare(
                conda_file.strpath,
                X_sample=self.X_sample,
                y_sample=self.y_sample,
                data_science_env=True,
            )

    def test_fetch_runtime_schema_with_python_jsonschema_objects_uninstalled(
        self, model, conda_file
    ):
        with mock.patch.dict(sys.modules, {"python_jsonschema_objects": None}):
            with pytest.raises(ModuleNotFoundError):
                model_artifact = model.prepare(
                    conda_file.strpath,
                    X_sample=self.X_sample,
                    y_sample=self.y_sample,
                    force_overwrite=True,
                    data_science_env=True,
                )
                model_artifact._generate_runtime_yaml()

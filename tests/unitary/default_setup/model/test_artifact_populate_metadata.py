#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import git
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
import yaml
from ads.common.model import ADSModel
from ads.common.model_artifact import ModelArtifact
from ads.common.model_export_util import prepare_generic_model
from ads.model.extractor.model_info_extractor_factory import ModelInfoExtractorFactory
from ads.model.model_metadata import (
    METADATA_SIZE_LIMIT,
    ModelMetadataItem,
    MetadataTaxonomyKeys,
)
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

tmp_model_dir = "/tmp/model"


@pytest.fixture
def conda_file(tmpdir_factory):
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

    conda_prefix = os.getenv("CONDA_PREFIX", None)
    os.environ["CONDA_PREFIX"] = conda_file.strpath
    yield conda_file
    if conda_prefix:
        os.environ["CONDA_PREFIX"] = conda_prefix


class TestModelArtifactPopulateMetadata:
    """Contains test cases for ads.catalog.model.py"""

    @patch("builtins.print")
    def test_model_artifact_populate_metadata(self, mock_print, conda_file):
        # prepare model artifact
        iris = load_iris(as_frame=True)
        X, y = iris["data"], iris["target"]
        clf = RandomForestClassifier().fit(X, y)
        X_sample = X.head(3)
        y_sample = y.head(3)
        rf_model = ADSModel.from_estimator(clf)
        model_artifact = rf_model.prepare(
            conda_file.strpath,
            X_sample=X_sample,
            y_sample=y_sample,
            force_overwrite=True,
            data_science_env=True,
        )

        # populate taxonomy metadata
        model_artifact.populate_metadata(rf_model)
        map = ModelInfoExtractorFactory.extract_info(rf_model)
        assert model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.ALGORITHM
        ).value == str(map[MetadataTaxonomyKeys.ALGORITHM])
        assert model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.FRAMEWORK
        ).value == str(map[MetadataTaxonomyKeys.FRAMEWORK])
        assert model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.FRAMEWORK_VERSION
        ).value == str(map[MetadataTaxonomyKeys.FRAMEWORK_VERSION])
        assert (
            model_artifact.metadata_taxonomy.get(
                MetadataTaxonomyKeys.HYPERPARAMETERS
            ).value
            == map[MetadataTaxonomyKeys.HYPERPARAMETERS]
        )
        assert (
            model_artifact.metadata_taxonomy.get(
                MetadataTaxonomyKeys.USE_CASE_TYPE
            ).value
            == None
        )

    def test_prepare_generic_model_populate_metadata_taxonomy(self):
        # prepare model artifact
        if not os.path.exists(tmp_model_dir):
            os.mkdir(tmp_model_dir)
        gamma_reg_model = linear_model.GammaRegressor()
        train_X = [[1, 2], [2, 3], [3, 4], [4, 3]]
        train_y = [19, 26, 33, 30]
        gamma_reg_model.fit(train_X, train_y)
        gamma_reg_model.score(train_X, train_y)
        test_X = [[1, 0], [2, 8]]
        gamma_reg_model.predict(test_X)

        generic_model_artifact = prepare_generic_model(
            tmp_model_dir,
            force_overwrite=True,
            model=gamma_reg_model,
            ignore_deployment_error=True,
            use_case_type="regression",
        )

        map = ModelInfoExtractorFactory.extract_info(gamma_reg_model)
        assert generic_model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.ALGORITHM
        ).value == str(map[MetadataTaxonomyKeys.ALGORITHM])
        assert generic_model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.FRAMEWORK
        ).value == str(map[MetadataTaxonomyKeys.FRAMEWORK])
        assert generic_model_artifact.metadata_taxonomy.get(
            MetadataTaxonomyKeys.FRAMEWORK_VERSION
        ).value == str(map[MetadataTaxonomyKeys.FRAMEWORK_VERSION])
        assert (
            generic_model_artifact.metadata_taxonomy.get(
                MetadataTaxonomyKeys.HYPERPARAMETERS
            ).value
            == map[MetadataTaxonomyKeys.HYPERPARAMETERS]
        )
        assert (
            generic_model_artifact.metadata_taxonomy.get(
                MetadataTaxonomyKeys.USE_CASE_TYPE
            ).value
            == "regression"
        )

        shutil.rmtree(tmp_model_dir)

    @patch.object(git.Repo, "active_branch")
    def test_prepare_generic_model_extract_metadata_custom(
        self, mock_active_branch, monkeypatch, conda_file
    ):
        monkeypatch.setenv(
            "NB_SESSION_OCID", "ocid1.datasciencenotebook.oc1.iad.<unique_ocid>"
        )
        generic_model_artifact = prepare_generic_model(
            conda_file.strpath,
            force_overwrite=True,
            ignore_deployment_error=True,
        )
        generic_model_artifact.populate_metadata()
        if hasattr(generic_model_artifact, "conda_env"):
            conda_env = generic_model_artifact.conda_env
        else:
            conda_env = None

        if hasattr(
            generic_model_artifact._runtime_info.MODEL_DEPLOYMENT, "INFERENCE_CONDA_ENV"
        ):
            env_type = (
                generic_model_artifact._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_TYPE._value
            )
            slug_name = (
                generic_model_artifact._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_SLUG._value
            )
            env_path = (
                generic_model_artifact._runtime_info.MODEL_DEPLOYMENT.INFERENCE_CONDA_ENV.INFERENCE_ENV_PATH._value
            )
        else:
            env_type = None
            slug_name = None
            env_path = None

        assert (
            generic_model_artifact.metadata_custom.get("CondaEnvironment").value
            == conda_env
        )
        assert (
            generic_model_artifact.metadata_custom.get("EnvironmentType").value
            == env_type
        )
        assert generic_model_artifact.metadata_custom.get("SlugName").value == slug_name
        assert (
            generic_model_artifact.metadata_custom.get("CondaEnvironmentPath").value
            == env_path
        )

        assert (
            "score.py"
            in generic_model_artifact.metadata_custom.get("ModelArtifacts").value
        )
        assert (
            "runtime.yaml"
            in generic_model_artifact.metadata_custom.get("ModelArtifacts").value
        )

    @patch.object(ModelMetadataItem, "to_json_file")
    def test_populate_metadata_taxonomy(self, mock_to_json_file):
        """Tests populating taxonomy metadata."""
        mock_artifact_folder = "model_artifact"
        mock_model_artifact = ModelArtifact(
            mock_artifact_folder,
            reload=False,
            create=False,
            ignore_deployment_error=True,
        )
        mock_metadata_info_map = {
            MetadataTaxonomyKeys.HYPERPARAMETERS: [[1] * METADATA_SIZE_LIMIT]
        }
        mock_model = MagicMock()

        with patch.object(
            ModelInfoExtractorFactory,
            "extract_info",
            return_value=mock_metadata_info_map,
        ) as mock_extract_info:
            mock_model_artifact._populate_metadata_taxonomy(mock_model)
            mock_extract_info.assert_called_with(mock_model)
            mock_to_json_file.assert_called_with(mock_artifact_folder)
            assert (
                mock_model_artifact.metadata_taxonomy[
                    MetadataTaxonomyKeys.HYPERPARAMETERS
                ].value
                == None
            )

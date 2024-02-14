#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from unittest import mock
import oci
import unittest
from collections import namedtuple
from dataclasses import asdict
from unittest.mock import ANY, MagicMock, patch

from ads.aqua.model import AquaModelApp, AquaModelSummary
from ads.model.datascience_model import DataScienceModel

MOCK_DATASCIENCE_MODEL = """
kind: datascienceModel
spec:
  artifact: test_artifact
  compartmentId: test_compartment_id
  customMetadataList:
    data:
    - category: training environment
      description: The conda environment where the model was trained.
      key: CondaEnvironment
      value: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/Oracle_AutoMLx_v23.4_for_CPU_on_Python_3.8/1.0/automlx234_p38_cpu_x86_64_v1
    - category: training profile
      description: The model serialization format.
      key: ModelSerializationFormat
      value: pkl
    - category: training environment
      description: The URI of the training conda environment.
      key: CondaEnvironmentPath
      value: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/Oracle_AutoMLx_v23.4_for_CPU_on_Python_3.8/1.0/automlx234_p38_cpu_x86_64_v1
    - category: other
      description: model by reference storage path
      key: Object Storage Path
      value: oci://test_path
    - category: training environment
      description: The slug name of the training conda environment.
      key: SlugName
      value: automlx234_p38_cpu_x86_64_v1
    - category: training environment
      description: The conda environment type, can be published or datascience.
      key: EnvironmentType
      value: data_science
    - category: other
      description: The model file name.
      key: ModelFileName
      value: model.pkl
    - category: training environment
      description: The list of files located in artifacts folder.
      key: ModelArtifacts
      value: .model-ignore, test_json_output.json, score.py, runtime.yaml, output_schema.json,
        input_schema.json, model.pkl, icon.txt
    - category: other
      description: null
      key: ClientLibrary
      value: ADS
  definedMetadataList:
    data:
    - key: ArtifactTestResults
      value:
        runtime_env_path:
          category: conda_env
          description: Check that field MODEL_DEPLOYMENT.INFERENCE_ENV_PATH is set
          error_msg: In runtime.yaml, the key MODEL_DEPLOYMENT.INFERENCE_ENV_PATH
            must have a value.
          success: true
          value: oci://service-conda-packs@id19sfcrra6z/service_pack/cpu/Oracle_AutoMLx_v23.4_for_CPU_on_Python_3.8/1.0/automlx234_p38_cpu_x86_64_v1
        runtime_env_python:
          category: conda_env
          description: Check that field MODEL_DEPLOYMENT.INFERENCE_PYTHON_VERSION
            is set to a value of 3.6 or higher
          error_msg: In runtime.yaml, the key MODEL_DEPLOYMENT.INFERENCE_PYTHON_VERSION
            must be set to a value of 3.6 or higher.
          success: true
          value: '3.8'
        runtime_path_exist:
          category: conda_env
          description: Check that the file path in MODEL_DEPLOYMENT.INFERENCE_ENV_PATH
            is correct.
          error_msg: In runtime.yaml, the key MODEL_DEPLOYMENT.INFERENCE_ENV_PATH
            does not exist.
          success: true
        runtime_version:
          category: runtime.yaml
          description: Check that field MODEL_ARTIFACT_VERSION is set to 3.0
          error_msg: In runtime.yaml, the key MODEL_ARTIFACT_VERSION must be set to
            3.0.
          success: true
        runtime_yaml:
          category: Mandatory Files Check
          description: Check that the file "runtime.yaml" exists and is in the top
            level directory of the artifact directory
          error_msg: The file 'runtime.yaml' is missing.
          success: true
        score_load_model:
          category: score.py
          description: Check that load_model() is defined
          error_msg: Function load_model is not present in score.py.
          success: true
        score_predict:
          category: score.py
          description: Check that predict() is defined
          error_msg: Function predict is not present in score.py.
          success: true
        score_predict_arg:
          category: score.py
          description: Check that all other arguments in predict() are optional and
            have default values
          error_msg: All formal arguments in the predict function must have default
            values, except that 'data' argument.
          success: true
        score_predict_data:
          category: score.py
          description: Check that the only required argument for predict() is named
            "data"
          error_msg: The predict function in score.py must have a formal argument
            named 'data'.
          success: true
        score_py:
          category: Mandatory Files Check
          description: Check that the file "score.py" exists and is in the top level
            directory of the artifact directory
          error_msg: The file 'score.py' is missing.
          key: score_py
          success: true
        score_syntax:
          category: score.py
          description: Check for Python syntax errors
          error_msg: 'There is Syntax error in score.py: '
          success: true
    - key: FrameworkVersion
      value: null
    - key: Hyperparameters
      value: null
    - key: Framework
      value: null
    - key: UseCaseType
      value: multinomial_classification
    - key: Algorithm
      value: None
  displayName: Mistral-7B-Instruct-v0.1-Fine-Tuned
  freeformTags:
    OCIAQUA: ''
    aquaFineTunedModel: test_aqua_fine_tuned_model
    license: Apache
    organization: Mistral AI
    task: text_generation
  id: test_id
  inputSchema:
    schema:
    - description: '0'
      domain:
        constraints: []
        stats:
          count: 10.0
          lower quartile: 14.75
          mean: 40.2
          median: 29.5
          sample maximum: 94.0
          sample minimum: 3.0
          standard deviation: 33.77309909117347
          upper quartile: 69.5
        values: Integer
      dtype: int64
      feature_type: Integer
      name: 0
      order: 0
      required: true
    version: '1.1'
  outputSchema:
    schema:
    - description: '0'
      domain:
        constraints: []
        stats:
          count: 10.0
          lower quartile: 219.25
          mean: 2642.6
          median: 872.5
          sample maximum: 8836.0
          sample minimum: 9.0
          standard deviation: 3482.95000123618
          upper quartile: 5227.0
        values: Integer
      dtype: int64
      feature_type: Integer
      name: 0
      order: 0
      required: true
    version: '1.1'
  projectId: test_project_id
  provenanceMetadata:
    artifactDir: null
    gitBranch: null
    gitCommit: null
    repositoryUrl: null
    trainingId: null
    trainingScriptPath: null
type: dataScienceModel
"""


class TestDataset:
    Response = namedtuple("Response", ["data", "status"])
    DataList = namedtuple("DataList", ["objects"])

    resource_summary_objects = []

    COMPARTMENT_ID = "ocid1.compartment.oc1..<OCID>"
    MOCK_ICON = "data:image/svg+xml;base64,########"


class TestAquaModel(unittest.TestCase):
    """Contains unittests for AquaModelApp."""

    @patch("ads.aqua.model.DataScienceModel.create")
    @patch("ads.aqua.model.DataScienceModel.download_artifact")
    @patch("ads.aqua.model.DataScienceModel.from_id")
    def test_create_from_custom_model(
        self, 
        mock_from_id, 
        mock_download_artifact, 
        mock_create
    ):
        mock_from_id.return_value = DataScienceModel.from_yaml(MOCK_DATASCIENCE_MODEL)
        model = AquaModelApp().create(
            model_id="test_model_id",
            project_id="test_project_id",
            comparment_id="test_compartment_id"
        )

        mock_from_id.assert_called_with("test_model_id")
        mock_download_artifact.assert_not_called()
        mock_create.assert_not_called()

    @patch("ads.aqua.model.DataScienceModel.create")
    @patch("ads.aqua.model.DataScienceModel.download_artifact")
    @patch("ads.aqua.model.DataScienceModel.from_id")
    def test_create_from_service_model(
        self, 
        mock_from_id, 
        mock_download_artifact, 
        mock_create
    ):
        mock_from_id.return_value = DataScienceModel.from_yaml(MOCK_DATASCIENCE_MODEL)
        with patch.dict(
            os.environ,
            {"ODSC_MODEL_COMPARTMENT_OCID": "test_compartment_id"},
            clear=True
        ):
            model = AquaModelApp().create(
                model_id="test_model_id",
                project_id="test_project_id",
                comparment_id="test_compartment_id"
            )

            assert os.environ.get("ODSC_MODEL_COMPARTMENT_OCID") == "test_compartment_id"
            assert mock_from_id.return_value.compartment_id == "test_compartment_id"

            mock_from_id.assert_called_with("test_model_id")
            mock_download_artifact.assert_called()
            #mock_create.assert_called()


    def test_list(self):
        """Tests list models succesfully."""
        pass

    def test_list_failed(self):
        """Tests list models succesfully."""
        pass

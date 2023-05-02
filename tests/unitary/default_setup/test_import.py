#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads import *
from ads.hpo.distributions import *
from ads.hpo.stopping_criterion import *


def test_import():
    import ads
    from ads.feature_engineering.feature_type.adsstring.oci_language import OCILanguage
    from ads.feature_engineering.feature_type.adsstring.string import ADSString
    from ads.text_dataset.backends import Base
    from ads.text_dataset.dataset import TextDatasetFactory as textfactory
    from ads.text_dataset.extractor import FileProcessor, FileProcessorFactory
    from ads.text_dataset.options import Options

    from ads.secrets.adb import ADBSecretKeeper
    from ads.secrets.oracledb import OracleDBSecretKeeper
    from ads.secrets.mysqldb import MySQLDBSecretKeeper

    from ads.secrets.auth_token import AuthTokenSecretKeeper
    from ads.secrets.big_data_service import BDSSecretKeeper
    from ads.bds.auth import has_kerberos_ticket, refresh_ticket, krbcontext

    from ads.dataset.factory import DatasetFactory
    from ads.automl.driver import AutoML
    from ads.automl.provider import OracleAutoMLProvider
    from ads.catalog.model import ModelCatalog
    from ads.model.model_metadata import UseCaseType
    from ads.dataset.dataset_browser import DatasetBrowser
    from ads.model.framework.automl_model import AutoMLModel

    from ads.model.generic_model import GenericModel

    from ads.model.framework.lightgbm_model import LightGBMModel
    from ads.model.framework.pytorch_model import PyTorchModel
    from ads.model.framework.sklearn_model import SklearnModel
    from ads.model.framework.tensorflow_model import TensorFlowModel
    from ads.model.framework.xgboost_model import XGBoostModel
    from ads.common.oci_logging import OCILogGroup

    # from ads.explanations.mlx_global_explainer import MLXGlobalExplainer
    # from ads.explanations.explainer import ADSExplainer
    from ads.evaluations.evaluator import ADSEvaluator
    from ads.model.deployment import ModelDeployer, ModelDeploymentProperties
    from ads.jobs import ScriptRuntime
    from ads.jobs import DataFlow, DataFlowRun, DataFlowRuntime
    from ads.jobs import Job, DataScienceJob, GitPythonRuntime
    from ads.jobs import NotebookRuntime
    from ads.jobs import DataScienceJobRun
    from ads.common.data import ADSData
    from ads.common.model import ADSModel
    from ads.evaluations.evaluation_plot import EvaluationPlot

    # from ads.hpo.search_cv import ADSTuner
    from ads.common.model import ADSModel
    from ads.common.model_artifact import ModelArtifact, _TRAINING_RESOURCE_OCID
    from ads.common.model_export_util import prepare_generic_model
    from ads.model.extractor.model_info_extractor_factory import (
        ModelInfoExtractorFactory,
    )
    from ads.model.model_metadata import (
        METADATA_SIZE_LIMIT,
        ModelMetadataItem,
        MetadataTaxonomyKeys,
    )
    from ads.model.extractor.tensorflow_extractor import TensorflowExtractor
    from ads.model.extractor.pytorch_extractor import PytorchExtractor
    from ads.model.runtime.model_provenance_details import (
        ModelProvenanceDetails,
        TrainingCode,
    )

    from ads.feature_engineering.schema import Schema
    from ads.config import JOB_RUN_OCID, NB_SESSION_OCID
    from ads.model.runtime.model_deployment_details import ModelDeploymentDetails
    from ads.model.deployment.model_deployment import (
        ModelDeployment,
        ModelDeploymentLogType,
    )
    from ads.model.common.utils import _extract_locals
    from ads.common.error import ChangesNotCommitted
    from ads.model.model_metadata import ModelProvenanceMetadata
    from ads.model.deployment.common.utils import State as ModelDeploymentState
    from ads.model.generic_model import GenericModel, NotActiveDeploymentError
    from ads.model.runtime.runtime_info import RuntimeInfo
    from ads.model.model_properties import ModelProperties
    from ads.model.artifact import (
        AritfactFolderStructureError,
        ArtifactNestedFolderError,
        ArtifactRequiredFilesError,
        ModelArtifact,
        _validate_artifact_dir,
    )
    from ads.model.runtime.env_info import InferenceEnvInfo, TrainingEnvInfo

    from ads.text_dataset.backends import OITCC, Base, Tika, PDFPlumber
    from ads.text_dataset.dataset import TextDatasetFactory as textfactory
    from ads.text_dataset.options import Options
    from ads.text_dataset.extractor import NotSupportedError
    from ads.text_dataset.backends import Base, Tika, PDFPlumber
    from ads.text_dataset.extractor import (
        FileProcessor,
        FileProcessorFactory,
        NotSupportedError,
        PDFProcessor,
        WordProcessor,
    )
    from ads.model.framework.huggingface_model import HuggingFacePipelineModel
    from ads.model.extractor.huggingface_extractor import HuggingFaceExtractor

    assert True

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import oci
from cachetools import TTLCache
from oci.data_science.models import (
    JobRun,
    Metadata,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.data import Tags
from ads.aqua.exception import (
    AquaFileExistsError,
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.utils import (
    JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING,
    NB_SESSION_IDENTIFIER,
    UNKNOWN,
    fire_and_forget,
    get_container_image,
    is_valid_ocid,
    upload_local_to_os,
)
from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link, get_files, get_log_links, upload_to_os
from ads.config import (
    AQUA_JOB_SUBNET_ID,
    COMPARTMENT_OCID,
    CONDA_BUCKET_NS,
    PROJECT_OCID,
)
from ads.jobs.ads_job import DataScienceJobRun, Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.jobs.builders.runtimes.base import Runtime
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment.model_deployment import ModelDeployment
from ads.model.model_metadata import (
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.model_version_set import ModelVersionSet
from ads.telemetry import telemetry

EVAL_TERMINATION_STATE = [
    JobRun.LIFECYCLE_STATE_SUCCEEDED,
    JobRun.LIFECYCLE_STATE_FAILED,
]


class EvaluationJobExitCode(Enum):
    SUCCESS = 0
    COMMON_ERROR = 1

    # Configuration-related issues
    INVALID_EVALUATION_CONFIG = 10
    EVALUATION_CONFIG_NOT_PROVIDED = 11
    INVALID_OUTPUT_DIR = 12
    INVALID_INPUT_DATASET_PATH = 13
    INVALID_EVALUATION_ID = 14
    INVALID_TARGET_EVALUATION_ID = 15
    INVALID_EVALUATION_CONFIG_VALIDATION = 16

    # Evaluation process issues
    OUTPUT_DIR_NOT_FOUND = 20
    INVALID_INPUT_DATASET = 21
    INPUT_DATA_NOT_FOUND = 22
    EVALUATION_ID_NOT_FOUND = 23
    EVALUATION_ALREADY_PERFORMED = 24
    EVALUATION_TARGET_NOT_FOUND = 25
    NO_SUCCESS_INFERENCE_RESULT = 26
    COMPUTE_EVALUATION_ERROR = 27
    EVALUATION_REPORT_ERROR = 28
    MODEL_INFERENCE_WRONG_RESPONSE_FORMAT = 29
    UNSUPPORTED_METRICS = 30
    METRIC_CALCULATION_FAILURE = 31


EVALUATION_JOB_EXIT_CODE_MESSAGE = {
    EvaluationJobExitCode.SUCCESS.value: "Success",
    EvaluationJobExitCode.COMMON_ERROR.value: "An error occurred during the evaluation, please check the log for more information.",
    EvaluationJobExitCode.INVALID_EVALUATION_CONFIG.value: "The provided evaluation configuration was not in the correct format, supported formats are YAML or JSON.",
    EvaluationJobExitCode.EVALUATION_CONFIG_NOT_PROVIDED.value: "The evaluation config was not provided.",
    EvaluationJobExitCode.INVALID_OUTPUT_DIR.value: "The specified output directory path is invalid.",
    EvaluationJobExitCode.INVALID_INPUT_DATASET_PATH.value: "Dataset path is invalid.",
    EvaluationJobExitCode.INVALID_EVALUATION_ID.value: "Evaluation ID was not found in the Model Catalog.",
    EvaluationJobExitCode.INVALID_TARGET_EVALUATION_ID.value: "Target evaluation ID was not found in the Model Deployment.",
    EvaluationJobExitCode.INVALID_EVALUATION_CONFIG_VALIDATION.value: "Validation errors in the evaluation config.",
    EvaluationJobExitCode.OUTPUT_DIR_NOT_FOUND.value: "Destination folder does not exist or cannot be used for writing, verify the folder's existence and permissions.",
    EvaluationJobExitCode.INVALID_INPUT_DATASET.value: "Input dataset is in an invalid format, ensure the dataset is in jsonl format and that includes the required columns: 'prompt', 'completion' (optional 'category').",
    EvaluationJobExitCode.INPUT_DATA_NOT_FOUND.value: "Input data file does not exist or cannot be use for reading, verify the file's existence and permissions.",
    EvaluationJobExitCode.EVALUATION_ID_NOT_FOUND.value: "Evaluation ID does not match any resource in the Model Catalog, or access may be blocked by policies.",
    EvaluationJobExitCode.EVALUATION_ALREADY_PERFORMED.value: "Evaluation already has an attached artifact, indicating that the evaluation has already been performed.",
    EvaluationJobExitCode.EVALUATION_TARGET_NOT_FOUND.value: "Target evaluation ID does not match any resources in Model Deployment.",
    EvaluationJobExitCode.NO_SUCCESS_INFERENCE_RESULT.value: "Inference process completed without producing expected outcome, verify the model parameters and config.",
    EvaluationJobExitCode.COMPUTE_EVALUATION_ERROR.value: "Evaluation process encountered an issue while calculating metrics.",
    EvaluationJobExitCode.EVALUATION_REPORT_ERROR.value: "Failed to save the evaluation report due to an error. Ensure the evaluation model is currently active and the specified path for the output report is valid and accessible. Verify these conditions and reinitiate the evaluation process.",
    EvaluationJobExitCode.MODEL_INFERENCE_WRONG_RESPONSE_FORMAT.value: "Evaluation encountered unsupported, or unexpected model output, verify the target evaluation model is compatible and produces the correct format.",
    EvaluationJobExitCode.UNSUPPORTED_METRICS.value: "None of the provided metrics are supported by the framework.",
    EvaluationJobExitCode.METRIC_CALCULATION_FAILURE.value: "All attempted metric calculations were unsuccessful. Please review the metric configurations and input data.",
}


class Resource(Enum):
    JOB = "jobs"
    MODEL = "models"
    MODEL_DEPLOYMENT = "modeldeployments"
    MODEL_VERSION_SET = "model-version-sets"


class DataScienceResource(Enum):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"


class EvaluationCustomMetadata(Enum):
    EVALUATION_SOURCE = "evaluation_source"
    EVALUATION_JOB_ID = "evaluation_job_id"
    EVALUATION_JOB_RUN_ID = "evaluation_job_run_id"
    EVALUATION_OUTPUT_PATH = "evaluation_output_path"
    EVALUATION_SOURCE_NAME = "evaluation_source_name"
    EVALUATION_ERROR = "aqua_evaluate_error"


class EvaluationModelTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"


class EvaluationJobTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"
    EVALUATION_MODEL_ID = "evaluation_model_id"


class EvaluationUploadStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    url: str = ""


@dataclass(repr=False)
class AquaEvalReport(DataClassSerializable):
    evaluation_id: str = ""
    content: str = ""


@dataclass(repr=False)
class ModelParams(DataClassSerializable):
    max_tokens: str = ""
    top_p: str = ""
    top_k: str = ""
    temperature: str = ""
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)


@dataclass(repr=False)
class AquaEvalParams(ModelParams, DataClassSerializable):
    shape: str = ""
    dataset_path: str = ""
    report_path: str = ""


@dataclass(repr=False)
class AquaEvalMetric(DataClassSerializable):
    key: str
    name: str
    description: str = ""


@dataclass(repr=False)
class AquaEvalMetricSummary(DataClassSerializable):
    metric: str = ""
    score: str = ""
    grade: str = ""


@dataclass(repr=False)
class AquaEvalMetrics(DataClassSerializable):
    id: str
    report: str
    metric_results: List[AquaEvalMetric] = field(default_factory=list)
    metric_summary_result: List[AquaEvalMetricSummary] = field(default_factory=list)


@dataclass(repr=False)
class AquaEvaluationCommands(DataClassSerializable):
    evaluation_id: str
    evaluation_target_id: str
    input_data: dict
    metrics: list
    output_dir: str
    params: dict


@dataclass(repr=False)
class AquaEvaluationSummary(DataClassSerializable):
    """Represents a summary of Aqua evalution."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    tags: dict
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    parameters: AquaEvalParams = field(default_factory=AquaEvalParams)


@dataclass(repr=False)
class AquaEvaluationDetail(AquaEvaluationSummary, DataClassSerializable):
    """Represents a details of Aqua evalution."""

    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    introspection: dict = field(default_factory=dict)


class RqsAdditionalDetails:
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"


class EvaluationConfig:
    PARAMS = "model_params"


@dataclass(repr=False)
class CreateAquaEvaluationDetails(DataClassSerializable):
    """Dataclass to create aqua model evaluation.

    Fields
    ------
    evaluation_source_id: str
        The evaluation source id. Must be either model or model deployment ocid.
    evaluation_name: str
        The name for evaluation.
    dataset_path: str
        The dataset path for the evaluation. Could be either a local path from notebook session
        or an object storage path.
    report_path: str
        The report path for the evaluation. Must be an object storage path.
    model_parameters: dict
        The parameters for the evaluation.
    shape_name: str
        The shape name for the evaluation job infrastructure.
    memory_in_gbs: float
        The memory in gbs for the shape selected.
    ocpus: float
        The ocpu count for the shape selected.
    block_storage_size: int
        The storage for the evaluation job infrastructure.
    compartment_id: (str, optional). Defaults to `None`.
        The compartment id for the evaluation.
    project_id: (str, optional). Defaults to `None`.
        The project id for the evaluation.
    evaluation_description: (str, optional). Defaults to `None`.
        The description for evaluation
    experiment_id: (str, optional). Defaults to `None`.
        The evaluation model version set id. If provided,
        evaluation model will be associated with it.
    experiment_name: (str, optional). Defaults to `None`.
        The evaluation model version set name. If provided,
        the model version set with the same name will be used if exists,
        otherwise a new model version set will be created with the name.
    experiment_description: (str, optional). Defaults to `None`.
        The description for the evaluation model version set.
    log_group_id: (str, optional). Defaults to `None`.
        The log group id for the evaluation job infrastructure.
    log_id: (str, optional). Defaults to `None`.
        The log id for the evaluation job infrastructure.
    metrics: (list, optional). Defaults to `None`.
        The metrics for the evaluation.
    """

    evaluation_source_id: str
    evaluation_name: str
    dataset_path: str
    report_path: str
    model_parameters: dict
    shape_name: str
    block_storage_size: int
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    evaluation_description: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    memory_in_gbs: Optional[float] = None
    ocpus: Optional[float] = None
    log_group_id: Optional[str] = None
    log_id: Optional[str] = None
    metrics: Optional[List] = None


class AquaEvaluationApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua evaluations within the
    Oracle Cloud Infrastructure Data Science service, serving as an interface
    for managing model evalutions.


    Methods
    -------
    create(evaluation_source_id, evaluation_name, ...) -> AquaEvaluationSummary:
        Creates Aqua evaluation for resource.
    get(model_id: str) -> AquaEvaluationSummary:
        Retrieves details of an Aqua evaluation by its unique identifier.
    list(compartment_id: str = None, project_id: str = None, **kwargs) -> List[AquaEvaluationSummary]:
        Lists all Aqua evaluation within a specified compartment and/or project.

    Note:
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    _report_cache = TTLCache(maxsize=10, ttl=timedelta(hours=5), timer=datetime.now)
    _metrics_cache = TTLCache(maxsize=10, ttl=timedelta(hours=5), timer=datetime.now)
    _eval_cache = TTLCache(maxsize=200, ttl=timedelta(hours=10), timer=datetime.now)
    _cache_lock = Lock()

    @telemetry(entry_point="plugin=evaluation&action=create", name="aqua")
    def create(
        self,
        create_aqua_evaluation_details: CreateAquaEvaluationDetails,
        **kwargs,
    ) -> "AquaEvaluationSummary":
        """Creates Aqua evaluation for resource.

        Parameters
        ----------
        create_aqua_evaluation_details: CreateAquaEvaluationDetails
            The CreateAquaEvaluationDetails data class which contains all
            required and optional fields to create the aqua evaluation.
        kwargs:
            The kwargs for the evaluation.

        Returns
        -------
        AquaEvaluationSummary:
            The instance of AquaEvaluationSummary.
        """
        if not is_valid_ocid(create_aqua_evaluation_details.evaluation_source_id):
            raise AquaValueError(
                f"Invalid evaluation source {create_aqua_evaluation_details.evaluation_source_id}. "
                "Specify either a model or model deployment id."
            )

        evaluation_source = None
        if (
            DataScienceResource.MODEL_DEPLOYMENT.value
            in create_aqua_evaluation_details.evaluation_source_id
        ):
            evaluation_source = ModelDeployment.from_id(
                create_aqua_evaluation_details.evaluation_source_id
            )
        elif (
            DataScienceResource.MODEL.value
            in create_aqua_evaluation_details.evaluation_source_id
        ):
            evaluation_source = DataScienceModel.from_id(
                create_aqua_evaluation_details.evaluation_source_id
            )
        else:
            raise AquaValueError(
                f"Invalid evaluation source {create_aqua_evaluation_details.evaluation_source_id}. "
                "Specify either a model or model deployment id."
            )

        if not ObjectStorageDetails.is_oci_path(
            create_aqua_evaluation_details.report_path
        ):
            raise AquaValueError(
                "Evaluation report path must be an object storage path."
            )

        evaluation_dataset_path = create_aqua_evaluation_details.dataset_path
        if not ObjectStorageDetails.is_oci_path(evaluation_dataset_path):
            # format: oci://<bucket>@<namespace>/<prefix>/<dataset_file_name>
            dataset_file = os.path.basename(evaluation_dataset_path)
            dst_uri = f"{create_aqua_evaluation_details.report_path.rstrip('/')}/{dataset_file}"
            try:
                upload_local_to_os(
                    src_uri=evaluation_dataset_path,
                    dst_uri=dst_uri,
                    auth=default_signer(),
                    force_overwrite=False,
                )
            except FileExistsError:
                raise AquaFileExistsError(
                    f"Dataset {dataset_file} already exists in {create_aqua_evaluation_details.report_path}. "
                    "Please use a new dataset file name or report path."
                )
            logger.debug(
                f"Uploaded local file {evaluation_dataset_path} to object storage {dst_uri}."
            )
            # tracks the size of dataset uploaded by user to the destination.
            self.telemetry.record_event_async(
                category="aqua/evaluation/upload",
                action="size",
                detail=os.path.getsize(os.path.expanduser(evaluation_dataset_path)),
            )
            evaluation_dataset_path = dst_uri

        evaluation_model_parameters = None
        try:
            evaluation_model_parameters = AquaEvalParams(
                shape=create_aqua_evaluation_details.shape_name,
                dataset_path=evaluation_dataset_path,
                report_path=create_aqua_evaluation_details.report_path,
                **create_aqua_evaluation_details.model_parameters,
            )
        except:
            raise AquaValueError(
                "Invalid model parameters. Model parameters should "
                f"be a dictionary with keys: {', '.join(list(ModelParams.__annotations__.keys()))}."
            )

        target_compartment = (
            create_aqua_evaluation_details.compartment_id or COMPARTMENT_OCID
        )
        target_project = create_aqua_evaluation_details.project_id or PROJECT_OCID

        experiment_model_version_set_id = create_aqua_evaluation_details.experiment_id
        experiment_model_version_set_name = (
            create_aqua_evaluation_details.experiment_name
        )

        if (
            not experiment_model_version_set_id
            and not experiment_model_version_set_name
        ):
            raise AquaValueError(
                "Either experiment id or experiment name must be provided."
            )

        if not experiment_model_version_set_id:
            try:
                model_version_set = ModelVersionSet.from_name(
                    name=experiment_model_version_set_name,
                    compartment_id=target_compartment,
                )
                if not utils._is_valid_mvs(
                    model_version_set, Tags.AQUA_EVALUATION.value
                ):
                    raise AquaValueError(
                        f"Invalid experiment name. Please provide an experiment with `{Tags.AQUA_EVALUATION.value}` in tags."
                    )
            except:
                logger.debug(
                    f"Model version set {experiment_model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )

                evaluation_mvs_freeform_tags = {
                    Tags.AQUA_EVALUATION.value: Tags.AQUA_EVALUATION.value,
                }

                model_version_set = (
                    ModelVersionSet()
                    .with_compartment_id(target_compartment)
                    .with_project_id(target_project)
                    .with_name(experiment_model_version_set_name)
                    .with_description(
                        create_aqua_evaluation_details.experiment_description
                    )
                    .with_freeform_tags(**evaluation_mvs_freeform_tags)
                    # TODO: decide what parameters will be needed
                    .create(**kwargs)
                )
                logger.debug(
                    f"Successfully created model version set {experiment_model_version_set_name} with id {model_version_set.id}."
                )
            experiment_model_version_set_id = model_version_set.id
        else:
            model_version_set = ModelVersionSet.from_id(experiment_model_version_set_id)
            if not utils._is_valid_mvs(model_version_set, Tags.AQUA_EVALUATION.value):
                raise AquaValueError(
                    f"Invalid experiment id. Please provide an experiment with `{Tags.AQUA_EVALUATION.value}` in tags."
                )
            experiment_model_version_set_name = model_version_set.name

        evaluation_model_custom_metadata = ModelCustomMetadata()
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_SOURCE.value,
            value=create_aqua_evaluation_details.evaluation_source_id,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_OUTPUT_PATH.value,
            value=create_aqua_evaluation_details.report_path,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_SOURCE_NAME.value,
            value=evaluation_source.display_name,
        )

        evaluation_model_taxonomy_metadata = ModelTaxonomyMetadata()
        evaluation_model_taxonomy_metadata[
            MetadataTaxonomyKeys.HYPERPARAMETERS
        ].value = {
            "model_params": {
                key: value for key, value in asdict(evaluation_model_parameters).items()
            }
        }

        evaluation_model = (
            DataScienceModel()
            .with_compartment_id(target_compartment)
            .with_project_id(target_project)
            .with_display_name(create_aqua_evaluation_details.evaluation_name)
            .with_description(create_aqua_evaluation_details.evaluation_description)
            .with_model_version_set_id(experiment_model_version_set_id)
            .with_custom_metadata_list(evaluation_model_custom_metadata)
            .with_defined_metadata_list(evaluation_model_taxonomy_metadata)
            .with_provenance_metadata(ModelProvenanceMetadata(training_id=UNKNOWN))
            # TODO uncomment this once the evaluation container will get the updated version of the ADS
            # .with_input_schema(create_aqua_evaluation_details.to_dict())
            # TODO: decide what parameters will be needed
            .create(
                remove_existing_artifact=False,  # TODO: added here for the purpose of demo and will revisit later
                **kwargs,
            )
        )
        logger.debug(
            f"Successfully created evaluation model {evaluation_model.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        # TODO: validate metrics if it's provided

        evaluation_job_freeform_tags = {
            EvaluationJobTags.AQUA_EVALUATION.value: EvaluationJobTags.AQUA_EVALUATION.value,
            EvaluationJobTags.EVALUATION_MODEL_ID.value: evaluation_model.id,
        }

        evaluation_job = Job(name=evaluation_model.display_name).with_infrastructure(
            DataScienceJob()
            .with_log_group_id(create_aqua_evaluation_details.log_group_id)
            .with_log_id(create_aqua_evaluation_details.log_id)
            .with_compartment_id(target_compartment)
            .with_project_id(target_project)
            .with_shape_name(create_aqua_evaluation_details.shape_name)
            .with_block_storage_size(create_aqua_evaluation_details.block_storage_size)
            .with_freeform_tag(**evaluation_job_freeform_tags)
        )
        if (
            create_aqua_evaluation_details.memory_in_gbs
            and create_aqua_evaluation_details.ocpus
        ):
            evaluation_job.infrastructure.with_shape_config_details(
                memory_in_gbs=create_aqua_evaluation_details.memory_in_gbs,
                ocpus=create_aqua_evaluation_details.ocpus,
            )
        if AQUA_JOB_SUBNET_ID:
            evaluation_job.infrastructure.with_subnet_id(AQUA_JOB_SUBNET_ID)
        else:
            if NB_SESSION_IDENTIFIER in os.environ:
                # apply default subnet id for job by setting ME_STANDALONE
                # so as to avoid using the notebook session's networking when running on it
                # https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/infra_and_runtime.html#networking
                evaluation_job.infrastructure.with_job_infrastructure_type(
                    JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING
                )

        container_image = self._get_evaluation_container(
            create_aqua_evaluation_details.evaluation_source_id
        )

        evaluation_job.with_runtime(
            self._build_evaluation_runtime(
                evaluation_id=evaluation_model.id,
                evaluation_source_id=(
                    create_aqua_evaluation_details.evaluation_source_id
                ),
                container_image=container_image,
                dataset_path=evaluation_dataset_path,
                report_path=create_aqua_evaluation_details.report_path,
                model_parameters=create_aqua_evaluation_details.model_parameters,
                metrics=create_aqua_evaluation_details.metrics,
            )
        ).create(
            **kwargs
        )  ## TODO: decide what parameters will be needed
        logger.debug(
            f"Successfully created evaluation job {evaluation_job.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_job_run = evaluation_job.run(
            name=evaluation_model.display_name,
            freeform_tags=evaluation_job_freeform_tags,
            wait=False,
        )
        logger.debug(
            f"Successfully created evaluation job run {evaluation_job_run.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_JOB_ID.value,
            value=evaluation_job.id,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID.value,
            value=evaluation_job_run.id,
        )
        updated_custom_metadata_list = [
            Metadata(**metadata)
            for metadata in evaluation_model_custom_metadata.to_dict()["data"]
        ]

        self.ds_client.update_model(
            model_id=evaluation_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list,
                freeform_tags={
                    EvaluationModelTags.AQUA_EVALUATION.value: EvaluationModelTags.AQUA_EVALUATION.value,
                },
            ),
        )

        self.ds_client.update_model_provenance(
            model_id=evaluation_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=evaluation_job_run.id
            ),
        )

        # tracks unique evaluation that were created for the given evaluation source
        self.telemetry.record_event_async(
            category="aqua/evaluation",
            action="create",
            detail=evaluation_source.display_name,
        )

        return AquaEvaluationSummary(
            id=evaluation_model.id,
            name=evaluation_model.display_name,
            console_url=get_console_link(
                resource=Resource.MODEL.value,
                ocid=evaluation_model.id,
                region=self.region,
            ),
            time_created=str(evaluation_model.dsc_model.time_created),
            lifecycle_state=evaluation_job_run.lifecycle_state or UNKNOWN,
            lifecycle_details=evaluation_job_run.lifecycle_details or UNKNOWN,
            experiment=AquaResourceIdentifier(
                id=experiment_model_version_set_id,
                name=experiment_model_version_set_name,
                url=get_console_link(
                    resource=Resource.MODEL_VERSION_SET.value,
                    ocid=experiment_model_version_set_id,
                    region=self.region,
                ),
            ),
            source=AquaResourceIdentifier(
                id=create_aqua_evaluation_details.evaluation_source_id,
                name=evaluation_source.display_name,
                url=get_console_link(
                    resource=(
                        Resource.MODEL_DEPLOYMENT.value
                        if DataScienceResource.MODEL_DEPLOYMENT.value
                        in create_aqua_evaluation_details.evaluation_source_id
                        else Resource.MODEL.value
                    ),
                    ocid=create_aqua_evaluation_details.evaluation_source_id,
                    region=self.region,
                ),
            ),
            job=AquaResourceIdentifier(
                id=evaluation_job.id,
                name=evaluation_job.name,
                url=get_console_link(
                    resource=Resource.JOB.value,
                    ocid=evaluation_job.id,
                    region=self.region,
                ),
            ),
            tags=dict(
                aqua_evaluation=EvaluationModelTags.AQUA_EVALUATION.value,
                evaluation_job_id=evaluation_job.id,
                evaluation_source=create_aqua_evaluation_details.evaluation_source_id,
                evaluation_experiment_id=experiment_model_version_set_id,
            ),
            parameters=AquaEvalParams(),
        )

    def _build_evaluation_runtime(
        self,
        evaluation_id: str,
        evaluation_source_id: str,
        container_image: str,
        dataset_path: str,
        report_path: str,
        model_parameters: dict,
        metrics: List = None,
    ) -> Runtime:
        """Builds evaluation runtime for Job."""
        # TODO the image name needs to be extracted from the mapping index.json file.
        runtime = (
            ContainerRuntime()
            .with_image(container_image)
            .with_environment_variable(
                **{
                    "AIP_SMC_EVALUATION_ARGUMENTS": json.dumps(
                        asdict(
                            self._build_launch_cmd(
                                evaluation_id=evaluation_id,
                                evaluation_source_id=evaluation_source_id,
                                dataset_path=dataset_path,
                                report_path=report_path,
                                model_parameters=model_parameters,
                                metrics=metrics,
                            )
                        )
                    ),
                    "CONDA_BUCKET_NS": CONDA_BUCKET_NS,
                },
            )
        )

        return runtime

    @staticmethod
    def _get_evaluation_container(source_id: str) -> str:
        # todo: use the source, identify if it is a model or a deployment. If latter, then fetch the base model id
        #   from the deployment object, and call ds_client.get_model() to get model details. Use custom metadata to
        #   get the container_type_key. Pass this key as container_type to get_container_image method.

        # fetch image name from config
        container_image = get_container_image(
            container_type="odsc-llm-evaluate",
        )
        logger.info(f"Aqua Image used for evaluating {source_id} :{container_image}")
        return container_image

    def _build_launch_cmd(
        self,
        evaluation_id: str,
        evaluation_source_id: str,
        dataset_path: str,
        report_path: str,
        model_parameters: dict,
        metrics: List = None,
    ):
        return AquaEvaluationCommands(
            evaluation_id=evaluation_id,
            evaluation_target_id=evaluation_source_id,
            input_data={
                "columns": {
                    "prompt": "prompt",
                    "completion": "completion",
                    "category": "category",
                },
                "format": Path(dataset_path).suffix,
                "url": dataset_path,
            },
            metrics=metrics,
            output_dir=report_path,
            params=model_parameters,
        )

    @telemetry(entry_point="plugin=evaluation&action=get", name="aqua")
    def get(self, eval_id) -> AquaEvaluationDetail:
        """Gets the information of an Aqua evalution.

        Parameters
        ----------
        eval_id: str
            The model OCID.

        Returns
        -------
        AquaEvaluationDetail:
            The instance of AquaEvaluationDetail.
        """
        logger.info(f"Fetching evaluation: {eval_id} details ...")

        resource = utils.query_resource(eval_id)
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        if not resource:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )
        jobrun_id = model_provenance.training_id
        job_run_details = self._fetch_jobrun(
            resource, use_rqs=False, jobrun_id=jobrun_id
        )

        try:
            log_id = job_run_details.log_details.log_id
        except Exception as e:
            logger.debug(f"Failed to get associated log. {str(e)}")
            log_id = ""

        try:
            loggroup_id = job_run_details.log_details.log_group_id
        except Exception as e:
            logger.debug(f"Failed to get associated loggroup. {str(e)}")
            loggroup_id = ""

        loggroup_url = get_log_links(region=self.region, log_group_id=loggroup_id)
        log_url = get_log_links(
            region=self.region,
            log_group_id=loggroup_id,
            log_id=log_id,
            compartment_id=job_run_details.compartment_id,
            source_id=jobrun_id
        ) if job_run_details else ""

        log_name = None
        loggroup_name = None

        if log_id:
            try:
                log = utils.query_resource(log_id, return_all=False)
                log_name = log.display_name if log else ""
            except:
                pass

        if loggroup_id:
            try:
                loggroup = utils.query_resource(loggroup_id, return_all=False)
                loggroup_name = loggroup.display_name if loggroup else ""
            except:
                pass

        try:
            introspection = json.loads(
                self._get_attribute_from_model_metadata(resource, "ArtifactTestResults")
            )
        except:
            introspection = {}

        summary = AquaEvaluationDetail(
            **self._process(resource),
            **self._get_status(model=resource, jobrun=job_run_details),
            job=self._build_job_identifier(
                job_run_details=job_run_details,
            ),
            log_group=AquaResourceIdentifier(loggroup_id, loggroup_name, loggroup_url),
            log=AquaResourceIdentifier(log_id, log_name, log_url),
            introspection=introspection,
        )
        summary.parameters.shape = (
            job_run_details.job_infrastructure_configuration_details.shape_name
        )
        return summary

    @telemetry(entry_point="plugin=evaluation&action=list", name="aqua")
    def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List[AquaEvaluationSummary]:
        """List Aqua evaluations in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments.

        Returns
        -------
        List[AquaEvaluationSummary]:
            The list of the `ads.aqua.evalution.AquaEvaluationSummary`.
        """
        logger.info(f"Fetching evaluations from compartment {compartment_id}.")
        models = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="datasciencemodel",
            tag_list=[EvaluationModelTags.AQUA_EVALUATION.value],
        )
        logger.info(f"Fetched {len(models)} evaluations.")

        # TODO: add filter based on project_id if needed.

        mapping = self._prefetch_resources(compartment_id)

        evaluations = []
        async_tasks = []
        for model in models:

            if model.identifier in self._eval_cache.keys():
                logger.debug(f"Retrieving evaluation {model.identifier} from cache.")
                evaluations.append(self._eval_cache.get(model.identifier))

            else:
                jobrun_id = self._get_attribute_from_model_metadata(
                    model, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID.value
                )
                job_run = mapping.get(jobrun_id)

                if not job_run:
                    async_tasks.append((model, jobrun_id))
                else:
                    evaluations.append(self._process_evaluation_summary(model, job_run))

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_model = {
                executor.submit(
                    self._fetch_jobrun, model, use_rqs=True, jobrun_id=jobrun_id
                ): model
                for model, jobrun_id in async_tasks
            }
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    jobrun = future.result()
                    evaluations.append(
                        self._process_evaluation_summary(model=model, jobrun=jobrun)
                    )
                except Exception as exc:
                    logger.error(
                        f"Processing evaluation: {model.identifier} generated an exception: {exc}"
                    )
                    evaluations.append(
                        self._process_evaluation_summary(model=model, jobrun=None)
                    )

        # tracks number of times deployment listing was called
        self.telemetry.record_event_async(category="aqua/evaluation", action="list")

        return evaluations

    def _process_evaluation_summary(
        self,
        model: oci.resource_search.models.ResourceSummary,
        jobrun: oci.resource_search.models.ResourceSummary = None,
    ) -> AquaEvaluationSummary:
        """Builds AquaEvaluationSummary from model and jobrun."""

        evaluation_summary = AquaEvaluationSummary(
            **self._process(model),
            **self._get_status(
                model=model,
                jobrun=jobrun,
            ),
            job=self._build_job_identifier(
                job_run_details=jobrun,
            ),
        )

        # Add evaluation in terminal state into cache
        if evaluation_summary.lifecycle_state in EVAL_TERMINATION_STATE:
            self._eval_cache.__setitem__(key=model.identifier, value=evaluation_summary)

        return evaluation_summary

    def _if_eval_artifact_exist(
        self, model: oci.resource_search.models.ResourceSummary
    ) -> bool:
        """Checks if the evaluation artifact exists."""
        try:
            response = self.ds_client.head_model_artifact(model_id=model.identifier)
            return True if response.status == 200 else False
        except oci.exceptions.ServiceError as ex:
            if ex.status == 404:
                logger.info("Evaluation artifact not found.")
                return False

    @telemetry(entry_point="plugin=evaluation&action=get_status", name="aqua")
    def get_status(self, eval_id: str) -> dict:
        """Gets evaluation's current status.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        dict
        """
        eval = utils.query_resource(eval_id)

        # TODO: add job_run_id as input param to skip the query below
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        if not eval:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )
        jobrun_id = model_provenance.training_id
        job_run_details = self._fetch_jobrun(eval, use_rqs=False, jobrun_id=jobrun_id)

        try:
            log_id = job_run_details.log_details.log_id
        except Exception as e:
            logger.debug(f"Failed to get associated log. {str(e)}")
            log_id = ""

        try:
            loggroup_id = job_run_details.log_details.log_group_id
        except Exception as e:
            logger.debug(f"Failed to get associated log. {str(e)}")
            loggroup_id = ""

        loggroup_url = get_log_links(region=self.region, log_group_id=loggroup_id)
        log_url = get_log_links(
            region=self.region,
            log_group_id=loggroup_id,
            log_id=log_id,
            compartment_id=job_run_details.compartment_id,
            source_id=jobrun_id
        ) if job_run_details else ""

        return dict(
            id=eval_id,
            **self._get_status(
                model=eval,
                jobrun=job_run_details,
            ),
            log_id=log_id,
            log_url=log_url,
            loggroup_id=loggroup_id,
            loggroup_url=loggroup_url,
        )

    def get_supported_metrics(self) -> dict:
        """Gets a list of supported metrics for evaluation."""
        # TODO: implement it when starting to support more metrics.
        return [
            {
                "use_case": ["text_generation"],
                "key": "bertscore",
                "name": "BERT Score",
                "description": (
                    "BERT Score is a metric for evaluating the quality of text "
                    "generation models, such as machine translation or summarization. "
                    "It utilizes pre-trained BERT contextual embeddings for both the "
                    "generated and reference texts, and then calculates the cosine "
                    "similarity between these embeddings."
                ),
                "args": {},
            },
            {
                "use_case": ["text_generation"],
                "key": "rouge",
                "name": "ROUGE Score",
                "description": (
                    "ROUGE scores compare a candidate document to a collection of "
                    "reference documents to evaluate the similarity between them. "
                    "The metrics range from 0 to 1, with higher scores indicating "
                    "greater similarity. ROUGE is more suitable for models that don't "
                    "include paraphrasing and do not generate new text units that don't "
                    "appear in the references."
                ),
                "args": {},
            },
        ]

    @telemetry(entry_point="plugin=evaluation&action=load_metrics", name="aqua")
    def load_metrics(self, eval_id: str) -> AquaEvalMetrics:
        """Loads evalution metrics markdown from artifacts.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        AquaEvalMetrics:
            An instance of AquaEvalMetrics.
        """
        if eval_id in self._metrics_cache.keys():
            logger.info(f"Returning metrics from cache.")
            eval_metrics = self._metrics_cache.get(eval_id)
            if len(eval_metrics.report) > 0:
                return eval_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading evaluation artifact: {eval_id}.")
            DataScienceModel.from_id(eval_id).download_artifact(
                temp_dir,
                auth=self._auth,
            )

            files_in_artifact = get_files(temp_dir)
            report_content = self._read_from_artifact(
                temp_dir, files_in_artifact, utils.EVALUATION_REPORT_MD
            )
            try:
                report = json.loads(
                    self._read_from_artifact(
                        temp_dir, files_in_artifact, utils.EVALUATION_REPORT_JSON
                    )
                )
            except Exception as e:
                logger.debug(
                    "Failed to load `report.json` from evaluation artifact" f"{str(e)}"
                )
                report = {}

        # TODO: after finalizing the format of report.json, move the constant to class
        eval_metrics = AquaEvalMetrics(
            id=eval_id,
            report=base64.b64encode(report_content).decode(),
            metric_results=[
                AquaEvalMetric(
                    key=metric_key,
                    name=metadata.get("name", utils.UNKNOWN),
                    description=metadata.get("description", utils.UNKNOWN),
                )
                for metric_key, metadata in report.get("metric_results", {}).items()
            ],
            metric_summary_result=[
                AquaEvalMetricSummary(**m)
                for m in report.get("metric_summary_result", [{}])
            ],
        )

        if report_content:
            self._metrics_cache.__setitem__(key=eval_id, value=eval_metrics)

        return eval_metrics

    def _read_from_artifact(self, artifact_dir, files, target):
        """Reads target file from artifacts.

        Parameters
        ----------
        artifact_dir: str
            Path of the artifact.
        files: list
            List of files name in artifacts.
        target: str
            Target file name.

        Return
        ------
        bytes
        """
        content = None
        for f in files:
            if os.path.basename(f) == target:
                logger.info(f"Reading {f}...")
                with open(os.path.join(artifact_dir, f), "rb") as f:
                    content = f.read()
                break

        if not content:
            raise AquaFileNotFoundError(
                "Related Resource Not Authorized Or Not Found:"
                f"Missing `{target}` in evaluation artifact."
            )
        return content

    @telemetry(entry_point="plugin=evaluation&action=download_report", name="aqua")
    def download_report(self, eval_id) -> AquaEvalReport:
        """Downloads HTML report from model artifact.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        AquaEvalReport:
            An instance of AquaEvalReport.

        Raises
        ------
        AquaFileNotFoundError:
            When missing `report.html` in evaluation artifact.
        """
        if eval_id in self._report_cache.keys():
            logger.info(f"Returning report from cache.")
            report = self._report_cache.get(eval_id)
            if report.content:
                return report

        with tempfile.TemporaryDirectory() as temp_dir:
            DataScienceModel.from_id(eval_id).download_artifact(
                temp_dir,
                auth=self._auth,
            )
            content = self._read_from_artifact(
                temp_dir, get_files(temp_dir), utils.EVALUATION_REPORT
            )

        report = AquaEvalReport(
            evaluation_id=eval_id, content=base64.b64encode(content).decode()
        )

        self._report_cache.__setitem__(key=eval_id, value=report)

        return report

    @telemetry(entry_point="plugin=evaluation&action=cancel", name="aqua")
    def cancel(self, eval_id) -> dict:
        """Cancels the job run for the given evaluation id.
        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
            dict containing id, status and time_accepted

        Raises
        ------
        AquaRuntimeError:
            if a model doesn't exist for the given eval_id
        AquaMissingKeyError:
            if training_id is missing the job run id
        """
        model = DataScienceModel.from_id(eval_id)
        if not model:
            raise AquaRuntimeError(
                f"Failed to get evaluation details for model {eval_id}"
            )
        job_run_id = model.provenance_metadata.training_id
        if not job_run_id:
            raise AquaMissingKeyError(
                "Model provenance is missing job run training_id key"
            )

        status = dict(id=eval_id, status=UNKNOWN, time_accepted="")
        run = DataScienceJobRun.from_ocid(job_run_id)
        if run.lifecycle_state in [
            DataScienceJobRun.LIFECYCLE_STATE_ACCEPTED,
            DataScienceJobRun.LIFECYCLE_STATE_IN_PROGRESS,
            DataScienceJobRun.LIFECYCLE_STATE_NEEDS_ATTENTION,
        ]:
            self._cancel_job_run(run, model)
            status = dict(
                id=eval_id,
                lifecycle_state="CANCELING",
                time_accepted=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z"),
            )
        return status

    @staticmethod
    @fire_and_forget
    def _cancel_job_run(run, model):
        try:
            run.cancel()
            logger.info(f"Canceling Job Run: {run.id} for evaluation {model.id}")
        except oci.exceptions.ServiceError as ex:
            logger.error(
                f"Exception occurred while canceling job run: {run.id} for evaluation {model.id}. "
                f"Exception message: {ex}"
            )

    @telemetry(entry_point="plugin=evaluation&action=delete", name="aqua")
    def delete(self, eval_id):
        """Deletes the job and the associated model for the given evaluation id.
        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
            dict containing id, status and time_accepted

        Raises
        ------
        AquaRuntimeError:
            if a model doesn't exist for the given eval_id
        AquaMissingKeyError:
            if training_id is missing the job run id
        """

        model = DataScienceModel.from_id(eval_id)
        if not model:
            raise AquaRuntimeError(
                f"Failed to get evaluation details for model {eval_id}"
            )

        try:
            job_id = model.custom_metadata_list.get(
                EvaluationCustomMetadata.EVALUATION_JOB_ID.value
            ).value
        except ValueError:
            raise AquaMissingKeyError(
                f"Custom metadata is missing {EvaluationCustomMetadata.EVALUATION_JOB_ID.value} key"
            )

        job = DataScienceJob.from_id(job_id)

        self._delete_job_and_model(job, model)

        status = dict(
            id=eval_id,
            lifecycle_state="DELETING",
            time_accepted=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z"),
        )
        return status

    @staticmethod
    @fire_and_forget
    def _delete_job_and_model(job, model):
        try:
            job.dsc_job.delete(force_delete=True)
            logger.info(f"Deleting Job: {job.job_id} for evaluation {model.id}")

            model.delete()
            logger.info(f"Deleting evaluation: {model.id}")
        except oci.exceptions.ServiceError as ex:
            logger.error(
                f"Exception occurred while deleting job: {job.job_id} for evaluation {model.id}. "
                f"Exception message: {ex}"
            )

    def load_evaluation_config(self, eval_id):
        # TODO
        return {
            "model_params": {
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 50,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "stop": [],
            },
            "shape": {
                "VM.Standard.E3.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
                "VM.Standard.E3.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
                "VM.Standard.E4.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
                "VM.Standard3.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
                "VM.Optimized3.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
                "VM.Standard.A1.Flex": {
                    "ocpu": 2,
                    "memory_in_gbs": 64,
                    "block_storage_size": 100,
                },
            },
            "default": {
                "ocpu": 2,
                "memory_in_gbs": 64,
                "block_storage_size": 100,
            },
        }

    def _get_attribute_from_model_metadata(
        self,
        model: oci.resource_search.models.ResourceSummary,
        target_attribute: str,
    ) -> str:
        try:
            return self._extract_metadata(
                model.additional_details.get(RqsAdditionalDetails.METADATA),
                target_attribute,
            )
        except:
            logger.debug(
                f"Missing `{target_attribute}` in custom metadata of the evaluation."
                f"Evaluation id: {model.identifier} "
            )
            return ""

    def _extract_metadata(self, metadata_list: List[Dict], key: str) -> Any:
        for metadata in metadata_list:
            if metadata.get("key") == key:
                return metadata.get("value")
        raise AquaMissingKeyError(
            f"Missing `{key}` in custom metadata of the evaluation."
        )

    def _get_source(
        self,
        evaluation: oci.resource_search.models.ResourceSummary,
        resources_mapping: dict = {},
    ) -> tuple:
        """Returns ocid and name of the model has been evaluated."""
        source_id = self._get_attribute_from_model_metadata(
            evaluation,
            EvaluationCustomMetadata.EVALUATION_SOURCE.value,
        )

        try:
            source = resources_mapping.get(source_id)
            source_name = (
                source.display_name
                if source
                else self._get_attribute_from_model_metadata(
                    evaluation, EvaluationCustomMetadata.EVALUATION_SOURCE_NAME.value
                )
            )

            if not source_name:
                resource_type = utils.get_resource_type(source_id)

                # TODO: adjust resource principal mapping
                if resource_type == "datasciencemodel":
                    source_name = self.ds_client.get_model(source_id).data.display_name
                elif resource_type == "datasciencemodeldeployment":
                    source_name = self.ds_client.get_model_deployment(
                        source_id
                    ).data.display_name
                else:
                    raise AquaRuntimeError(
                        f"Not supported source type: {resource_type}"
                    )
        except Exception as e:
            logger.debug(
                f"Failed to retrieve source information for evaluation {evaluation.identifier}."
            )
            source_name = ""

        return (source_id, source_name)

    def _get_experiment_info(
        self, model: oci.resource_search.models.ResourceSummary
    ) -> tuple:
        """Returns ocid and name of the experiment."""
        return (
            model.additional_details.get(RqsAdditionalDetails.MODEL_VERSION_SET_ID),
            model.additional_details.get(RqsAdditionalDetails.MODEL_VERSION_SET_NAME),
        )

    def _process(
        self,
        model: oci.resource_search.models.ResourceSummary,
        resources_mapping: dict = {},
    ) -> dict:
        """Constructs AquaEvaluationSummary from `oci.resource_search.models.ResourceSummary`."""

        tags = {}
        tags.update(model.defined_tags or {})
        tags.update(model.freeform_tags or {})

        model_id = model.identifier
        console_url = get_console_link(
            resource="models",
            ocid=model_id,
            region=self.region,
        )
        source_model_id, source_model_name = self._get_source(model, resources_mapping)
        experiment_id, experiment_name = self._get_experiment_info(model)
        parameters = self._fetch_runtime_params(model)

        return dict(
            id=model_id,
            name=model.display_name,
            console_url=console_url,
            time_created=model.time_created,
            tags=tags,
            experiment=self._build_resource_identifier(
                id=experiment_id,
                name=experiment_name,
            ),
            source=self._build_resource_identifier(
                id=source_model_id, name=source_model_name
            ),
            parameters=parameters,
        )

    def _build_resource_identifier(
        self, id: str = None, name: str = None
    ) -> AquaResourceIdentifier:
        """Constructs AquaResourceIdentifier based on the given ocid and display name."""
        try:
            resource_type = utils.CONSOLE_LINK_RESOURCE_TYPE_MAPPING.get(
                utils.get_resource_type(id)
            )

            return AquaResourceIdentifier(
                id=id,
                name=name,
                url=get_console_link(
                    resource=resource_type,
                    ocid=id,
                    region=self.region,
                ),
            )
        except Exception as e:
            logger.error(
                f"Failed to construct AquaResourceIdentifier from given id=`{id}`, and name=`{name}`, {str(e)}"
            )
            return AquaResourceIdentifier()

    def _get_jobrun(
        self, model: oci.resource_search.models.ResourceSummary, mapping: dict = {}
    ) -> Union[
        oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
    ]:
        jobrun_id = self._get_attribute_from_model_metadata(
            model, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID.value
        )
        job_run = mapping.get(jobrun_id)

        if not job_run:
            job_run = self._fetch_jobrun(model, use_rqs=True, jobrun_id=jobrun_id)
        return job_run

    def _fetch_jobrun(
        self,
        resource: oci.resource_search.models.ResourceSummary,
        use_rqs: bool = True,
        jobrun_id: str = None,
    ) -> Union[
        oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
    ]:
        """Extracts job run id from metadata, and gets related job run information."""

        jobrun_id = jobrun_id or self._get_attribute_from_model_metadata(
            resource, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID.value
        )

        logger.info(f"Fetching associated job run: {jobrun_id}")

        try:
            jobrun = (
                utils.query_resource(jobrun_id, return_all=False)
                if use_rqs
                else self.ds_client.get_job_run(jobrun_id).data
            )
        except Exception as e:
            logger.debug(
                f"Failed to retreive job run: {jobrun_id}. " f"DEBUG INFO: {str(e)}"
            )
            jobrun = None

        return jobrun

    def _fetch_runtime_params(
        self, resource: oci.resource_search.models.ResourceSummary
    ) -> AquaEvalParams:
        """Extracts model parameters from metadata. Shape is the shape used in job run."""
        try:
            params = json.loads(
                self._get_attribute_from_model_metadata(
                    resource, MetadataTaxonomyKeys.HYPERPARAMETERS
                )
            )
            if not params.get(EvaluationConfig.PARAMS):
                raise AquaMissingKeyError(
                    "model parameters have not been saved in correct format in model taxonomy.",
                    service_payload={"params": params},
                )
            # TODO: validate the format of parameters.
            # self._validate_params(params)

            return AquaEvalParams(**params[EvaluationConfig.PARAMS])
        except Exception as e:
            logger.debug(
                f"Failed to retrieve model parameters for the model: {str(resource)}."
                f"DEBUG INFO: {str(e)}."
            )
            return AquaEvalParams()

    def _build_job_identifier(
        self,
        job_run_details: Union[
            oci.data_science.models.JobRun, oci.resource_search.models.ResourceSummary
        ] = None,
    ) -> AquaResourceIdentifier:
        try:
            job_id = (
                job_run_details.id
                if isinstance(job_run_details, oci.data_science.models.JobRun)
                else job_run_details.identifier
            )
            return self._build_resource_identifier(
                id=job_id, name=job_run_details.display_name
            )

        except Exception as e:
            logger.debug(
                f"Failed to get job details from job_run_details: {job_run_details}"
                f"DEBUG INFO:{str(e)}"
            )
            return AquaResourceIdentifier()

    # TODO: fix the logic for determine termination state
    def _get_status(
        self,
        model: oci.resource_search.models.ResourceSummary,
        jobrun: Union[
            oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
        ] = None,
    ) -> dict:
        """Builds evaluation status based on the model status and job run status.
        When detect `aqua_evaluation_error` in custom metadata, the jobrun is failed.
        However, if jobrun failed before saving this meta, we need to check the existance
        of the evaluation artifact.

        """
        # TODO: revisit for CANCELED evaluation
        job_run_status = (
            JobRun.LIFECYCLE_STATE_FAILED
            if self._get_attribute_from_model_metadata(
                model, EvaluationCustomMetadata.EVALUATION_ERROR.value
            )
            else None
        )

        model_status = model.lifecycle_state
        job_run_status = job_run_status or (
            jobrun.lifecycle_state
            if jobrun and not jobrun.lifecycle_state == JobRun.LIFECYCLE_STATE_DELETED
            else (
                JobRun.LIFECYCLE_STATE_SUCCEEDED
                if self._if_eval_artifact_exist(model)
                else JobRun.LIFECYCLE_STATE_FAILED
            )
        )

        lifecycle_state = utils.LifecycleStatus.get_status(
            evaluation_status=model_status, job_run_status=job_run_status
        )

        try:
            lifecycle_details = (
                utils.LIFECYCLE_DETAILS_MISSING_JOBRUN
                if not jobrun
                else self._extract_job_lifecycle_details(jobrun.lifecycle_details)
            )
        except:
            # ResourceSummary does not have lifecycle_details attr
            lifecycle_details = ""

        return dict(
            lifecycle_state=(
                lifecycle_state
                if isinstance(lifecycle_state, str)
                else lifecycle_state.value
            ),
            lifecycle_details=lifecycle_details,
        )

    def _prefetch_resources(self, compartment_id) -> dict:
        """Fetches all AQUA resources."""
        # TODO: handle cross compartment/tenency resources
        # TODO: add cache
        resources = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="all",
            tag_list=[EvaluationModelTags.AQUA_EVALUATION.value, "OCI_AQUA"],
            connect_by_ampersands=False,
            return_all=False,
        )
        logger.debug(f"Fetched {len(resources)} AQUA resources.")
        return {item.identifier: item for item in resources}

    def _extract_job_lifecycle_details(self, lifecycle_details: str) -> str:
        """
        Extracts the exit code from a job lifecycle detail string and associates it
        with a corresponding message from the EVALUATION_JOB_EXIT_CODE_MESSAGE dictionary.

        This method searches the provided lifecycle detail string for an exit code pattern.
        Upon finding an exit code, it retrieves the related human-readable message
        from a predefined dictionary of exit codes and their meanings. If the exit code
        is not found within the string, or if it does not exist in the dictionary,
        the original `lifecycle_details` message will be returned.

        Parameters
        ----------
        lifecycle_details : str
            A string containing the details of the job's lifecycle, typically including an exit code.

        Returns
        -------
        str
            A message that combines the extracted exit code with its corresponding descriptive text.
            If no exit code is found, or if the exit code is not in the dictionary,
            the original `lifecycle_details` message will be returned.

        Examples
        --------
        >>> _extract_job_lifecycle_details("Job run artifact execution failed with exit code 16")
        'The evaluation configuration is invalid due to content validation errors.'

        >>> _extract_job_lifecycle_details("Job completed successfully.")
        'Job completed successfully.'
        """
        if not lifecycle_details:
            return lifecycle_details

        message = lifecycle_details
        try:
            # Extract exit code
            match = re.search(r"exit code (\d+)", lifecycle_details)
            if match:
                exit_code = int(match.group(1))
                exit_code_message = EVALUATION_JOB_EXIT_CODE_MESSAGE.get(exit_code)
                if exit_code_message:
                    message = f"{exit_code_message} Exit code: {exit_code}."
        except:
            pass

        return message

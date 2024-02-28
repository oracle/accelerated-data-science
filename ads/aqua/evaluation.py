#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import oci
from cachetools import TTLCache
from oci.data_science.models import (
    Metadata,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.exception import (
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.utils import MODEL_PARAMETERS, UNKNOWN, is_valid_ocid, upload_file_to_os
from ads.common import oci_client as oc
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link, get_files
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.jobs.builders.runtimes.base import Runtime
from ads.jobs.builders.runtimes.python_runtime import PythonRuntime
from ads.model.datascience_model import DataScienceModel
from ads.model.deployment.model_deployment import ModelDeployment
from ads.model.model_metadata import (
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.model_version_set import ModelVersionSet


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


class EvaluationModelTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"


class EvaluationJobTags(Enum):
    AQUA_EVALUATION = "aqua_evaluation"
    EVALUATION_MODEL_ID = "evaluation_model_id"


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
class AquaEvalParams(DataClassSerializable):
    shape: str = ""
    max_tokens: str = ""
    top_p: str = ""
    top_k: str = ""
    temperature: str = ""


@dataclass(repr=False)
class AquaEvalMetric(DataClassSerializable):
    key: str
    name: str
    content: str
    description: str = ""


@dataclass(repr=False)
class AquaEvalMetrics(DataClassSerializable):
    id: str
    metrics: List[AquaEvalMetric] = field(default_factory=list)


@dataclass(repr=False)
class AquaEvalMetrics(DataClassSerializable):
    id: str
    metrics: List[AquaEvalMetric] = field(default_factory=list)


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


class RqsAdditionalDetails:
    METADATA = "metadata"
    CREATED_BY = "createdBy"
    DESCRIPTION = "description"
    MODEL_VERSION_SET_ID = "modelVersionSetId"
    MODEL_VERSION_SET_NAME = "modelVersionSetName"
    PROJECT_ID = "projectId"
    VERSION_LABEL = "versionLabel"


class EvaluationTags:
    AQUA_EVALUATION = "aqua_evaluation"


class EvaluationMetadata:
    EVALUATION_SOURCE = "evaluation_source"
    HYPERPARAMETERS = "Hyperparameters"
    EVALUATION_JOB_ID = "evaluation_job_id"
    EVALUATION_SOURCE_NAME = "evaluation_source_name"


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
    memory_in_gbs: float
    ocpus: float
    block_storage_size: int
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    evaluation_description: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    log_group_id: Optional[str] = None
    log_id: Optional[str] = None
    metrics: Optional[List] = None


# TODO: Remove later
BUCKET_URI = "oci://ming-dev@ociodscdev/evaluation/sample_response"
SOURCE = "oci://lu_bucket@ociodscdev/evaluation_dummy_script.py"
SUBNET_ID = os.environ.get("SUBNET_ID", None)


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
    _cache_lock = Lock()

    def create(
        self, create_aqua_evaluation_details: CreateAquaEvaluationDetails, **kwargs
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

        evaluation_model_parameters = None
        try:
            evaluation_model_parameters = AquaEvalParams(
                shape=create_aqua_evaluation_details.shape_name,
                **create_aqua_evaluation_details.model_parameters,
            )
        except:
            raise AquaValueError(
                "Invalid model parameters. Model parameters should "
                f"be a dictionary with keys: {', '.join(MODEL_PARAMETERS)}."
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
            except:
                logger.debug(
                    f"Model version set {experiment_model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )
                model_version_set = (
                    ModelVersionSet()
                    .with_compartment_id(target_compartment)
                    .with_project_id(target_project)
                    .with_name(experiment_model_version_set_name)
                    .with_description(
                        create_aqua_evaluation_details.experiment_description
                    )
                    # TODO: decide what parameters will be needed
                    .create(**kwargs)
                )
                logger.debug(
                    f"Successfully created model version set {experiment_model_version_set_name} with id {model_version_set.id}."
                )
            experiment_model_version_set_id = model_version_set.id
        else:
            model_version_set = ModelVersionSet.from_id(experiment_model_version_set_id)
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

        evaluation_model_freeform_tags = {
            EvaluationModelTags.AQUA_EVALUATION.value: EvaluationModelTags.AQUA_EVALUATION.value,
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
            .with_freeform_tags(**evaluation_model_freeform_tags)
            # TODO: decide what parameters will be needed
            .create(
                remove_existing_artifact=False,  # TODO: added here for the puopose of demo and will revisit later
                **kwargs,
            )
        )
        logger.debug(
            f"Successfully created evaluation model {evaluation_model.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_dataset_path = create_aqua_evaluation_details.dataset_path
        if not ObjectStorageDetails.is_oci_path(evaluation_dataset_path):
            # format: oci://<bucket>@<namespace>/<evaluation_model_id>/<dataset_file_name>
            dst_uri = f"{create_aqua_evaluation_details.report_path}/{evaluation_model.id}/{os.path.basename(evaluation_dataset_path)}"
            upload_file_to_os(
                src_uri=evaluation_dataset_path,
                dst_uri=dst_uri,
                auth=self._auth,
                force_overwrite=True,
            )
            logger.debug(
                f"Uploaded local file {evaluation_dataset_path} to object storage {dst_uri}."
            )
            evaluation_dataset_path = dst_uri

        # TODO: validat metrics if it's provided

        evaluation_job_freeform_tags = {
            EvaluationJobTags.AQUA_EVALUATION.value: EvaluationJobTags.AQUA_EVALUATION.value,
            EvaluationJobTags.EVALUATION_MODEL_ID.value: evaluation_model.id,
        }

        evaluation_job = (
            Job(name=evaluation_model.display_name)
            .with_infrastructure(
                DataScienceJob()
                .with_log_group_id(create_aqua_evaluation_details.log_group_id)
                .with_log_id(create_aqua_evaluation_details.log_id)
                .with_compartment_id(target_compartment)
                .with_project_id(target_project)
                .with_shape_name(create_aqua_evaluation_details.shape_name)
                .with_shape_config_details(
                    memory_in_gbs=create_aqua_evaluation_details.memory_in_gbs,
                    ocpus=create_aqua_evaluation_details.ocpus,
                )
                .with_block_storage_size(
                    create_aqua_evaluation_details.block_storage_size
                )
                .with_freeform_tag(**evaluation_job_freeform_tags)
                .with_subnet_id(SUBNET_ID)
            )
            .with_runtime(
                self._build_evaluation_runtime(
                    dataset_path=evaluation_dataset_path,
                    report_path=create_aqua_evaluation_details.report_path,
                    model_parameters=create_aqua_evaluation_details.model_parameters,
                    metrics=create_aqua_evaluation_details.metrics,
                )
            )
            .create(**kwargs)  ## TODO: decide what parameters will be needed
        )
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
                custom_metadata_list=updated_custom_metadata_list
            ),
        )

        self.ds_client.update_model_provenance(
            model_id=evaluation_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=evaluation_job_run.id
            ),
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
        dataset_path: str,
        report_path: str,
        model_parameters: dict,
        metrics: List = None,
        **kwargs,
    ) -> Runtime:
        """Builds evaluation runtime for Job."""
        # TODO: update the logic to evaluate the model or model deployment
        runtime = (
            PythonRuntime()
            .with_service_conda("pytorch21_p39_gpu_v1")
            .with_source(SOURCE)
        )

        return runtime

    def get(self, eval_id) -> AquaEvaluationSummary:
        """Gets the information of an Aqua evalution.

        Parameters
        ----------
        eval_id: str
            The model OCID.

        Returns
        -------
        AquaEvaluationSummary:
            The instance of AquaEvaluationSummary.
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

        summary = AquaEvaluationSummary(
            **self._process(resource),
            **self._get_status(
                resource.lifecycle_state, job_status=job_run_details.lifecycle_state
            ),
            job=self._build_job_identifier(
                job_run_details=job_run_details,
            ),
        )
        summary.parameters.shape = (
            job_run_details.job_infrastructure_configuration_details.shape_name
        )
        return summary

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
            tag_list=[EvaluationTags.AQUA_EVALUATION],
        )
        logger.info(f"Fetched {len(models)} evaluations.")

        # TODO: add filter based on project_id if needed.

        mapping = self._prefetch_resources(compartment_id)

        # TODO: check if caching for service model list can be used
        evaluations = []
        for model in models:
            job_run = self._get_jobrun(model, mapping)
            job_status = job_run.lifecycle_state if job_run else None

            evaluations.append(
                AquaEvaluationSummary(
                    **self._process(model),
                    **self._get_status(
                        model_status=model.lifecycle_state,
                        job_status=job_status,
                    ),
                    job=self._build_job_identifier(
                        job_run_details=job_run,
                    ),
                )
            )
        return evaluations

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
        # TODO: add job_run_id as input param
        eval = utils.query_resource(eval_id)
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        if not eval:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )
        jobrun_id = model_provenance.training_id
        job_run_details = self._fetch_jobrun(eval, use_rqs=True, jobrun_id=jobrun_id)

        return dict(
            id=eval_id,
            **self._get_status(
                model_status=eval.lifecycle_state,
                job_status=job_run_details.lifecycle_state,
            ),
        )

    def load_metrics(self, eval_id: str) -> AquaEvalMetrics:
        """Loads evalution metrics markdown from artifacts.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        AquaEvalMetrics:
            An instancec of AquaEvalMetrics.
        """
        if eval_id in self._metrics_cache.keys():
            logger.info(f"Returning metrics from cache.")
            eval_metrics = self._metrics_cache.get(eval_id)
            if len(eval_metrics.metrics) > 0:
                return eval_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading evaluation artifact: {eval_id}.")
            DataScienceModel.from_id(eval_id).download_artifact(
                temp_dir,
                auth=self._auth,
            )
            metrics = []
            metric_markdown = {}
            report = None
            for file in get_files(temp_dir):
                if file.endswith(".md"):
                    metric_key = Path(file).stem
                    logger.info(f"Reading {file}...")
                    with open(os.path.join(temp_dir, file), "rb") as f:
                        content = f.read()

                    metric_markdown[metric_key] = base64.b64encode(content).decode()

                if file == utils.EVALUATION_REPORT_JSON:
                    logger.info(f"Loading {utils.EVALUATION_REPORT_JSON}...")
                    with open(
                        os.path.join(temp_dir, utils.EVALUATION_REPORT_JSON), "rb"
                    ) as f:
                        report = json.loads(f.read())

            if not report:
                raise AquaFileNotFoundError(
                    "Related Resource Not Authorized Or Not Found:"
                    f"Missing `{utils.EVALUATION_REPORT_JSON}` in evaluation artifact."
                )

            # TODO: after finalizing the format of report.json, move the constant to class
            metrics_results = report.get("metric_results")
            missing_content = False
            for k, v in metrics_results.items():
                content = metric_markdown.get(k, utils.UNKNOWN)
                if not content:
                    missing_content = True
                    logger.error(
                        "Related Resource Not Authorized Or Not Found:"
                        f"Missing `{k}.md` in evaluation artifact."
                    )

                metrics.append(
                    AquaEvalMetric(
                        key=k,
                        name=v.get("name", utils.UNKNOWN),
                        content=content,
                        description=v.get("description"),
                    )
                )

        eval_metrics = AquaEvalMetrics(id=eval_id, metrics=metrics)

        if not missing_content:
            self._metrics_cache.__setitem__(key=eval_id, value=eval_metrics)

        return eval_metrics

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
            content = ""
            for file in get_files(temp_dir):
                if os.path.basename(file) == utils.EVALUATION_REPORT:
                    report_path = os.path.join(temp_dir, utils.EVALUATION_REPORT)
                    with open(report_path, "rb") as f:
                        content = f.read()
                    break

            if not content:
                error_msg = "Related Resource Not Authorized Or Not Found:" + (
                    f"Missing `{utils.EVALUATION_REPORT}` in evaluation artifact."
                )
                raise AquaFileNotFoundError(error_msg)

        report = AquaEvalReport(
            evaluation_id=eval_id, content=base64.b64encode(content).decode()
        )

        self._report_cache.__setitem__(key=eval_id, value=report)

        return report

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
            EvaluationMetadata.EVALUATION_SOURCE,
        )

        try:
            source = resources_mapping.get(source_id)
            source_name = (
                source.display_name
                if source
                else self._get_attribute_from_model_metadata(
                    evaluation, EvaluationMetadata.EVALUATION_SOURCE_NAME
                )
            )

            if not source_name:
                resource_type = utils.get_resource_type(source_id)

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
            model, EvaluationMetadata.EVALUATION_JOB_ID
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
            resource, EvaluationMetadata.EVALUATION_JOB_ID
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
                    resource, EvaluationMetadata.HYPERPARAMETERS
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

    def _get_status(self, model_status, job_status) -> dict:
        """Build evaluation status based on the model status and job run status."""
        lifecycle_state = utils.LifecycleStatus.get_status(
            evaluation_status=model_status, job_run_status=job_status
        )
        return dict(
            lifecycle_state=(
                lifecycle_state
                if isinstance(lifecycle_state, str)
                else lifecycle_state.value
            ),
            lifecycle_details=(
                utils.UNKNOWN
                if isinstance(lifecycle_state, str)
                else lifecycle_state.detail
            ),
        )

    def _prefetch_resources(self, compartment_id) -> dict:
        """Fetches all AQUA resources."""
        # TODO: handle cross compartment/tenency resources
        # TODO: add cache
        resources = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="all",
            tag_list=[EvaluationTags.AQUA_EVALUATION, "OCI_AQUA"],
            connect_by_ampersands=False,
            return_all=False,
        )
        logger.info(f"Fetched {len(resources)} AQUA resources.")
        return {item.identifier: item for item in resources}

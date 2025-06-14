#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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

from ads.aqua import logger
from ads.aqua.app import AquaApp
from ads.aqua.common import utils
from ads.aqua.common.entities import AquaMultiModelRef
from ads.aqua.common.enums import (
    DataScienceResource,
    Resource,
    RqsAdditionalDetails,
    Tags,
)
from ads.aqua.common.errors import (
    AquaFileExistsError,
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.common.utils import (
    extract_id_and_name_from_tag,
    fire_and_forget,
    is_valid_ocid,
    upload_local_to_os,
)
from ads.aqua.config.evaluation.evaluation_service_config import (
    DEFAULT_EVALUATION_CONTAINER,
    EvaluationServiceConfig,
    MetricConfig,
)
from ads.aqua.constants import (
    CONSOLE_LINK_RESOURCE_TYPE_MAPPING,
    EVALUATION_REPORT,
    EVALUATION_REPORT_JSON,
    EVALUATION_REPORT_MD,
    JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING,
    LIFECYCLE_DETAILS_MISSING_JOBRUN,
    NB_SESSION_IDENTIFIER,
)
from ads.aqua.evaluation.constants import (
    EVAL_TERMINATION_STATE,
    EvaluationConfig,
    EvaluationCustomMetadata,
    EvaluationMetricResult,
    EvaluationReportJson,
)
from ads.aqua.evaluation.entities import (
    AquaEvalMetric,
    AquaEvalMetrics,
    AquaEvalMetricSummary,
    AquaEvalParams,
    AquaEvalReport,
    AquaEvaluationCommands,
    AquaEvaluationDetail,
    AquaEvaluationSummary,
    AquaResourceIdentifier,
    CreateAquaEvaluationDetails,
)
from ads.aqua.evaluation.errors import EVALUATION_JOB_EXIT_CODE_MESSAGE
from ads.aqua.model.constants import ModelCustomMetadataFields
from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import (
    UNKNOWN,
    get_console_link,
    get_files,
    get_log_links,
    read_file,
)
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
from ads.model.deployment import ModelDeploymentContainerRuntime
from ads.model.deployment.model_deployment import ModelDeployment
from ads.model.generic_model import ModelDeploymentRuntimeType
from ads.model.model_metadata import (
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelCustomMetadataItem,
    ModelProvenanceMetadata,
    ModelTaxonomyMetadata,
)
from ads.model.model_version_set import ModelVersionSet
from ads.telemetry import telemetry


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
    _deletion_cache = TTLCache(
        maxsize=10, ttl=timedelta(minutes=10), timer=datetime.now
    )
    _cache_lock = Lock()

    @telemetry(entry_point="plugin=evaluation&action=create", name="aqua")
    def create(
        self,
        create_aqua_evaluation_details: CreateAquaEvaluationDetails = None,
        **kwargs,
    ) -> "AquaEvaluationSummary":
        """Creates Aqua evaluation for resource.\n
        For detailed information about CLI flags see: https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/cli-tips.md#model-evaluation

        Parameters
        ----------
        create_aqua_evaluation_details: CreateAquaEvaluationDetails
            The CreateAquaEvaluationDetails data class which contains all
            required and optional fields to create the aqua evaluation.
            kwargs:
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
                force_overwrite: (bool, optional). Defaults to `False`.
                    Whether to force overwrite the existing file in object storage.
                freeform_tags: (dict, optional)
                    Freeform tags for the evaluation model
                defined_tags: (dict, optional)
                    Defined tags for the evaluation model

        kwargs:
            The kwargs for creating CreateAquaEvaluationDetails instance if
            no create_aqua_evaluation_details provided.

        Returns
        -------
        AquaEvaluationSummary:
            The instance of AquaEvaluationSummary.
        """
        if not create_aqua_evaluation_details:
            try:
                create_aqua_evaluation_details = CreateAquaEvaluationDetails(**kwargs)
            except Exception as ex:
                custom_errors = {
                    ".".join(map(str, e["loc"])): e["msg"]
                    for e in json.loads(ex.json())
                }
                raise AquaValueError(
                    f"Invalid create evaluation parameters. Error details: {custom_errors}."
                ) from ex

        if not is_valid_ocid(create_aqua_evaluation_details.evaluation_source_id):
            raise AquaValueError(
                f"Invalid evaluation source {create_aqua_evaluation_details.evaluation_source_id}. "
                "Specify either a model or model deployment id."
            )

        # The model to evaluate
        evaluation_source = None
        eval_inference_configuration: Dict = {}

        if (
            DataScienceResource.MODEL_DEPLOYMENT
            in create_aqua_evaluation_details.evaluation_source_id
        ):
            evaluation_source = ModelDeployment.from_id(
                create_aqua_evaluation_details.evaluation_source_id
            )

            if Tags.MULTIMODEL_TYPE_TAG in evaluation_source.freeform_tags:
                multi_model_id = evaluation_source.freeform_tags.get(
                    Tags.AQUA_MODEL_ID_TAG, UNKNOWN
                )

                if not multi_model_id:
                    raise AquaRuntimeError(
                        f"Invalid multi model deployment {multi_model_id}."
                        f"Make sure the {Tags.AQUA_MODEL_ID_TAG} tag is added to the deployment."
                    )

                aqua_model = DataScienceModel.from_id(multi_model_id)
                AquaEvaluationApp.validate_model_name(
                    aqua_model, create_aqua_evaluation_details
                )

            try:
                if (
                    evaluation_source.runtime.type
                    == ModelDeploymentRuntimeType.CONTAINER
                ):
                    runtime = ModelDeploymentContainerRuntime.from_dict(
                        evaluation_source.runtime.to_dict()
                    )
                    inference_config = (
                        self.get_container_config().to_dict().get("inference")
                    )
                    for container in inference_config:
                        if container.name == runtime.image[: runtime.image.rfind(":")]:
                            eval_inference_configuration = (
                                container.spec.evaluation_configuration
                            )
            except Exception as ex:
                logger.debug(
                    f"Could not load inference config details for the evaluation source id: "
                    f"{create_aqua_evaluation_details.evaluation_source_id}. Please check if the container"
                    f" runtime has the correct SMC image information.\nError: {str(ex)}"
                )
        elif (
            DataScienceResource.MODEL
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
                    force_overwrite=create_aqua_evaluation_details.force_overwrite,
                )
            except FileExistsError as err:
                raise AquaFileExistsError(
                    f"Dataset {dataset_file} already exists in {create_aqua_evaluation_details.report_path}. "
                    "Please use a new dataset file name, report path or set `force_overwrite` as True."
                ) from err
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

        evaluation_model_parameters = AquaEvalParams(
            shape=create_aqua_evaluation_details.shape_name,
            dataset_path=evaluation_dataset_path,
            report_path=create_aqua_evaluation_details.report_path,
            **create_aqua_evaluation_details.model_parameters,
        )

        evaluation_model_defined_tags = (
            create_aqua_evaluation_details.defined_tags or {}
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
                if not utils._is_valid_mvs(model_version_set, Tags.AQUA_EVALUATION):
                    raise AquaValueError(
                        f"Invalid experiment name. Please provide an experiment with `{Tags.AQUA_EVALUATION}` in tags."
                    )
            except Exception:
                logger.info(
                    f"Model version set {experiment_model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )

                evaluation_mvs_freeform_tags = {
                    Tags.AQUA_EVALUATION: Tags.AQUA_EVALUATION,
                }
                evaluation_mvs_freeform_tags = {
                    **evaluation_mvs_freeform_tags,
                    **(create_aqua_evaluation_details.freeform_tags or {}),
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
                    .with_defined_tags(**evaluation_model_defined_tags)
                    # TODO: decide what parameters will be needed
                    .create(**kwargs)
                )
                logger.debug(
                    f"Successfully created model version set {experiment_model_version_set_name} with id {model_version_set.id}."
                )
            experiment_model_version_set_id = model_version_set.id
        else:
            model_version_set = ModelVersionSet.from_id(experiment_model_version_set_id)
            if not utils._is_valid_mvs(model_version_set, Tags.AQUA_EVALUATION):
                raise AquaValueError(
                    f"Invalid experiment id. Please provide an experiment with `{Tags.AQUA_EVALUATION}` in tags."
                )
            experiment_model_version_set_name = model_version_set.name

        evaluation_model_custom_metadata = ModelCustomMetadata()
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_SOURCE,
            value=create_aqua_evaluation_details.evaluation_source_id,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_OUTPUT_PATH,
            value=create_aqua_evaluation_details.report_path,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_SOURCE_NAME,
            value=evaluation_source.display_name,
        )

        evaluation_model_taxonomy_metadata = ModelTaxonomyMetadata()
        evaluation_model_taxonomy_metadata[
            MetadataTaxonomyKeys.HYPERPARAMETERS
        ].value = {"model_params": evaluation_model_parameters.to_dict()}

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
            .with_defined_tags(**evaluation_model_defined_tags)
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
            Tags.AQUA_EVALUATION: Tags.AQUA_EVALUATION,
            Tags.AQUA_EVALUATION_MODEL_ID: evaluation_model.id,
        }
        evaluation_job_freeform_tags = {
            **evaluation_job_freeform_tags,
            **(create_aqua_evaluation_details.freeform_tags or {}),
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
            .with_defined_tag(**evaluation_model_defined_tags)
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
        elif NB_SESSION_IDENTIFIER in os.environ:
            # apply default subnet id for job by setting ME_STANDALONE
            # so as to avoid using the notebook session's networking when running on it
            # https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/infra_and_runtime.html#networking
            evaluation_job.infrastructure.with_job_infrastructure_type(
                JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING
            )

        container_image = (
            create_aqua_evaluation_details.container_image_uri
            or self._get_evaluation_container(
                create_aqua_evaluation_details.evaluation_source_id
            )
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
                model_parameters={
                    **create_aqua_evaluation_details.model_parameters,
                },
                metrics=create_aqua_evaluation_details.metrics,
                inference_configuration=eval_inference_configuration or {},
            )
        ).create(**kwargs)  ## TODO: decide what parameters will be needed
        logger.debug(
            f"Successfully created evaluation job {evaluation_job.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_job_run = evaluation_job.run(
            name=evaluation_model.display_name,
            freeform_tags=evaluation_job_freeform_tags,
            defined_tags=evaluation_model_defined_tags,
            wait=False,
        )
        logger.debug(
            f"Successfully created evaluation job run {evaluation_job_run.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_JOB_ID,
            value=evaluation_job.id,
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID,
            value=evaluation_job_run.id,
        )
        updated_custom_metadata_list = [
            Metadata(**metadata)
            for metadata in evaluation_model_custom_metadata.to_dict()["data"]
        ]

        evaluation_model_freeform_tags = {
            Tags.AQUA_EVALUATION: Tags.AQUA_EVALUATION,
            **(create_aqua_evaluation_details.freeform_tags or {}),
        }

        self.ds_client.update_model(
            model_id=evaluation_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list,
                freeform_tags=evaluation_model_freeform_tags,
            ),
        )

        self.ds_client.update_model_provenance(
            model_id=evaluation_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=evaluation_job_run.id
            ),
        )

        # tracks shapes used in evaluation that were created for the given evaluation source
        self.telemetry.record_event_async(
            category="aqua/evaluation/create",
            action="shape",
            detail=create_aqua_evaluation_details.shape_name,
            value=self._get_service_model_name(evaluation_source),
        )

        # tracks unique evaluation that were created for the given evaluation source
        self.telemetry.record_event_async(
            category="aqua/evaluation",
            action="create",
            detail=self._get_service_model_name(evaluation_source),
        )

        return AquaEvaluationSummary(
            id=evaluation_model.id,
            name=evaluation_model.display_name,
            console_url=get_console_link(
                resource=Resource.MODEL,
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
                    resource=Resource.MODEL_VERSION_SET,
                    ocid=experiment_model_version_set_id,
                    region=self.region,
                ),
            ),
            source=AquaResourceIdentifier(
                id=create_aqua_evaluation_details.evaluation_source_id,
                name=evaluation_source.display_name,
                url=get_console_link(
                    resource=(
                        Resource.MODEL_DEPLOYMENT
                        if DataScienceResource.MODEL_DEPLOYMENT
                        in create_aqua_evaluation_details.evaluation_source_id
                        else Resource.MODEL
                    ),
                    ocid=create_aqua_evaluation_details.evaluation_source_id,
                    region=self.region,
                ),
            ),
            job=AquaResourceIdentifier(
                id=evaluation_job.id,
                name=evaluation_job.name,
                url=get_console_link(
                    resource=Resource.JOB,
                    ocid=evaluation_job.id,
                    region=self.region,
                ),
            ),
            tags={
                "aqua_evaluation": Tags.AQUA_EVALUATION,
                "evaluation_job_id": evaluation_job.id,
                "evaluation_source": create_aqua_evaluation_details.evaluation_source_id,
                "evaluation_experiment_id": experiment_model_version_set_id,
                **evaluation_model_freeform_tags,
                **evaluation_model_defined_tags,
            },
            parameters=AquaEvalParams(),
        )

    @staticmethod
    def validate_model_name(
        evaluation_source: DataScienceModel,
        create_aqua_evaluation_details: CreateAquaEvaluationDetails,
    ) -> None:
        """
        Validates the user input for the model name when creating an Aqua evaluation.

        This function verifies that:
        - The model group is not empty.
        - The model multi metadata is present in the DataScienceModel metadata.
        - The user provided a non-empty model name.
        - The provided model name exists in the DataScienceModel metadata.
        - The deployment configuration contains core metadata required for validation.

        Parameters
        ----------
        evaluation_source : DataScienceModel
            The DataScienceModel object containing metadata about each model in the deployment.
        create_aqua_evaluation_details : CreateAquaEvaluationDetails
            Contains required and optional fields for creating the Aqua evaluation.

        Raises
        ------
        AquaValueError
            If the user fails to provide a model name or if the provided model name does not match
            any of the valid model names in the deployment metadata.
        AquaRuntimeError
            If the metadata is missing the model group count or if the model group count is invalid.
        """
        user_model_parameters = create_aqua_evaluation_details.model_parameters
        custom_metadata_list = evaluation_source.custom_metadata_list
        user_model_name = user_model_parameters.get("model")

        # Ensure that a non-empty model name was provided.
        if not user_model_name:
            error_message = (
                "No model name was provided for evaluation. For multi-model deployment, "
                "a model must be specified in the model parameters."
            )
            logger.debug(error_message)
            raise AquaValueError(error_message)

        # Retrieve and convert the model group count from metadata.
        model_count = custom_metadata_list.get(
            ModelCustomMetadataFields.MULTIMODEL_GROUP_COUNT
        )
        try:
            model_group_count = int(model_count.value)
        except Exception as ex:
            error_message = (
                "Missing or invalid `MULTIMODEL_GROUP_COUNT` "
                f"in custom metadata for evaluation source ID '{create_aqua_evaluation_details.evaluation_source_id}'. "
                f"Details: {ex}"
            )
            logger.error(error_message)

        if model_group_count < 1:
            logger.error(
                f"Invalid model group count: {model_group_count} for evaluation source ID "
                f"'{create_aqua_evaluation_details.evaluation_source_id}'. A valid multi-model deployment "
                f"requires at least one model."
            )
            raise AquaRuntimeError(
                f"Unable to retrieve details for the multi-model deployment evaluation. A valid multi-model deployment "
                f"must include at least one model. However, the provided evaluation source ID "
                f"'{create_aqua_evaluation_details.evaluation_source_id}' does not contain any information about deployed models."
            )

        multi_model_metadata_value = custom_metadata_list.get(
            ModelCustomMetadataFields.MULTIMODEL_METADATA,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.MULTIMODEL_METADATA),
        ).value

        if not multi_model_metadata_value:
            error_message = (
                f"Required model metadata is missing for evaluation source ID: {evaluation_source.id}. "
                f"A valid multi-model deployment requires {ModelCustomMetadataFields.MULTIMODEL_METADATA}. "
                "Please recreate the model deployment and retry the evaluation, as an issue occurred during the initialization of the model group."
            )
            logger.debug(error_message)
            raise AquaRuntimeError(error_message)

        try:
            multi_model_metadata = json.loads(
                evaluation_source.dsc_model.get_custom_metadata_artifact(
                    metadata_key_name=ModelCustomMetadataFields.MULTIMODEL_METADATA
                ).decode("utf-8")
            )
        except Exception as ex:
            error_message = (
                f"Error fetching {ModelCustomMetadataFields.MULTIMODEL_METADATA} "
                f"from custom metadata for evaluation source ID '{evaluation_source.id}'. "
                f"Details: {ex}"
            )
            logger.error(error_message)
            raise AquaRuntimeError(error_message) from ex

        # Build the list of valid model names from custom metadata.
        model_names = []
        for metadata in multi_model_metadata:
            model = AquaMultiModelRef(**metadata)
            model_names.append(model.model_name)
            model_names.extend(ft.model_name for ft in (model.fine_tune_weights or []) if ft.model_name)

        # Check if the provided model name is among the valid names.
        if user_model_name not in model_names:
            error_message = (
                f"Provided model name '{user_model_name}' does not match any valid model names {model_names} "
                f"for evaluation source ID '{create_aqua_evaluation_details.evaluation_source_id}'. "
                "Please provide the correct model name."
            )
            logger.debug(error_message)
            raise AquaValueError(error_message)

    def _build_evaluation_runtime(
        self,
        evaluation_id: str,
        evaluation_source_id: str,
        container_image: str,
        dataset_path: str,
        report_path: str,
        model_parameters: dict,
        metrics: List = None,
        inference_configuration: dict = None,
    ) -> Runtime:
        """Builds evaluation runtime for Job."""
        # TODO the image name needs to be extracted from the mapping index.json file.
        runtime = (
            ContainerRuntime()
            .with_image(container_image)
            .with_environment_variable(
                **{
                    "AIP_SMC_EVALUATION_ARGUMENTS": json.dumps(
                        {
                            **self._build_launch_cmd(
                                evaluation_id=evaluation_id,
                                evaluation_source_id=evaluation_source_id,
                                dataset_path=dataset_path,
                                report_path=report_path,
                                model_parameters=model_parameters,
                                metrics=metrics,
                            ).to_dict(),
                            **(inference_configuration or {}),
                        },
                    ),
                    "CONDA_BUCKET_NS": CONDA_BUCKET_NS,
                },
            )
        )

        return runtime

    @staticmethod
    def _get_service_model_name(
        source: Union[ModelDeployment, DataScienceModel],
    ) -> str:
        """Gets the service model name from source. If it's ModelDeployment, needs to check
        if its model has been fine tuned or not.

        Parameters
        ----------
        source: Union[ModelDeployment, DataScienceModel]
            An instance of either ModelDeployment or DataScienceModel

        Returns
        -------
        str:
            The service model name of source.
        """
        if isinstance(source, ModelDeployment):
            fine_tuned_model_tag = source.freeform_tags.get(
                Tags.AQUA_FINE_TUNED_MODEL_TAG, UNKNOWN
            )
            if not fine_tuned_model_tag:
                return source.freeform_tags.get(Tags.AQUA_MODEL_NAME_TAG)
            else:
                return extract_id_and_name_from_tag(fine_tuned_model_tag)[1]

        return source.display_name

    def _get_evaluation_container(self, source_id: str) -> str:
        # todo: use the source, identify if it is a model or a deployment. If latter, then fetch the base model id
        #   from the deployment object, and call ds_client.get_model() to get model details. Use custom metadata to
        #   get the container_type_key. Pass this key as container_type to get_container_image method.

        # fetch image name from config
        container_image = self.get_container_image(
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
                "format": Path(dataset_path).suffix,
                "url": dataset_path,
            },
            metrics=metrics or [],
            output_dir=report_path,
            params=model_parameters or {},
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
        if not resource:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

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
        log_url = (
            get_log_links(
                region=self.region,
                log_group_id=loggroup_id,
                log_id=log_id,
                compartment_id=job_run_details.compartment_id,
                source_id=jobrun_id,
            )
            if job_run_details
            else ""
        )

        log_name = None
        loggroup_name = None

        if log_id:
            try:
                log = utils.query_resource(log_id, return_all=False)
                log_name = log.display_name if log else ""
            except Exception as ex:
                logger.debug(f"Failed to get associated log name. Error: {ex}")
                pass

        if loggroup_id:
            try:
                loggroup = utils.query_resource(loggroup_id, return_all=False)
                loggroup_name = loggroup.display_name if loggroup else ""
            except Exception as ex:
                logger.debug(f"Failed to get associated loggroup name. Error: {ex}")
                pass

        try:
            introspection = json.loads(
                self._get_attribute_from_model_metadata(resource, "ArtifactTestResults")
            )
        except Exception as ex:
            logger.debug(
                f"There was an issue loading the model attribute as json object for evaluation {eval_id}. "
                f"Setting introspection to empty.\n Error:{ex}"
            )
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
    def list(self, compartment_id: str = None) -> List[AquaEvaluationSummary]:
        """List Aqua evaluations in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.

        Returns
        -------
        List[AquaEvaluationSummary]:
            The list of the `ads.aqua.evalution.AquaEvaluationSummary`.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        logger.info(f"Fetching evaluations from compartment {compartment_id}.")
        models = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="datasciencemodel",
            tag_list=[Tags.AQUA_EVALUATION],
        )
        logger.info(f"Fetched {len(models)} evaluations.")

        mapping = self._prefetch_resources(compartment_id)

        evaluations = []
        async_tasks = []
        for model in models:
            if model.identifier in self._eval_cache:
                logger.debug(f"Retrieving evaluation {model.identifier} from cache.")
                evaluations.append(self._eval_cache.get(model.identifier))

            else:
                jobrun_id = self._get_attribute_from_model_metadata(
                    model, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID
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
                    logger.debug(
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
            return response.status == 200
        except oci.exceptions.ServiceError as ex:
            if ex.status == 404:
                logger.debug(f"Evaluation artifact not found for {model.identifier}.")
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

        if not eval:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )

        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        jobrun_id = model_provenance.training_id
        job_run_details = self._fetch_jobrun(eval, use_rqs=False, jobrun_id=jobrun_id)

        try:
            log_id = job_run_details.log_details.log_id
        except Exception as e:
            logger.debug(f"Failed to get associated log.\nError: {str(e)}")
            log_id = ""

        try:
            loggroup_id = job_run_details.log_details.log_group_id
        except Exception as e:
            logger.debug(f"Failed to get associated log.\nError: {str(e)}")
            loggroup_id = ""

        loggroup_url = get_log_links(region=self.region, log_group_id=loggroup_id)
        log_url = (
            get_log_links(
                region=self.region,
                log_group_id=loggroup_id,
                log_id=log_id,
                compartment_id=job_run_details.compartment_id,
                source_id=jobrun_id,
            )
            if job_run_details
            else ""
        )
        return {
            "id": eval_id,
            **self._get_status(
                model=eval,
                jobrun=job_run_details,
            ),
            "log_id": log_id,
            "log_url": log_url,
            "loggroup_id": loggroup_id,
            "loggroup_url": loggroup_url,
        }

    def get_supported_metrics(self) -> List[MetricConfig]:
        """Gets a list of supported metrics for evaluation."""
        containers = self.list_service_containers()
        container_item = next(
            c
            for c in containers
            if c.is_latest and c.family_name == DEFAULT_EVALUATION_CONTAINER
        )
        evaluation_service_config = EvaluationServiceConfig.from_oci_container_config(
            container_item
        )
        return evaluation_service_config.ui_config.metrics

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
        if eval_id in self._metrics_cache:
            logger.info("Returning metrics from cache.")
            eval_metrics = self._metrics_cache.get(eval_id)
            if len(eval_metrics.report) > 0:
                return eval_metrics

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading evaluation artifact: {eval_id}.")

            dsc_model = DataScienceModel.from_id(eval_id)
            if dsc_model.if_model_custom_metadata_artifact_exist(
                EVALUATION_REPORT_MD
            ) and dsc_model.if_model_custom_metadata_artifact_exist(
                EVALUATION_REPORT_JSON
            ):
                logger.info(
                    f"Fetching {EVALUATION_REPORT_MD} and {EVALUATION_REPORT_JSON} from custom metadata..."
                )
                dsc_model.get_custom_metadata_artifact(EVALUATION_REPORT_MD, temp_dir)
                dsc_model.get_custom_metadata_artifact(EVALUATION_REPORT_JSON, temp_dir)
            else:
                logger.info("Fetching Evaluation Reports from OSS bucket...")
                dsc_model.download_artifact(
                    temp_dir,
                    auth=self._auth,
                )

            files_in_artifact = get_files(temp_dir)
            md_report_content = self._read_from_artifact(
                temp_dir, files_in_artifact, EVALUATION_REPORT_MD
            )

            # json report not available for failed evaluation
            try:
                json_report = json.loads(
                    self._read_from_artifact(
                        temp_dir, files_in_artifact, EVALUATION_REPORT_JSON
                    )
                )
            except Exception as e:
                logger.debug(
                    f"Failed to load `report.json` from evaluation artifact.\nError: {str(e)}"
                )
                json_report = {}

        eval_metrics = AquaEvalMetrics(
            id=eval_id,
            report=base64.b64encode(md_report_content).decode(),
            metric_results=[
                AquaEvalMetric(
                    key=metadata.get(EvaluationMetricResult.SHORT_NAME, utils.UNKNOWN),
                    name=metadata.get(EvaluationMetricResult.NAME, utils.UNKNOWN),
                    description=metadata.get(
                        EvaluationMetricResult.DESCRIPTION, utils.UNKNOWN
                    ),
                )
                for _, metadata in json_report.get(
                    EvaluationReportJson.METRIC_RESULT, {}
                ).items()
            ],
            metric_summary_result=[
                AquaEvalMetricSummary(**m)
                for m in json_report.get(
                    EvaluationReportJson.METRIC_SUMMARY_RESULT, [{}]
                )
            ],
        )

        if md_report_content:
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
        if eval_id in self._report_cache:
            logger.info("Returning report from cache.")
            report = self._report_cache.get(eval_id)
            if report.content:
                return report

        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading evaluation artifact for {eval_id}.")
            dsc_model = DataScienceModel.from_id(eval_id)
            if_custom_metadata_exists = (
                dsc_model.if_model_custom_metadata_artifact_exist(EVALUATION_REPORT)
            )
            if if_custom_metadata_exists:
                logger.info(f"Fetching {EVALUATION_REPORT} from custom metadata.")
                dsc_model.get_custom_metadata_artifact(EVALUATION_REPORT, temp_dir)
            else:
                logger.info(f"Fetching {EVALUATION_REPORT} from Model artifact.")
                dsc_model.download_artifact(
                    temp_dir,
                    auth=self._auth,
                )
            files_in_artifact = get_files(temp_dir)
            if not len(files_in_artifact):
                try:
                    evaluation_output_path = dsc_model.custom_metadata_list.get(
                        EvaluationCustomMetadata.EVALUATION_OUTPUT_PATH
                    ).value
                    report_path = (
                        evaluation_output_path.rstrip("/")
                        + "/"
                        + eval_id
                        + "/"
                        + EVALUATION_REPORT
                    )
                    logger.info(
                        f"Fetching {EVALUATION_REPORT} from {report_path} for evaluation {eval_id}"
                    )
                    content = read_file(
                        file_path=report_path, auth=default_signer()
                    ).encode()
                except ValueError as err:
                    raise AquaValueError(
                        f"{EvaluationCustomMetadata.EVALUATION_OUTPUT_PATH} is missing from custom metadata for the model {eval_id}"
                    ) from err
            else:
                content = self._read_from_artifact(
                    temp_dir, files_in_artifact, EVALUATION_REPORT
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

        job_run_id = (
            model.provenance_metadata.training_id if model.provenance_metadata else None
        )
        if not job_run_id:
            raise AquaMissingKeyError(
                "Model provenance is missing job run training_id key"
            )

        status = {"id": eval_id, "lifecycle_state": UNKNOWN, "time_accepted": UNKNOWN}
        run = DataScienceJobRun.from_ocid(job_run_id)
        if run.lifecycle_state in [
            DataScienceJobRun.LIFECYCLE_STATE_ACCEPTED,
            DataScienceJobRun.LIFECYCLE_STATE_IN_PROGRESS,
            DataScienceJobRun.LIFECYCLE_STATE_NEEDS_ATTENTION,
        ]:
            self._cancel_job_run(run, model)
            status = {
                "id": eval_id,
                "lifecycle_state": "CANCELING",
                "time_accepted": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z"),
            }
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
            if a model doesn't exist for the given eval_id.
        AquaMissingKeyError:
            if job/jobrun id is missing.
        """

        model = DataScienceModel.from_id(eval_id)
        if not model:
            raise AquaRuntimeError(
                f"Failed to get evaluation details for model {eval_id}"
            )

        try:
            job_id = model.custom_metadata_list.get(
                EvaluationCustomMetadata.EVALUATION_JOB_ID
            ).value
        except Exception as ex:
            raise AquaMissingKeyError(
                f"Custom metadata is missing {EvaluationCustomMetadata.EVALUATION_JOB_ID} key"
            ) from ex

        job = DataScienceJob.from_id(job_id)

        self._delete_job_and_model(job, model)

        try:
            jobrun_id = model.custom_metadata_list.get(
                EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID
            ).value
            jobrun = utils.query_resource(jobrun_id, return_all=False)
        except Exception:
            logger.debug("Associated Job Run OCID is missing.")
            jobrun = None

        self._eval_cache.pop(key=eval_id, default=None)
        self._deletion_cache.__setitem__(key=eval_id, value="")

        status = {
            "id": eval_id,
            "lifecycle_state": jobrun.lifecycle_state if jobrun else "DELETING",
            "time_accepted": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%z"),
        }
        return status

    @staticmethod
    @fire_and_forget
    def _delete_job_and_model(job: DataScienceJob, model: DataScienceModel):
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

    def load_evaluation_config(
        self, container: Optional[str] = DEFAULT_EVALUATION_CONTAINER
    ) -> Dict:
        """Loads evaluation config."""

        logger.info("Loading evaluation container config.")
        # retrieve the evaluation config by container family name
        containers = self.list_service_containers()
        container_item = next(
            c for c in containers if c.is_latest and c.family_name == container
        )
        evaluation_config = EvaluationServiceConfig.from_oci_container_config(
            container_item
        )

        # convert the new config representation to the old one
        return {
            "model_params": evaluation_config.ui_config.model_params.default,
            "shape": {
                shape.name: shape.to_dict()
                for shape in evaluation_config.ui_config.shapes
            },
            "default": (
                evaluation_config.ui_config.shapes[0].to_dict()
                if len(evaluation_config.ui_config.shapes) > 0
                else {}
            ),
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
        except Exception:
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
        resources_mapping: dict = None,
    ) -> tuple:
        """Returns ocid and name of the model has been evaluated."""
        source_id = self._get_attribute_from_model_metadata(
            evaluation,
            EvaluationCustomMetadata.EVALUATION_SOURCE,
        )

        try:
            source_name = None
            if resources_mapping:
                source = resources_mapping.get(source_id)
                source_name = (
                    source.display_name
                    if source
                    else self._get_attribute_from_model_metadata(
                        evaluation, EvaluationCustomMetadata.EVALUATION_SOURCE_NAME
                    )
                )

            # try to resolve source_name from source id
            if source_id and not source_name:
                resource_type = utils.get_resource_type(source_id)

                if resource_type.startswith("datasciencemodeldeployment"):
                    source_name = self.ds_client.get_model_deployment(
                        source_id
                    ).data.display_name
                elif resource_type.startswith("datasciencemodel"):
                    source_name = self.ds_client.get_model(source_id).data.display_name
                else:
                    raise AquaRuntimeError(
                        f"Not supported source type: {resource_type}"
                    )
        except Exception as ex:
            logger.debug(
                f"Failed to retrieve source information for evaluation {evaluation.identifier}.\nError: {str(ex)}"
            )
            source_name = ""

        return source_id, source_name

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
        resources_mapping: dict = None,
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
        source_model_id, source_model_name = self._get_source(
            model, resources_mapping if resources_mapping else {}
        )
        experiment_id, experiment_name = self._get_experiment_info(model)
        parameters = self._fetch_runtime_params(model)

        return {
            "id": model_id,
            "name": model.display_name,
            "console_url": console_url,
            "time_created": str(model.time_created),
            "tags": tags,
            "experiment": self._build_resource_identifier(
                id=experiment_id,
                name=experiment_name,
            ),
            "source": self._build_resource_identifier(
                id=source_model_id, name=source_model_name
            ),
            "parameters": parameters,
        }

    def _build_resource_identifier(
        self, id: str = None, name: str = None
    ) -> AquaResourceIdentifier:
        """Constructs AquaResourceIdentifier based on the given ocid and display name."""
        try:
            resource_type = CONSOLE_LINK_RESOURCE_TYPE_MAPPING.get(
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
            logger.debug(
                f"Failed to construct AquaResourceIdentifier from given id=`{id}`, and name=`{name}`. "
                f"DEBUG INFO: {str(e)}"
            )
            return AquaResourceIdentifier()

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
            resource, EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID
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
                f"Failed to retreive job run: {jobrun_id}. DEBUG INFO: {str(e)}"
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
                    "model parameters have not been saved in correct format in model taxonomy. ",
                    service_payload={"params": params},
                )

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
                f"Failed to get job details from job_run_details: {job_run_details} "
                f"DEBUG INFO:{str(e)}"
            )
            return AquaResourceIdentifier()

    def _get_status(
        self,
        model: oci.resource_search.models.ResourceSummary,
        jobrun: Union[
            oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
        ] = None,
    ) -> dict:
        """Builds evaluation status based on the model status and job run status.
        When missing jobrun information, the status will be decided based on:

            * If the evaluation just has been deleted, the jobrun status should be deleted.
            * When detect `aqua_evaluation_error` in custom metadata, the jobrun is failed.
            * If jobrun failed before saving this meta, we need to check the existance
            of the evaluation artifact.

        """
        model_status = model.lifecycle_state
        job_run_status = None

        if jobrun:
            job_run_status = jobrun.lifecycle_state

        if jobrun is None:
            if model.identifier in self._deletion_cache:
                job_run_status = JobRun.LIFECYCLE_STATE_DELETED

            elif self._get_attribute_from_model_metadata(
                model, EvaluationCustomMetadata.EVALUATION_ERROR
            ):
                job_run_status = JobRun.LIFECYCLE_STATE_FAILED

            elif self._if_eval_artifact_exist(model):
                job_run_status = JobRun.LIFECYCLE_STATE_SUCCEEDED
            else:
                job_run_status = JobRun.LIFECYCLE_STATE_FAILED

        lifecycle_state = utils.LifecycleStatus.get_status(
            evaluation_status=model_status, job_run_status=job_run_status
        )

        try:
            lifecycle_details = (
                LIFECYCLE_DETAILS_MISSING_JOBRUN
                if not jobrun
                else self._extract_job_lifecycle_details(jobrun.lifecycle_details)
            )
        except Exception:
            # ResourceSummary does not have lifecycle_details attr
            lifecycle_details = ""

        return {
            "lifecycle_state": (
                lifecycle_state if isinstance(lifecycle_state, str) else lifecycle_state
            ),
            "lifecycle_details": lifecycle_details,
        }

    def _prefetch_resources(self, compartment_id) -> dict:
        """Fetches all AQUA resources."""
        resources = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="all",
            tag_list=[Tags.AQUA_EVALUATION, "OCI_AQUA"],
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
        'Validation errors in the evaluation config. Exit code: 16.'

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
        except Exception:
            pass

        return message

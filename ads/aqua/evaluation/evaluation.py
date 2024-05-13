#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.evaluation
~~~~~~~~~~~~~~

This module contains AquaEvaluationApp.
"""

import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import List, Union

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
from ads.aqua.common.wrapper import AquaJobRun, AquaModelResource
from ads.aqua.constants import (
    JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING,
    NB_SESSION_IDENTIFIER,
    UNKNOWN,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.enums import DataScienceResource, Resource, Tags
from ads.aqua.evaluation.const import *
from ads.aqua.evaluation.entities import *
from ads.aqua.evaluation.errors import EVALUATION_JOB_EXIT_CODE_MESSAGE
from ads.aqua.exception import (
    AquaFileExistsError,
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.utils import (
    extract_id_and_name_from_tag,
    fire_and_forget,
    get_container_image,
    is_valid_ocid,
    upload_local_to_os,
)
from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import get_console_link, get_files
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


class AquaEvaluationApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua evaluations within the
    Oracle Cloud Infrastructure Data Science service, serving as an interface
    for managing model evalutions.


    Methods
    -------
    create(evaluation_source_id, evaluation_name, ...) -> AquaEvaluationSummary:
        Creates Aqua evaluation for resource.
    get(model_id: str) -> AquaEvaluationDetails:
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
        """Creates Aqua evaluation for resource.

        Parameters
        ----------
        create_aqua_evaluation_details: CreateAquaEvaluationDetails
            The CreateAquaEvaluationDetails data class which contains all
            required and optional fields to create the aqua evaluation.
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
            except:
                raise AquaValueError(
                    "Invalid create evaluation parameters. Allowable parameters are: "
                    f"{', '.join(list(asdict(CreateAquaEvaluationDetails).keys()))}."
                )

        if not is_valid_ocid(create_aqua_evaluation_details.evaluation_source_id):
            raise AquaValueError(
                f"Invalid evaluation source {create_aqua_evaluation_details.evaluation_source_id}. "
                "Specify either a model or model deployment id."
            )

        evaluation_source = None
        if (
            DataScienceResource.MODEL_DEPLOYMENT
            in create_aqua_evaluation_details.evaluation_source_id
        ):
            evaluation_source = ModelDeployment.from_id(
                create_aqua_evaluation_details.evaluation_source_id
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
            except FileExistsError:
                raise AquaFileExistsError(
                    f"Dataset {dataset_file} already exists in {create_aqua_evaluation_details.report_path}. "
                    "Please use a new dataset file name, report path or set `force_overwrite` as True."
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
                if not utils._is_valid_mvs(model_version_set, Tags.AQUA_EVALUATION):
                    raise AquaValueError(
                        f"Invalid experiment name. Please provide an experiment with `{Tags.AQUA_EVALUATION}` in tags."
                    )
            except:
                logger.debug(
                    f"Model version set {experiment_model_version_set_name} doesn't exist. "
                    "Creating new model version set."
                )

                evaluation_mvs_freeform_tags = {
                    Tags.AQUA_EVALUATION: Tags.AQUA_EVALUATION,
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
            EvaluationJobTags.AQUA_EVALUATION: EvaluationJobTags.AQUA_EVALUATION,
            EvaluationJobTags.EVALUATION_MODEL_ID: evaluation_model.id,
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

        self.ds_client.update_model(
            model_id=evaluation_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list,
                freeform_tags={
                    EvaluationModelTags.AQUA_EVALUATION: EvaluationModelTags.AQUA_EVALUATION,
                },
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
            tags=dict(
                aqua_evaluation=EvaluationModelTags.AQUA_EVALUATION,
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
    def _get_service_model_name(
        source: Union[ModelDeployment, DataScienceModel]
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
        if not resource:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )

        aqua_model = AquaModelResource(model=resource)
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        jobrun_id = model_provenance.training_id
        aqua_jobrun = self._fetch_jobrun(resource, use_rqs=False, jobrun_id=jobrun_id)

        summary = AquaEvaluationDetail(
            **self._process(aqua_model),
            **self._get_status(model=aqua_model, jobrun=aqua_jobrun),
            job=aqua_jobrun.to_aquaresourceidentifier(),
            log_group=aqua_jobrun.get_loggroup_identifier(),
            log=aqua_jobrun.get_log_identifier(),
            introspection=aqua_model.introspection,
        )
        summary.parameters.set_shape(aqua_jobrun.shape_name)

        return summary

    @telemetry(entry_point="plugin=evaluation&action=list", name="aqua")
    def list(
        self, compartment_id: str = None, project_id: str = None
    ) -> List[AquaEvaluationSummary]:
        """List Aqua evaluations in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.

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
            tag_list=[EvaluationModelTags.AQUA_EVALUATION],
        )
        aqua_models = [AquaModelResource(model) for model in models]
        logger.info(f"Fetched {len(models)} evaluations.")

        mapping = self._prefetch_resources(compartment_id)

        evaluations = []
        async_tasks = []
        for model in aqua_models:
            if model.id in self._eval_cache.keys():
                logger.debug(f"Retrieving evaluation {model.id} from cache.")
                evaluations.append(self._eval_cache.get(model.id))

            else:
                jobrun_id = model.get(EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID)
                job_run = mapping.get(jobrun_id)

                if not job_run:
                    async_tasks.append((model, jobrun_id))
                else:
                    evaluations.append(
                        self._process_evaluation_summary(
                            model, AquaJobRun(job_run, self.region)
                        )
                    )

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
                        f"Processing evaluation: {model.id} generated an exception: {exc}"
                    )
                    evaluations.append(
                        self._process_evaluation_summary(model=model, jobrun=None)
                    )

        # tracks number of times deployment listing was called
        self.telemetry.record_event_async(category="aqua/evaluation", action="list")

        return evaluations

    def _process_evaluation_summary(
        self,
        model: AquaModelResource,
        jobrun: AquaJobRun,
    ) -> AquaEvaluationSummary:
        """Builds AquaEvaluationSummary from model and jobrun."""

        evaluation_summary = AquaEvaluationSummary(
            **self._process(model),
            **self._get_status(
                model=model,
                jobrun=jobrun,
            ),
            job=jobrun.to_aquaresourceidentifier(),
        )
        # Add evaluation in terminal state into cache
        if evaluation_summary.lifecycle_state in EVAL_TERMINATION_STATE:
            self._eval_cache.__setitem__(key=model.id, value=evaluation_summary)

        return evaluation_summary

    @telemetry(entry_point="plugin=evaluation&action=get_status", name="aqua")
    def get_status(self, eval_id: str) -> AquaEvaluationStatus:
        """Gets evaluation's current status.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        AquaEvaluationStatus
        """

        eval = utils.query_resource(eval_id)

        if not eval:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )
        aqua_model = AquaModelResource(eval)
        model_provenance = self.ds_client.get_model_provenance(eval_id).data

        jobrun_id = model_provenance.training_id
        aqua_jobrun = self._fetch_jobrun(eval, use_rqs=False, jobrun_id=jobrun_id)

        return AquaEvaluationStatus(
            id=eval_id,
            **self._get_status(
                model=aqua_model,
                jobrun=aqua_jobrun,
            ),
            log_id=aqua_jobrun.log_id,
            log_url=aqua_jobrun.log_url,
            loggroup_id=aqua_jobrun.log_group_id,
            loggroup_url=aqua_jobrun.log_group_url,
        )

    def get_supported_metrics(self) -> dict:
        """Gets a list of supported metrics for evaluation."""
        # TODO: implement it when starting to support more metrics.
        return [
            {
                "use_case": ["text_generation"],
                "key": "bertscore",
                "name": "bertscore",
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
                "name": "rouge",
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
            {
                "use_case": ["text_generation"],
                "key": "bleu",
                "name": "bleu",
                "description": (
                    "BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the "
                    "quality of text which has been machine-translated from one natural language to another. "
                    "Quality is considered to be the correspondence between a machine's output and that of a "
                    "human: 'the closer a machine translation is to a professional human translation, "
                    "the better it is'."
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
            md_report_content = self._read_from_artifact(
                temp_dir, files_in_artifact, utils.EVALUATION_REPORT_MD
            )

            # json report not availiable for failed evaluation
            try:
                json_report = json.loads(
                    self._read_from_artifact(
                        temp_dir, files_in_artifact, utils.EVALUATION_REPORT_JSON
                    )
                )
            except Exception as e:
                logger.debug(
                    "Failed to load `report.json` from evaluation artifact" f"{str(e)}"
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

        html_report = AquaEvalReport(
            evaluation_id=eval_id, content=base64.b64encode(content).decode()
        )

        self._report_cache.__setitem__(key=eval_id, value=html_report)

        return html_report

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
        except Exception:
            raise AquaMissingKeyError(
                f"Custom metadata is missing {EvaluationCustomMetadata.EVALUATION_JOB_ID} key"
            )

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

        status = dict(
            id=eval_id,
            lifecycle_state=jobrun.lifecycle_state if jobrun else "DELETING",
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
        """Loads evaluation config."""
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
                    "ocpu": 8,
                    "memory_in_gbs": 128,
                    "block_storage_size": 200,
                },
                "VM.Standard.E4.Flex": {
                    "ocpu": 8,
                    "memory_in_gbs": 128,
                    "block_storage_size": 200,
                },
                "VM.Standard3.Flex": {
                    "ocpu": 8,
                    "memory_in_gbs": 128,
                    "block_storage_size": 200,
                },
                "VM.Optimized3.Flex": {
                    "ocpu": 8,
                    "memory_in_gbs": 128,
                    "block_storage_size": 200,
                },
            },
            "default": {
                "ocpu": 8,
                "memory_in_gbs": 128,
                "block_storage_size": 200,
            },
        }

    def _get_source(
        self,
        evaluation: AquaModelResource,
        resources_mapping: dict = {},
    ) -> tuple:
        """Returns ocid and name of the model has been evaluated."""
        # source_id = evaluation.get_from_meta(EvaluationCustomMetadata.EVALUATION_SOURCE.value)
        source_id = evaluation.get(EvaluationCustomMetadata.EVALUATION_SOURCE)

        try:
            source = resources_mapping.get(source_id)
            source_name = (
                source.display_name
                if source
                else evaluation.get(EvaluationCustomMetadata.EVALUATION_SOURCE_NAME)
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
                f"Failed to retrieve source information for evaluation {evaluation.id}."
                f"DEBUG INFO: {str(ex)}"
            )
            source_name = ""

        return (source_id, source_name)

    def _process(
        self,
        model: AquaModelResource,
        resources_mapping: dict = {},
    ) -> dict:
        """Constructs common fields."""
        source_model_id, source_model_name = self._get_source(model, resources_mapping)

        return dict(
            id=model.id,
            name=model.display_name,
            time_created=model.time_created,
            tags=model.tags,
            experiment=AquaResourceIdentifier.from_data(
                dict(
                    id=model.model_version_set_id,
                    name=model.model_version_set_name,
                    region=self.region,
                )
            ),
            source=AquaResourceIdentifier.from_data(
                dict(id=source_model_id, name=source_model_name, region=self.region)
            ),
            parameters=self._fetch_runtime_params(model),
            region=self.region,
        )

    def _fetch_jobrun(
        self,
        resource: AquaModelResource,
        use_rqs: bool = True,
        jobrun_id: str = None,
    ) -> AquaJobRun:
        """Extracts job run id from metadata, and gets related job run information."""

        jobrun_id = jobrun_id or resource.get(
            EvaluationCustomMetadata.EVALUATION_JOB_RUN_ID
        )

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

        finally:
            return AquaJobRun(jobrun=jobrun, region=self.region)

    def _fetch_runtime_params(self, model: AquaModelResource) -> AquaEvalParams:
        """Extracts model parameters from metadata. Shape is the shape used in job run."""
        try:
            params = json.loads(model.get(MetadataTaxonomyKeys.HYPERPARAMETERS))
            if not params.get(EvaluationConfig.PARAMS):
                raise AquaMissingKeyError(
                    "model parameters have not been saved in correct format in model taxonomy. ",
                    service_payload={"params": params},
                )

            return AquaEvalParams(**params[EvaluationConfig.PARAMS])
        except Exception as e:
            logger.debug(
                f"Failed to retrieve model parameters for the model: {model.id}."
                f"DEBUG INFO: {str(e)}."
            )
            return AquaEvalParams()

    def _get_status(
        self,
        model: AquaModelResource,
        jobrun: AquaJobRun,
    ) -> dict:
        """Builds evaluation status based on the model status and job run status.
        When missing jobrun information, the status will be decided based on:

            * If the evaluation just has been deleted, the jobrun status should be deleted.
            * When detect `aqua_evaluation_error` in custom metadata, the jobrun is failed.
            * If jobrun failed before saving this meta, we need to check the existance
            of the evaluation artifact.

        """
        job_run_status = jobrun.lifecycle_state

        if jobrun.is_missing():
            if model.id in self._deletion_cache.keys():
                job_run_status = JobRun.LIFECYCLE_STATE_DELETED

            elif model.get(EvaluationCustomMetadata.EVALUATION_ERROR):
                job_run_status = JobRun.LIFECYCLE_STATE_FAILED

            elif model.check_artifact_exist(self.ds_client):
                job_run_status = JobRun.LIFECYCLE_STATE_SUCCEEDED
            else:
                job_run_status = JobRun.LIFECYCLE_STATE_FAILED

        lifecycle_state = utils.get_status(
            model_status=model.lifecycle_state, job_run_status=job_run_status
        )

        try:
            lifecycle_details = (
                utils.LIFECYCLE_DETAILS_MISSING_JOBRUN
                if jobrun.is_missing()
                else utils.extract_job_lifecycle_details(
                    jobrun.lifecycle_details, EVALUATION_JOB_EXIT_CODE_MESSAGE
                )
            )
        except:
            # ResourceSummary does not have lifecycle_details attr
            lifecycle_details = ""

        return dict(
            lifecycle_state=lifecycle_state,
            lifecycle_details=lifecycle_details,
        )

    def _prefetch_resources(self, compartment_id) -> dict:
        """Fetches all AQUA resources."""
        # TODO: handle cross compartment/tenency resources
        resources = utils.query_resources(
            compartment_id=compartment_id,
            resource_type="all",
            tag_list=[EvaluationModelTags.AQUA_EVALUATION, "OCI_AQUA"],
            connect_by_ampersands=False,
            return_all=False,
        )
        logger.debug(f"Fetched {len(resources)} AQUA resources.")
        return {item.identifier: item for item in resources}

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
from enum import Enum
import json
from dataclasses import dataclass, field, asdict
import os
from typing import List, Optional
from urllib.parse import urlparse

import fsspec

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaValueError
from ads.aqua.utils import MODEL_PARAMETERS, UNKNOWN, upload_file_to_os
from ads.common import oci_client as oc
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link
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
    ModelTaxonomyMetadata
)
from ads.model.model_version_set import ModelVersionSet

from oci.data_science.models import (
    Metadata, 
    UpdateModelDetails, 
    UpdateModelProvenanceDetails
)

class Resource(Enum):
    JOB="jobs"
    MODEL="models"
    MODEL_DEPLOYMENT="modeldeployments"
    MODEL_VERSION_SET="model-version-sets"

class DataScienceResource(Enum):
    MODEL_DEPLOYMENT="datasciencemodeldeployment"
    MODEL="datasciencemodel"

class EvaluationCustomMetadata(Enum):
    EVALUATION_SOURCE="evaluation_source"
    EVALUATION_JOB_ID="evaluation_job_id"
    EVALUATION_OUTPUT_PATH="evaluation_output_path"
    EVALUATION_SOURCE_NAME="evaluation_source_name"

class EvaluationModelTags(Enum):
    AQUA_EVALUATION="aqua_evaluation"

class EvaluationJobTags(Enum):
    AQUA_EVALUATION="aqua_evaluation"
    EVALUATION_MODEL_ID="evaluation_model_id"

@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str
    name: str
    url: str


@dataclass(repr=False)
class AquaEvalParams(DataClassSerializable):
    shape: str
    max_tokens: str
    top_p: str
    top_k: str
    temperature: str


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

@dataclass(repr=False)
class AquaEvaluation(AquaEvaluationSummary, DataClassSerializable):
    """Represents an Aqua evaluation."""

    parameters: AquaEvalParams = field(default_factory=AquaEvalParams)


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
    compartment_id: Optional[str]=None
    project_id: Optional[str]=None
    evaluation_description: Optional[str]=None
    experiment_id: Optional[str]=None
    experiment_name: Optional[str]=None
    experiment_description: Optional[str]=None
    log_group_id: Optional[str]=None
    log_id: Optional[str]=None
    metrics: Optional[List]=None

# TODO: Remove later
BUCKET_URI = "oci://ming-dev@ociodscdev/evaluation/sample_response"
SOURCE="oci://lu_bucket@ociodscdev/evaluation_dummy_script.py"
SUBNET_ID=os.environ.get("SUBNET_ID", None)


class AquaEvaluationApp(AquaApp):
    """Contains APIs for Aqua model evaluation.


    Methods
    -------
    create(evaluation_source_id, evaluation_name, ...) -> AquaEvaluationSummary:
        Creates Aqua evaluation for resource.
    get(model_id: str) -> AquaModel:
        Retrieves details of an Aqua evaluation by its unique identifier.
    list(compartment_id: str = None, project_id: str = None, **kwargs) -> List[AquaEvaluation]:
        Lists all Aqua evaluation within a specified compartment and/or project.

    Note:
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    def create(
        self,
        create_aqua_evaluation_details: CreateAquaEvaluationDetails,
        **kwargs
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
        
        if not ObjectStorageDetails.is_oci_path(create_aqua_evaluation_details.report_path):
            raise AquaValueError("Evaluation report path must be an object storage path.")
        
        evaluation_model_parameters = None
        try:
            evaluation_model_parameters = AquaEvalParams(
                shape=create_aqua_evaluation_details.shape_name,
                **create_aqua_evaluation_details.model_parameters
            )
        except:
            raise AquaValueError(
               "Invalid model parameters. Model parameters should "
               f"be a dictionary with keys: {', '.join(MODEL_PARAMETERS)}."
            )

        target_compartment = (
            create_aqua_evaluation_details.compartment_id or COMPARTMENT_OCID
        )
        target_project = (
            create_aqua_evaluation_details.project_id or PROJECT_OCID
        )

        experiment_model_version_set_id = create_aqua_evaluation_details.experiment_id
        experiment_model_version_set_name = create_aqua_evaluation_details.experiment_name

        if (
            not experiment_model_version_set_id and 
            not experiment_model_version_set_name
        ):
            raise AquaValueError(
                "Either experiment id or experiment name must be provided."
            )

        if not experiment_model_version_set_id:
            try:
                model_version_set = ModelVersionSet.from_name(
                    name=experiment_model_version_set_name,
                    compartment_id=target_compartment
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
            value=create_aqua_evaluation_details.report_path
        )
        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_SOURCE_NAME.value,
            value=evaluation_source.display_name
        )

        evaluation_model_taxonomy_metadata = ModelTaxonomyMetadata()
        evaluation_model_taxonomy_metadata[
            MetadataTaxonomyKeys.HYPERPARAMETERS
        ].value={
            "model_params": {
                key: value for key, value in 
                asdict(evaluation_model_parameters).items()
            }
        }

        evaluation_model_freeform_tags = {
            EvaluationModelTags.AQUA_EVALUATION.value: EvaluationModelTags.AQUA_EVALUATION.value,
        }

        evaluation_model = (
            DataScienceModel()
            .with_compartment_id(target_compartment)
            .with_project_id(target_project)
            .with_display_name(
                create_aqua_evaluation_details.evaluation_name
            )
            .with_description(
                create_aqua_evaluation_details.evaluation_description
            )
            .with_model_version_set_id(experiment_model_version_set_id)
            .with_custom_metadata_list(evaluation_model_custom_metadata)
            .with_defined_metadata_list(evaluation_model_taxonomy_metadata)
            .with_provenance_metadata(
                ModelProvenanceMetadata(
                    training_id=UNKNOWN
                )
            )
            .with_freeform_tags(**evaluation_model_freeform_tags)
            # TODO: decide what parameters will be needed
            .create(
                remove_existing_artifact=False, # TODO: added here for the puopose of demo and will revisit later
                **kwargs
            )
        )
        logger.debug(
            f"Successfully created evaluation model {evaluation_model.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_dataset_path = create_aqua_evaluation_details.dataset_path
        if not ObjectStorageDetails.is_oci_path(evaluation_dataset_path):
            # format: oci://<bucket>@<namespace>/<evaluation_model_id>/<dataset_file_name>
            dst_uri = (
                f"{create_aqua_evaluation_details.report_path}/{evaluation_model.id}/{os.path.basename(evaluation_dataset_path)}"
            )
            upload_file_to_os(
                src_uri=evaluation_dataset_path,
                dst_uri=dst_uri,
                auth=self._auth,
                force_overwrite=True
            )
            logger.debug(
                f"Uploaded local file {evaluation_dataset_path} to object storage {dst_uri}."
            )
            evaluation_dataset_path = dst_uri            

        # TODO: validat metrics if it's provided

        evaluation_job_freeform_tags = {
            EvaluationJobTags.AQUA_EVALUATION.value: EvaluationJobTags.AQUA_EVALUATION.value,
            EvaluationJobTags.EVALUATION_MODEL_ID.value: evaluation_model.id
        }

        evaluation_job = (
            Job(name=evaluation_model.display_name)
            .with_infrastructure(
                DataScienceJob()
                .with_log_group_id(
                    create_aqua_evaluation_details.log_group_id
                )
                .with_log_id(
                    create_aqua_evaluation_details.log_id
                )
                .with_compartment_id(target_compartment)
                .with_project_id(target_project)
                .with_shape_name(
                    create_aqua_evaluation_details.shape_name
                )
                .with_shape_config_details(
                    memory_in_gbs=create_aqua_evaluation_details.memory_in_gbs,
                    ocpus=create_aqua_evaluation_details.ocpus
                )
                .with_block_storage_size(
                    create_aqua_evaluation_details.block_storage_size
                )
                .with_freeform_tag(
                    **evaluation_job_freeform_tags
                )
                .with_subnet_id(SUBNET_ID)
            )
            .with_runtime(
                self._build_evaluation_runtime(
                    dataset_path=evaluation_dataset_path,
                    report_path=create_aqua_evaluation_details.report_path,
                    model_parameters=create_aqua_evaluation_details.model_parameters,
                    metrics=create_aqua_evaluation_details.metrics
                )
            )
            .create(
                **kwargs ## TODO: decide what parameters will be needed
            )
        )
        logger.debug(
            f"Successfully created evaluation job {evaluation_job.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )

        evaluation_model_custom_metadata.add(
            key=EvaluationCustomMetadata.EVALUATION_JOB_ID.value,
            value=evaluation_job.id
        )
        updated_custom_metadata_list = [
            Metadata(**metadata) for metadata in 
            evaluation_model_custom_metadata.to_dict()["data"]
        ]

        self.ds_client.update_model(
            model_id=evaluation_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list
            )
        )

        evaluation_job_run = evaluation_job.run(
            name=evaluation_model.display_name,
            freeform_tags=evaluation_job_freeform_tags,
            wait=False
        )
        logger.debug(
            f"Successfully created evaluation job run {evaluation_job_run.id} for {create_aqua_evaluation_details.evaluation_source_id}."
        )
        
        self.ds_client.update_model_provenance(
            model_id=evaluation_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=evaluation_job_run.id
            )
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
            lifecycle_state=evaluation_job_run.lifecycle_state,
            lifecycle_details=evaluation_job_run.lifecycle_details,
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
                evaluation_experiment_id=experiment_model_version_set_id
            )
        )
    
    def _build_evaluation_runtime(
        self,
        dataset_path: str,
        report_path: str,
        model_parameters: dict,
        metrics: List=None,
        **kwargs
    ) -> Runtime:
        """Builds evaluation runtime for Job."""
        # TODO: update the logic to evaluate the model or model deployment
        runtime = (
            PythonRuntime()
            .with_service_conda("pytorch21_p39_gpu_v1")
            .with_source(SOURCE)
        )

        return runtime

    def get(self, eval_id) -> AquaEvaluation:
        """Gets the information of an Aqua evalution.

        Parameters
        ----------
        eval_id: str
            The model OCID.

        Returns
        -------
        AquaEvaluation:
            The instance of AquaEvaluation.
        """
        # Mock response
        response_file = f"{BUCKET_URI}/get.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            model = json.load(f)
        return AquaEvaluation(**model)

    def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List[AquaEvaluation]:
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
        List[AquaEvaluation]:
            The list of the `ads.aqua.evalution.AquaEvaluation`.
        """
        # Mock response
        response_file = f"{BUCKET_URI}/list.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            models = json.load(f)

        evaluations = []
        for model in models:
            evaluations.append(AquaEvaluation(**model))
        return evaluations

    def get_status(self, eval_id: str) -> dict:
        return {
            "id": eval_id,
            "lifecycle_state": "ACTIVE",
            "lifecycle_details": "This is explanation for lifecycle_state.",
        }

    def load_metrics(self, eval_id: str) -> dict:
        """Loads evalution metrics markdown from artifacts.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        dict:
            A dictionary contains default model parameters.
        """
        # Mock response
        response_file = f"{BUCKET_URI}/metrics.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            metrics = json.load(f)
        return AquaEvalMetrics(
            id=eval_id, metrics=[AquaEvalMetric(**m) for m in metrics]
        )

    def download_report(self, eval_id) -> dict:
        """Downloads HTML report from model artifact.

        Parameters
        ----------
        eval_id: str
            The evaluation ocid.

        Returns
        -------
        dict:
            A dictionary contains json response.
        """
        # Mock response
        os_uri = (
            "oci://license_checker@ociodscdev/evaluation/report/evaluation_report.html"
        )

        p = urlparse(os_uri)
        os_client = oc.OCIClientFactory(**self._auth).object_storage
        res = os_client.get_object(
            namespace_name=p.hostname, bucket_name=p.username, object_name=p.path[1:]
        )

        content = res.data.raw.read()
        return dict(evaluation_id=eval_id, content=base64.b64encode(content).decode())

    def get_supported_metrics(self) -> list:
        """Lists supported metrics."""

        # TODO: implementation
        return [
            {
                "use_case": ["one", "two", "three"],
                "key": "bertscore",
                "name": "BERT Score",
                "description": "BERTScore computes the semantic similarity between two pieces of text using BERT embeddings.",
                "args": {},
            },
        ]

    def load_evaluation_config(self, model_id: str) -> dict:
        """Loads `evaluation_config.json` stored in artifact."""
        # TODO: Implementation
        logger.info(f"Loading evaluation config for model: {model_id}")
        return {
            "model_params": {
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 50,
            },
            "shape": {
                "BM.A10.2": {
                    "count": 1,
                    "gpu_memory": 0.8,
                    "tensor_parallel": 1,
                    "enforce_eager": 3,
                    "max_model_len": 2048,
                },
                "VM.A10.2": {
                    "count": 1,
                    "gpu_memory": 0.8,
                    "tensor_parallel": 1,
                    "enforce_eager": 3,
                    "max_model_len": 2048,
                },
            },
        }

    def _upload_data(self, src_uri, dst_uri):
        """Uploads data file from notebook session to object storage."""
        # This method will be called in create()
        # if src is os : pass to script
        # else: copy to dst_uri then pass dst_uri to script
        pass

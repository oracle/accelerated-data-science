#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import oci

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaError, AquaMissingKeyError, AquaRuntimeError
from ads.common import oci_client as oc
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link, get_files
from ads.model.datascience_model import DataScienceModel


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    console_url: str = ""


@dataclass(repr=False)
class AquaEvalParams(DataClassSerializable):
    shape: str = ""
    max_tokens: str = ""
    top_p: str = ""
    top_k: str = ""
    temperature: str = ""


@dataclass(repr=False)
class AquaEvalMetric(DataClassSerializable):
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


class EvaluationConfig:
    PARAMS = "model_params"


class AquaEvaluationApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua evaluations within the
    Oracle Cloud Infrastructure Data Science service, serving as an interface
    for managing model evalutions.


    Methods
    -------
    get(model_id: str) -> AquaEvaluationSummary:
        Retrieves details of an Aqua evaluation by its unique identifier.
    list(compartment_id: str = None, project_id: str = None, **kwargs) -> List[AquaEvaluationSummary]:
        Lists all Aqua evaluation within a specified compartment and/or project.

    Note:
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    def create(self):
        return {
            "id": "ocid1.datasciencemodel.<OCID>",
            "name": "test_evaluation",
            "console_url": "https://cloud.oracle.com/data-science/models/ocid1.datasciencemodel.<OCID>?region=us-ashburn-1",
            "lifecycle_state": "ACCEPTED",
            "lifecycle_details": "TODO",
            "time_created": "2024-02-21 21:06:11.444000+00:00",
            "experiment": {
                "id": "ocid1.datasciencemodelversionset.<OCID>",
                "name": "test_43210123456789012",
                "console_url": "https://cloud.oracle.com/data-science/model-version-sets/ocid1.datasciencemodelversionset.<OCID>?region=us-ashburn-1",
            },
            "source": {
                "id": "ocid1.datasciencemodel.<OCID>",
                "name": "Mistral-7B-Instruct-v0.1-Fine-Tuned",
                "console_url": "https://cloud.oracle.com/data-science/models/ocid1.datasciencemodel.<OCID>?region=us-ashburn-1",
            },
            "job": {
                "id": "ocid1.datasciencejob.<OCID>",
                "name": "test_evaluation",
                "console_url": "https://cloud.oracle.com/data-science/jobs/ocid1.datasciencejob.<OCID>?region=us-ashburn-1",
            },
            "tags": {
                "aqua_evaluation": "aqua_evaluation",
            },
        }

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

        resource = DataScienceModel.from_id(eval_id)

        # resource = utils.query_resource(eval_id)
        # resource = self.ds_client.get_model(eval_id).data
        # Need separate call to get model provenance: get_model_provenance(eval_id)

        if not resource:
            raise AquaRuntimeError(
                f"Failed to retrieve evalution {eval_id}."
                "Please check if the OCID is correct."
            )

        job_run_details = self._fetch_jobrun(resource, use_rqs=False)
        shape = (
            job_run_details.job_infrastructure_configuration_details.shape_name
            if job_run_details
            else utils.UNKNOWN
        )

        return AquaEvaluationSummary(
            **self._process(resource, shape),
            **self._get_status(
                resource.status, job_status=job_run_details.lifecycle_state
            ),
            job=self._build_job_identifier(
                job_run_details=job_run_details,
            ),
        )

    def _get_status(self, model_status, job_status) -> dict:
        """Build evaluation status based on the model status and job run status."""
        lifecycle_state = utils.LifecycleStatus.get_status(model_status, job_status)
        return dict(
            lifecycle_state=lifecycle_state.value,
            lifecycle_details=lifecycle_state.detail,
        )

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
            status_list=["ACTIVE"],
        )
        logger.info(f"Fetched {len(models)} evaluations.")

        # TODO: add filter based on project_id if needed.

        evaluations = []
        for model in models:
            job_run = self._fetch_jobrun(model)
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
        return {
            "id": eval_id,
            "lifecycle_state": "ACTIVE",
            "lifecycle_details": "This is explanation for lifecycle_state.",
        }

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
        # TODO: add caching
        with tempfile.TemporaryDirectory() as temp_dir:
            DataScienceModel.from_id(eval_id).download_artifact(
                temp_dir, auth=self._auth
            )
            metrics = []
            for file in get_files(temp_dir):
                if file.name.endswith(".md"):
                    with open(file, "rb") as f:
                        content = f.read()
                    metrics.append(
                        AquaEvalMetric(
                            name=f.name, content=base64.b64encode(content).decode()
                        )
                    )
        return AquaEvalMetrics(id=eval_id, metrics=metrics)

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
        # TODO: add caching
        with tempfile.TemporaryDirectory() as temp_dir:
            DataScienceModel.from_id(eval_id).download_artifact(
                temp_dir,
                auth=self._auth,
            )
            for file in get_files(temp_dir):
                if file == "report.zip":
                    with ZipFile(zip_file_path) as zip_file:
                        zip_file.extractall(self.target_dir)

        return dict(evaluation_id=eval_id, content=base64.b64encode(content).decode())

    def _get_source_id(
        self,
        model: Union[
            "oci.resource_search.models.ResourceSummary",
            "ads.model.datascience_model.DataScienceModel",
        ],
    ) -> str:
        try:
            return (
                self._extract_metadata(
                    model.additional_details.get(RqsAdditionalDetails.METADATA),
                    EvaluationMetadata.EVALUATION_SOURCE,
                )
                if isinstance(model, oci.resource_search.models.ResourceSummary)
                else model.custom_metadata_list.get(
                    EvaluationMetadata.EVALUATION_SOURCE
                ).value
            )
        except:
            raise AquaMissingKeyError(
                f"Missing `evaluation_source` in custom metadata of model."
            )

    def _get_model_id(
        self,
        model: Union[
            "oci.resource_search.models.ResourceSummary",
            "ads.model.datascience_model.DataScienceModel",
        ],
    ) -> str:
        return (
            model.identifier
            if isinstance(model, oci.resource_search.models.ResourceSummary)
            else model.id
        )

    def _get_experiment_info(
        self,
        model: Union[
            "oci.resource_search.models.ResourceSummary",
            "ads.model.datascience_model.DataScienceModel",
        ],
    ) -> tuple:
        return (
            (
                model.additional_details.get(RqsAdditionalDetails.MODEL_VERSION_SET_ID),
                model.additional_details.get(
                    RqsAdditionalDetails.MODEL_VERSION_SET_NAME
                ),
            )
            if isinstance(model, oci.resource_search.models.ResourceSummary)
            else (model.model_version_set_id, model.model_version_set_name)
        )

    def _get_source_info(
        self,
        model: Union[
            "oci.resource_search.models.ResourceSummary",
            "ads.model.datascience_model.DataScienceModel",
        ],
        model_id: str,
    ) -> tuple:
        source_model_id = utils.UNKNOWN
        source_model_name = utils.UNKNOWN
        try:
            source_model_id = self._get_source_id(model)
            logger.info(f"Fetching source model {source_model_id} info.")

            source_model = utils.query_resource(source_model_id, return_all=False)
            source_model_name = source_model.display_name
        except Exception as e:
            logger.debug(f"{str(e)}: {model_id}.")
        return (source_model_id, source_model_name)

    def _process(
        self,
        model: Union[
            "oci.resource_search.models.ResourceSummary",
            "ads.model.datascience_model.DataScienceModel",
        ],
        shape: str = None,
    ) -> dict:
        """Constructs AquaEvaluationSummary from `oci.resource_search.models.ResourceSummary`
        or `ads.model.datascience_model.DataScienceModel`.
        """

        tags = {}
        tags.update(model.defined_tags or {})
        tags.update(model.freeform_tags or {})

        model_id = self._get_model_id(model)
        console_url = get_console_link(
            resource="models",
            ocid=model_id,
            region=self.region,
        )
        source_model_id, source_model_name = self._get_source_info(model, model_id)
        experiment_id, experiment_name = self._get_experiment_info(model)
        parameters = self._fetch_runtime_params(model, shape)

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
                console_url=get_console_link(
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

    def _fetch_jobrun(
        self,
        resource: Union[
            oci.resource_search.models.ResourceSummary,
            "ads.model.datascience_model.DataScienceModel",
        ],
        use_rqs: bool = True,
    ) -> Union[
        oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
    ]:
        """Extracts job run id from metadata, and gets related job run information."""

        if isinstance(resource, DataScienceModel):
            try:
                jobrun_id = resource.provenance_metadata.training_id
            except:
                logger.debug(
                    f"Resource {resource.id} missing job run information in model provenance: {resource.provenance_metadata}."
                )
        elif isinstance(resource, oci.resource_search.models.ResourceSummary):
            try:
                metadata = resource.additional_details.get(
                    RqsAdditionalDetails.METADATA
                )
                jobrun_id = self._extract_metadata(
                    metadata, EvaluationMetadata.EVALUATION_JOB_ID
                )
            except:
                logger.debug(
                    f"Resource {resource.identifier} missing job run information in custom metadata: {metadata}."
                )
        else:
            logger.error(f"{type(resource)} is not valid.")
            raise AquaError(f"{type(resource)} is not valid.", status=500)

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

    def _fetch_metrics(self, resource) -> AquaEvalMetrics:
        """Extracts evaluation's metrics from metadata."""
        metadata = resource.additional_details.get(RqsAdditionalDetails.METADATA)

        all_metrics = json.loads(
            self._extract_metadata(metadata, EvaluationMetadata.ARTFACTTESTRESULTS)
        )

        metrics = []
        for k, v in all_metrics.items():
            # TODO: add description for different metric. The description below is for bertscore.
            metrics.append(
                AquaEvalMetric(
                    name=k,
                    description="Semantic similarity between tokens of reference and hypothesis",
                    value=v,
                )
            )
        return AquaEvalMetrics(data=metrics)

    def _get_hyperparameters(self, model):
        params = ""
        if isinstance(model, DataScienceModel):
            params = model.defined_metadata_list.get(
                EvaluationMetadata.HYPERPARAMETERS
            ).value

        else:
            metadata = model.additional_details.get(RqsAdditionalDetails.METADATA)
            params = json.loads(
                self._extract_metadata(metadata, EvaluationMetadata.HYPERPARAMETERS)
            )
        return params

    def _fetch_runtime_params(self, resource, shape: str = None) -> AquaEvalParams:
        """Extracts model parameters from metadata. Shape is the shape used in job run."""
        try:
            params = self._get_hyperparameters(resource)
            # TODO: validate the format of parameters.
            # self._validate_params(params)

            return AquaEvalParams(
                **params[EvaluationConfig.PARAMS],
                shape=shape,
            )
        except Exception as e:
            logger.debug(
                f"Failed to retrieve model parameters for the model: {str(resource)}."
                f"DEBUG INFO: {str(e)}."
            )
            return AquaEvalParams(shape=shape)

    def _extract_metadata(self, metadata_list: List[Dict], key: str) -> Any:
        for metadata in metadata_list:
            if metadata.get("key") == key:
                return metadata.get("value")
        logger.error(f"Missing target key: {key} in metadata {metadata_list}.")
        return ""

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

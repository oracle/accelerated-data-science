#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import oci

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaRuntimeError
from ads.common import oci_client as oc
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    console_url: str = ""


@dataclass(repr=False)
class AquaEvalParams(DataClassSerializable):
    shape: str
    max_tokens: str
    top_p: str
    top_k: str
    temperature: str


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

        resource = utils.query_resource(eval_id)

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

        return AquaEvaluationDetails(
            **self._process(resource),
            **self._get_job_details(
                job_run_details=job_run_details,
                evaluation_status=resource.lifecycle_state,
            ),
            parameters=self._fetch_runtime_params(resource, shape),
            # metrics=self._fetch_metrics(resource),
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
        )
        logger.info(f"Fetched {len(models)} evaluations.")

        # TODO: add filter based on project_id if needed.

        evaluations = []
        for model in models:
            job_run = self._fetch_jobrun(model)

            evaluations.append(
                AquaEvaluationSummary(
                    **self._process(model),
                    **self._get_job_details(
                        job_run_details=job_run, evaluation_status=model.lifecycle_state
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
        # TODO: add caching
        pass

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

    def _process(self, model: "oci.resource_search.models.ResourceSummary") -> dict:
        """Constructs AquaEvaluationSummary from `oci.resource_search.models.ResourceSummary`."""

        tags = {}
        tags.update(model.defined_tags or {})
        tags.update(model.freeform_tags or {})

        # TODO: discuss if we want to use metadata/tags to save source model name
        source_model_id = self._extract_metadata(
            model.additional_details.get(RqsAdditionalDetails.METADATA),
            EvaluationMetadata.EVALUATION_SOURCE,
        )
        logger.info(f"Source model ocid: {source_model_id}.")

        try:
            source_model = utils.query_resource(source_model_id, return_all=False)
            source_model_name = source_model.display_name
        except:
            source_model_name = ""

        return dict(
            id=model.identifier,
            name=model.display_name,
            console_url=get_console_link(
                resource="models",
                ocid=model.identifier,
                region=self.region,
            ),
            time_created=model.time_created,
            tags=tags,
            experiment=self._get_resource_identifier(
                id=model.additional_details.get(
                    RqsAdditionalDetails.MODEL_VERSION_SET_ID
                ),
                name=model.additional_details.get(
                    RqsAdditionalDetails.MODEL_VERSION_SET_NAME
                ),
            ),
            source=self._get_resource_identifier(
                id=source_model_id, name=source_model_name
            ),
        )

    def _get_resource_identifier(
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
        self, resource: oci.resource_search.models.ResourceSummary, use_rqs: bool = True
    ) -> Union[
        oci.resource_search.models.ResourceSummary, oci.data_science.models.JobRun
    ]:
        """Extracts job run id from freeform tags, and gets related job run information."""

        jobrun_id = resource.freeform_tags.get(EvaluationTags.EVALUATION_JOB)
        if not jobrun_id:
            logger.error(
                f"Resource {resource.identifier} missing job run key in tags: {str(resource.freeform_tags)}."
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

    def _fetch_runtime_params(self, resource, shape: str = None) -> AquaEvalParams:
        """Extracts runtime parameters from metadata. Shape is the shape used in job run."""
        metadata = resource.additional_details.get(RqsAdditionalDetails.METADATA)
        params = json.loads(
            self._extract_metadata(metadata, EvaluationMetadata.HYPERPARAMETERS)
        )
        # TODO: validate the format of parameters.
        # self._validate_params(params)
        return AquaEvalParams(
            **params[EvaluationConfig.PARAMS],
            **params[EvaluationConfig.CONFIG],
            shape=shape,
        )

    def _extract_metadata(self, metadata_list: List[Dict], key: str) -> Any:
        for metadata in metadata_list:
            if metadata.get("key") == key:
                return metadata.get("value")
        logger.error(f"Missing target key: {key} in metadata {metadata_list}.")
        return ""

    def _get_job_details(
        self,
        job_run_details: Union[
            oci.data_science.models.JobRun, oci.resource_search.models.ResourceSummary
        ] = None,
        evaluation_status: str = None,
    ) -> dict:
        try:
            lifecycle_state = utils.LifecycleStatus.get_status(
                evaluation_status, job_run_details.lifecycle_state
            )
            job_id = (
                job_run_details.id
                if isinstance(job_run_details, oci.data_science.models.JobRun)
                else job_run_details.identifier
            )
            return dict(
                lifecycle_state=lifecycle_state.value,
                lifecycle_details=lifecycle_state.detail,
                job=self._get_resource_identifier(
                    id=job_id, name=job_run_details.display_name
                ),
            )
        except Exception as e:
            logger.debug(
                f"Failed to get job details from job_run_details: {job_run_details}"
                f"DEBUG INFO:{str(e)}"
            )
            return dict(
                lifecycle_state=utils.UNKNOWN,
                lifecycle_details=utils.UNKNOWN,
                job=AquaResourceIdentifier(),
            )

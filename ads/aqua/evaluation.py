#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Union
from zipfile import ZipFile

import oci
from cachetools import TTLCache

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.exception import (
    AquaFileNotFoundError,
    AquaMissingKeyError,
    AquaRuntimeError,
)
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link, get_files
from ads.model.datascience_model import DataScienceModel


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    id: str = ""
    name: str = ""
    console_url: str = ""


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

    _report_cache = TTLCache(maxsize=10, ttl=timedelta(hours=5), timer=datetime.now)
    _cache_lock = Lock()

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

        # resource = DataScienceModel.from_id(eval_id)

        resource = utils.query_resource(eval_id)
        model_provenance = self.ds_client.get_model_provenance(eval_id).data
        # resource = self.ds_client.get_model(eval_id).data
        # Need separate call to get model provenance: get_model_provenance(eval_id)

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
        # WIP
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
            report_zip_name = ""
            for file in get_files(temp_dir):
                # report will be a zip archive in model artifact
                if file.endswith(".zip"):
                    zip_file_path = os.path.join(temp_dir, file)
                    with ZipFile(zip_file_path) as zip_file:
                        zip_file.extractall(temp_dir)
                    report_zip_name = file
                    break

            try:
                report_path = os.path.join(temp_dir, utils.EVALUATION_REPORT)
                with open(report_path, "rb") as f:
                    content = f.read()
            except FileNotFoundError as e:
                error_msg = "Related Resource Not Authorized Or Not Found:" + (
                    (
                        f"Found report zip in evaluation artifact: `{report_zip_name}`."
                        f"Expected zip name is `{utils.EVALUATION_REPORT_ZIP}`."
                    )
                    if report_zip_name
                    else f"Missing `{utils.EVALUATION_REPORT_ZIP}` in evaluation artifact."
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

    def _get_jobrun(
        self, model: oci.resource_search.models.ResourceSummary, mapping: dict
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
        lifecycle_state = utils.LifecycleStatus.get_status(model_status, job_status)
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

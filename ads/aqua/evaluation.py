#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import base64
import json
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse

import fsspec

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.common import oci_client as oc
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


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
    parameters: AquaEvalParams = field(default_factory=AquaEvalParams)


# TODO: Remove later
BUCKET_URI = "oci://ming-dev@ociodscdev/evaluation/sample_response"


class AquaEvaluationApp(AquaApp):
    """Contains APIs for Aqua model evaluation.


    Methods
    -------
    get(model_id: str) -> AquaModel:
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
        # Mock response
        response_file = f"{BUCKET_URI}/get.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            model = json.load(f)
        return AquaEvaluationSummary(**model)

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
        # Mock response
        response_file = f"{BUCKET_URI}/list.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            models = json.load(f)

        evaluations = []
        for model in models:
            evaluations.append(AquaEvaluationSummary(**model))
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

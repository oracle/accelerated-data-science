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
    gpu_memory: str


@dataclass(repr=False)
class AquaEvalMetric(DataClassSerializable):
    name: str
    description: str
    value: dict


@dataclass(repr=False)
class AquaEvaluationSummary(DataClassSerializable):
    """Represents a summary of Aqua evalution."""

    id: str
    name: str
    console_url: str
    lifecycle_state: str
    lifecycle_details: str
    time_created: str
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    tags: dict


@dataclass(repr=False)
class AquaEvaluationDetails(AquaEvaluationSummary):
    """Represents a detail of Aqua evalution."""

    parameters: AquaEvalParams
    metrics: List[dict] = field(default_factory=list)


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

    def get(self, eval_id) -> AquaEvaluationDetails:
        """Gets the information of an Aqua evalution.

        Parameters
        ----------
        eval_id: str
            The model OCID.

        Returns
        -------
        AquaEvaluationDetails:
            The instance of AquaEvaluationDetails.
        """
        # Mock response
        response_file = f"{BUCKET_URI}/get.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            model = json.load(f)
        return AquaEvaluationDetails(**model)

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

        # real implementation
        # models = self._rqs_models(compartment_id)
        # evaluations = []
        # for model in models:
        #     evaluations.append(AquaEvaluationSummary(**self._process(model)))
        # return evaluations

    def load_params(self, model_id: str) -> dict:
        """Loads default params from `evaluation_config.json` in model artifact.

        Parameters
        ----------
        model_id: str
            The source model ocid. It can be modeldeplyment or model.

        Returns
        -------
        dict:
            A dictionary contains default model parameters.
        """
        # Mock response
        response_file = f"{BUCKET_URI}/param.json"
        logger.info(f"Loading mock response from {response_file}.")
        with fsspec.open(response_file, "r", **self._auth) as f:
            default_params = json.load(f)
        return default_params

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

    def _upload_data(self, src_uri, dst_uri):
        """Uploads data file from notebook session to object storage."""
        # This method will be called in create()
        # if src is os : pass to script
        # else: copy to dst_uri then pass dst_uri to script
        pass

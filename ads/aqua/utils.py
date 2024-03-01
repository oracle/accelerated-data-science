#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA utils and constants."""
import base64
import json
import logging
import os
import random
import re
import sys
from enum import Enum
from pathlib import Path
from string import Template
from typing import List

import fsspec
from oci.data_science.models import JobRun, Model

from ads.aqua.exception import AquaError, AquaFileNotFoundError, AquaRuntimeError, AquaValueError
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import upload_to_os
from ads.config import TENANCY_OCID

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ODSC_AQUA")

UNKNOWN = ""
README = "README.md"
DEPLOYMENT_CONFIG = "deployment_config.json"
EVALUATION_REPORT_JSON = "report.json"
EVALUATION_REPORT = "report.html"
UNKNOWN_JSON_STR = "{}"
CONSOLE_LINK_RESOURCE_TYPE_MAPPING = dict(
    datasciencemodel="models",
    datasciencemodeldeployment="model-deployments",
    datasciencejob="jobs",
)
CONDA_BUCKET_NS = os.environ.get("CONDA_BUCKET_NS", "ociodscdev")
SOURCE_FILE = "run.sh"
CONDA_URI = f"oci://ads-evaluation@{CONDA_BUCKET_NS}/conda_environments/gpu/PyTorch 2.1 for GPU on Python 3.9/1.0/pytorch21_p39_gpu_v1"
CONDA_REGION = "us-ashburn-1"
BERT_SCORE_PATH = "/home/datascience/conda/pytorch21_p39_gpu_v1/bertscore/bertscore.py"
BERT_BASE_MULTILINGUAL_CASED = (
    "/home/datascience/conda/pytorch21_p39_gpu_v1/bert-base-multilingual-cased/"
)
DEFAULT_MAX_TOKEN = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
DEFAULT_BLOCK_STORAGE_SIZE = 100
DEFAULT_MEMORY_IN_GBS = 32
DEFAULT_OCPUS = 2

DEFAULT_MODEL_PARAMS_CONFIGS = {
    "model_params": {
        "max_tokens": DEFAULT_MAX_TOKEN,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
    },
    "default": {
        "ocpus": DEFAULT_OCPUS,
        "memory_in_gbs": DEFAULT_MEMORY_IN_GBS,
        "block_storage_size": DEFAULT_BLOCK_STORAGE_SIZE,
    },
}

# TODO: remove later
SUBNET_ID = os.environ.get("SUBNET_ID", None)


class LifecycleStatus(Enum):
    UNKNOWN = ""

    @property
    def detail(self) -> str:
        """Returns the detail message corresponding to the status."""
        return LIFECYCLE_DETAILS_MAPPING.get(
            self.name, f"No detail available for the status {self.name}."
        )

    @staticmethod
    def get_status(evaluation_status: str, job_run_status: str = None):
        """
        Maps the combination of evaluation status and job run status to a standard status.

        Parameters
        ----------
        evaluation_status (str):
            The status of the evaluation.
        job_run_status (str):
            The status of the job run.

        Returns
        -------
        LifecycleStatus
            The mapped status ("Completed", "In Progress", "Canceled").
        """
        if not job_run_status:
            logger.error("Failed to get jobrun state.")
            # case1 : failed to create jobrun
            # case2: jobrun is deleted - rqs cannot retreive deleted resource
            return JobRun.LIFECYCLE_STATE_NEEDS_ATTENTION

        status = LifecycleStatus.UNKNOWN
        if evaluation_status == Model.LIFECYCLE_STATE_ACTIVE:
            if (
                job_run_status == JobRun.LIFECYCLE_STATE_IN_PROGRESS
                or job_run_status == JobRun.LIFECYCLE_STATE_ACCEPTED
            ):
                status = JobRun.LIFECYCLE_STATE_IN_PROGRESS
            elif (
                job_run_status == JobRun.LIFECYCLE_STATE_FAILED
                or job_run_status == JobRun.LIFECYCLE_STATE_NEEDS_ATTENTION
            ):
                status = JobRun.LIFECYCLE_STATE_FAILED
            else:
                status = job_run_status
        else:
            status = evaluation_status

        return status


LIFECYCLE_DETAILS_MAPPING = {
    JobRun.LIFECYCLE_STATE_SUCCEEDED: "The evaluation ran successfully.",
    JobRun.LIFECYCLE_STATE_IN_PROGRESS: "The evaluation is running.",
    JobRun.LIFECYCLE_STATE_FAILED: "The evaluation failed.",
    JobRun.LIFECYCLE_STATE_NEEDS_ATTENTION: "Missing jobrun information.",
}
SUPPORTED_FILE_FORMATS = ["jsonl"]
MODEL_BY_REFERENCE_OSS_PATH_KEY = "Object Storage Path"
MODEL_PARAMETERS = ["max_tokens", "temperature", "top_p", "top_k"]


def get_logger():
    return logger


def random_color_generator(word: str):
    seed = sum([ord(c) for c in word]) % 13
    random.seed(seed)
    r = random.randint(10, 245)
    g = random.randint(10, 245)
    b = random.randint(10, 245)

    text_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.5 else "white"

    return f"#{r:02x}{g:02x}{b:02x}", text_color


def svg_to_base64_datauri(svg_contents: str):
    base64_encoded_svg_contents = base64.b64encode(svg_contents.encode())
    return "data:image/svg+xml;base64," + base64_encoded_svg_contents.decode()


def create_word_icon(label: str, width: int = 150, return_as_datauri=True):
    match = re.findall(r"(^[a-zA-Z]{1}).*?(\d+[a-z]?)", label)
    icon_text = "".join(match[0] if match else [label[0]])
    icon_color, text_color = random_color_generator(label)
    cx = width / 2
    cy = width / 2
    r = width / 2
    fs = int(r / 25)

    t = Template(
        """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="${width}" height="${width}">

            <style>
                text {
                    font-size: ${fs}em;
                    font-family: lucida console, Fira Mono, monospace;
                    text-anchor: middle;
                    stroke-width: 1px;
                    font-weight: bold;
                    alignment-baseline: central;
                }

            </style>

            <circle cx="${cx}" cy="${cy}" r="${r}" fill="${icon_color}" />
            <text x="50%" y="50%" fill="${text_color}">${icon_text}</text>
        </svg>
    """.strip()
    )

    icon_svg = t.substitute(**locals())
    if return_as_datauri:
        return svg_to_base64_datauri(icon_svg)
    else:
        return icon_svg


def get_artifact_path(custom_metadata_list: List) -> str:
    """Get the artifact path from the custom metadata list of model.

    Parameters
    ----------
    custom_metadata_list: List
        A list of custom metadata of model.

    Returns
    -------
    str:
        The artifact path from model.
    """
    for custom_metadata in custom_metadata_list:
        if custom_metadata.key == MODEL_BY_REFERENCE_OSS_PATH_KEY:
            return custom_metadata.value
    logger.debug("Failed to get artifact path from custom metadata.")
    return UNKNOWN


def read_file(file_path: str, **kwargs) -> str:
    try:
        with fsspec.open(file_path, "r", **kwargs.get("auth", {})) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}. {e}")
        return UNKNOWN


def is_valid_ocid(ocid: str) -> bool:
    """Checks if the given ocid is valid.

    Parameters
    ----------
    ocid: str
        Oracle Cloud Identifier (OCID).

    Returns
    -------
    bool:
        Whether the given ocid is valid.
    """
    pattern = r"^ocid1\.([a-z0-9_]+)\.([a-z0-9]+)\.([a-z0-9]*)(\.[^.]+)?\.([a-z0-9_]+)$"
    match = re.match(pattern, ocid)
    return bool(match)


def get_resource_type(ocid: str) -> str:
    """Gets resource type based on the given ocid.

    Parameters
    ----------
    ocid: str
        Oracle Cloud Identifier (OCID).

    Returns
    -------
    str:
        The resource type indicated in the given ocid.

    Raises
    -------
    ValueError:
        When the given ocid is not a valid ocid.
    """
    if not is_valid_ocid(ocid):
        raise ValueError(
            f"The given ocid {ocid} is not a valid ocid."
            "Check out this page https://docs.oracle.com/en-us/iaas/Content/General/Concepts/identifiers.htm to see more details."
        )
    return ocid.split(".")[1]


def query_resource(
    ocid, return_all: bool = True
) -> "oci.resource_search.models.ResourceSummary":
    """Use Search service to find a single resource within a tenancy.

    Parameters
    ----------
    ocid: str
        Oracle Cloud Identifier (OCID).
    return_all: bool
        Whether to return allAdditionalFields.

    Returns
    -------
    oci.resource_search.models.ResourceSummary:
        The retrieved resource.
    """

    return_all = " return allAdditionalFields " if return_all else " "
    resource_type = get_resource_type(ocid)
    query = f"query {resource_type} resources{return_all}where (identifier = '{ocid}')"
    logger.info(query)

    resources = OCIResource.search(
        query,
        type=SEARCH_TYPE.STRUCTURED,
        tenant_id=TENANCY_OCID,
    )
    if len(resources) == 0:
        raise AquaRuntimeError(
            f"Failed to retreive source {resource_type}'s information.",
            service_payload={"query": query, "tenant_id": TENANCY_OCID},
        )
    return resources[0]


def query_resources(
    compartment_id,
    resource_type: str,
    return_all: bool = True,
    tag_list: list = None,
    status_list: list = None,
    connect_by_ampersands: bool = True,
    **kwargs,
) -> List["oci.resource_search.models.ResourceSummary"]:
    """Use Search service to find resources within compartment.

    Parameters
    ----------
    compartment_id: str
        The compartment ocid.
    resource_type: str
        The type of the target resources.
    return_all: bool
        Whether to return allAdditionalFields.
    tag_list: list
        List of tags will be applied for filtering.
    status_list: list
        List of lifecycleState will be applied for filtering.
    connect_by_ampersands: bool
        Whether to use `&&` to group multiple conditions.
        if `connect_by_ampersands=False`, `||` will be used.
    **kwargs:
        Additional arguments.

    Returns
    -------
    List[oci.resource_search.models.ResourceSummary]:
        The retrieved resources.
    """
    return_all = " return allAdditionalFields " if return_all else " "
    condition_lifecycle = _construct_condition(
        field_name="lifecycleState",
        allowed_values=status_list,
        connect_by_ampersands=False,
    )
    condition_tags = _construct_condition(
        field_name="freeformTags.key",
        allowed_values=tag_list,
        connect_by_ampersands=connect_by_ampersands,
    )
    query = f"query {resource_type} resources{return_all}where (compartmentId = '{compartment_id}'{condition_lifecycle}{condition_tags})"
    logger.info(query)
    logger.info(f"tenant_id=`{TENANCY_OCID}`")

    return OCIResource.search(
        query, type=SEARCH_TYPE.STRUCTURED, tenant_id=TENANCY_OCID, **kwargs
    )


def _construct_condition(
    field_name: str, allowed_values: list = None, connect_by_ampersands: bool = True
) -> str:
    """Returns tag condition applied in query statement.

    Parameters
    ----------
    field_name: str
        The field_name keyword is the resource attribute against which the
        operation and chosen value of that attribute are evaluated.
    allowed_values: list
        List of value will be applied for filtering.
    connect_by_ampersands: bool
        Whether to use `&&` to group multiple tag conditions.
        if `connect_by_ampersands=False`, `||` will be used.

    Returns
    -------
    str:
        The tag condition.
    """
    if not allowed_values:
        return ""

    joint = "&&" if connect_by_ampersands else "||"
    formatted_tags = [f"{field_name} = '{value}'" for value in allowed_values]
    joined_tags = f" {joint} ".join(formatted_tags)
    condition = f" && ({joined_tags})" if joined_tags else ""
    return condition


def upload_file_to_os(
    src_uri: str, dst_uri: str, auth: dict = None, force_overwrite: bool = False
):
    expanded_path = os.path.expanduser(src_uri)
    if not os.path.isfile(expanded_path):
        raise AquaFileNotFoundError("Invalid input file path. Specify a valid one.")
    if Path(expanded_path).suffix.lstrip(".") not in SUPPORTED_FILE_FORMATS:
        raise AquaValueError(
            f"Invalid input file. Only {', '.join(SUPPORTED_FILE_FORMATS)} files are supported."
        )
    if os.path.getsize(expanded_path) == 0:
        raise AquaValueError("Empty input file. Specify a valid file path.")

    upload_to_os(
        src_uri=expanded_path,
        dst_uri=dst_uri,
        auth=auth,
        force_overwrite=force_overwrite,
    )

def load_default_aqua_config(artifact_path: str, **kwargs) -> dict:
    config = json.loads(
        read_file(file_path=artifact_path, **kwargs) or UNKNOWN_JSON_STR
    )
    if not config:
        raise AquaError(
            f"Config file {artifact_path} is either empty or missing.",
            500,
        )
    return config

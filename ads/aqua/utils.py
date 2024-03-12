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
from typing import List, Union
from functools import wraps
import asyncio

import fsspec
import oci
from oci.data_science.models import JobRun, Model

from ads.aqua.constants import RqsAdditionalDetails
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.exception import AquaFileNotFoundError, AquaRuntimeError, AquaValueError
from ads.aqua.data import Tags
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import get_console_link, upload_to_os
from ads.config import AQUA_CONFIG_FOLDER, AQUA_SERVICE_MODELS_BUCKET, TENANCY_OCID
from ads.model import DataScienceModel

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("ODSC_AQUA")

UNKNOWN = ""
UNKNOWN_DICT = {}
README = "README.md"
DEPLOYMENT_CONFIG = "deployment_config.json"
EVALUATION_REPORT_JSON = "report.json"
EVALUATION_REPORT_MD = "report.md"
EVALUATION_REPORT = "report.html"
UNKNOWN_JSON_STR = "{}"
CONSOLE_LINK_RESOURCE_TYPE_MAPPING = dict(
    datasciencemodel="models",
    datasciencemodeldeployment="model-deployments",
    datasciencejob="jobs",
    datasciencejobrun="jobruns",
    datasciencemodelversionset="model-version-sets",
)
CONDA_BUCKET_NS = os.environ.get("CONDA_BUCKET_NS", "ociodscdev")
SOURCE_FILE = "run.sh"
CONDA_URI = f"oci://aqua-eval-test-bucket@{CONDA_BUCKET_NS}/conda_environments/gpu/PyTorch 2.1 for GPU on Python 3.9/1.0/pytorch21_p39_gpu_v1"
CONDA_REGION = "us-ashburn-1"
BERT_SCORE_PATH = "/home/datascience/conda/pytorch21_p39_gpu_v1/bertscore/bertscore.py"
BERT_BASE_MULTILINGUAL_CASED = (
    "/home/datascience/conda/pytorch21_p39_gpu_v1/bert-base-multilingual-cased/"
)
FINE_TUNING_RUNTIME_CONTAINER = "iad.ocir.io/ociodscdev/aqua_ft_cuda121:0.3.16.19"
DEFAULT_FT_BLOCK_STORAGE_SIZE = 256
DEFAULT_FT_REPLICA = 1
DEFAULT_FT_BATCH_SIZE = 1
DEFAULT_FT_VALIDATION_SET_SIZE = 0.1

HF_MODELS = "/home/datascience/conda/pytorch21_p39_gpu_v1/"
MAXIMUM_ALLOWED_DATASET_IN_BYTE = 52428800  # 1024 x 1024 x 50 = 50MB
JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING = "ME_STANDALONE"
NB_SESSION_IDENTIFIER = "NB_SESSION_OCID"
LIFECYCLE_DETAILS_MISSING_JOBRUN = "The asscociated JobRun resource has been deleted."


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
MODEL_BY_REFERENCE_OSS_PATH_KEY = "artifact_location"


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
            if ObjectStorageDetails.is_oci_path(custom_metadata.value):
                artifact_path = custom_metadata.value
            else:
                artifact_path = ObjectStorageDetails(
                    AQUA_SERVICE_MODELS_BUCKET, CONDA_BUCKET_NS, custom_metadata.value
                ).path
            return artifact_path
    logger.debug("Failed to get artifact path from custom metadata.")
    return UNKNOWN


def read_file(file_path: str, **kwargs) -> str:
    try:
        with fsspec.open(file_path, "r", **kwargs.get("auth", {})) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read file {file_path}. {e}")
        return UNKNOWN


def load_config(file_path: str, config_file_name: str, **kwargs) -> dict:
    artifact_path = f"{file_path.rstrip('/')}/{config_file_name}"
    config = json.loads(
        read_file(file_path=artifact_path, **kwargs) or UNKNOWN_JSON_STR
    )
    if not config:
        raise AquaFileNotFoundError(
            f"Config file `{config_file_name}` is either empty or missing at {artifact_path}",
            500,
        )
    return config


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
            f"Failed to retreive {resource_type}'s information.",
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


def upload_local_to_os(
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
    if os.path.getsize(expanded_path) > MAXIMUM_ALLOWED_DATASET_IN_BYTE:
        raise AquaValueError(
            f"Local dataset file can't exceed {MAXIMUM_ALLOWED_DATASET_IN_BYTE} bytes."
        )

    upload_to_os(
        src_uri=expanded_path,
        dst_uri=dst_uri,
        auth=auth,
        force_overwrite=force_overwrite,
    )


def sanitize_response(oci_client, response: list):
    """Builds a JSON POST object for the response from OCI clients.

    Parameters
    ----------
    oci_client
        OCI client object

    response
        list of results from the OCI client

    Returns
    -------
        The serialized form of data.

    """
    return oci_client.base_client.sanitize_for_serialization(response)


def _build_resource_identifier(
    id: str = None, name: str = None, region: str = None
) -> AquaResourceIdentifier:
    """Constructs AquaResourceIdentifier based on the given ocid and display name."""
    try:
        resource_type = CONSOLE_LINK_RESOURCE_TYPE_MAPPING.get(get_resource_type(id))

        return AquaResourceIdentifier(
            id=id,
            name=name,
            url=get_console_link(
                resource=resource_type,
                ocid=id,
                region=region,
            ),
        )
    except Exception as e:
        logger.error(
            f"Failed to construct AquaResourceIdentifier from given id=`{id}`, and name=`{name}`, {str(e)}"
        )
        return AquaResourceIdentifier()


def _get_experiment_info(
    model: Union[oci.resource_search.models.ResourceSummary, DataScienceModel]
) -> tuple:
    """Returns ocid and name of the experiment."""
    return (
        (
            model.additional_details.get(RqsAdditionalDetails.MODEL_VERSION_SET_ID),
            model.additional_details.get(RqsAdditionalDetails.MODEL_VERSION_SET_NAME),
        )
        if isinstance(model, oci.resource_search.models.ResourceSummary)
        else (model.model_version_set_id, model.model_version_set_name)
    )


def _build_job_identifier(
    job_run_details: Union[
        oci.data_science.models.JobRun, oci.resource_search.models.ResourceSummary
    ] = None,
    **kwargs,
) -> AquaResourceIdentifier:
    try:
        job_id = (
            job_run_details.id
            if isinstance(job_run_details, oci.data_science.models.JobRun)
            else job_run_details.identifier
        )
        return _build_resource_identifier(
            id=job_id, name=job_run_details.display_name, **kwargs
        )

    except Exception as e:
        logger.debug(
            f"Failed to get job details from job_run_details: {job_run_details}"
            f"DEBUG INFO:{str(e)}"
        )
        return AquaResourceIdentifier()


def get_container_image(
    custom_metadata_list, config_file_name: str, container_type: str
) -> str:
    """Gets the image name from the given model and container type.
    Parameters
    ----------
    custom_metadata_list:
        custom metadata of a given model
    config_file_name: str
        name of the config file
    container_type: str
        type of container, can be either deployment-container, finetune-container, evaluation-container

    Returns
    -------
    Dict:
        A dict of allowed configs.
    """

    try:
        container_type = custom_metadata_list.get(container_type).value
    except ValueError:
        raise AquaValueError(
            f"{container_type} key is not available in the custom metadata field."
        )

    # todo: currently loads config within ads, artifact_path will be an external bucket
    config = load_config(
        AQUA_CONFIG_FOLDER,
        config_file_name=config_file_name,
    )

    if container_type not in config:
        raise AquaValueError(
            f"{config_file_name} does not have config details for model: {container_type}"
        )

    container_image = None
    mapping = config[container_type]
    versions = [obj["version"] for obj in mapping]
    # todo: assumes numbered versions, update if `latest` is used
    latest = max(versions)
    for obj in mapping:
        if obj["version"] == latest:
            container_image = f"{obj['name']}:{obj['version']}"
            break

    if not container_image:
        raise AquaValueError(
            f"{config_file_name} is missing name and/or version details."
        )

    return container_image


def fire_and_forget(func):
    """Decorator to push execution of methods to the background."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, func, *args, *kwargs)

    return wrapped


def get_base_model_from_tags(tags):
    base_model_ocid = ""
    base_model_name = ""
    if Tags.AQUA_FINE_TUNED_MODEL_TAG.value in tags:
        tag = tags[Tags.AQUA_FINE_TUNED_MODEL_TAG.value]
        if "#" in tag:
            base_model_ocid, base_model_name = tag.split("#")

        if not (is_valid_ocid(base_model_ocid) and base_model_name):
            raise AquaValueError(
                f"{Tags.AQUA_FINE_TUNED_MODEL_TAG.value} tag should have the format `Service Model OCID#Model Name`."
            )

    return base_model_ocid, base_model_name


def get_resource_name(ocid: str) -> str:
    """Gets resource name based on the given ocid.

    Parameters
    ----------
    ocid: str
        Oracle Cloud Identifier (OCID).

    Returns
    -------
    str:
        The resource name indicated in the given ocid.

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
    try:
        resource = query_resource(ocid, return_all=False)
        name = resource.display_name if resource else ""
    except:
        name = ""
    return name

#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA utils and constants."""

import asyncio
import base64
import json
import logging
import os
import random
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Union

import fsspec
import oci
from cachetools import TTLCache, cached
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import HfApi, ModelInfo
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from oci.data_science.models import JobRun, Model
from oci.object_storage.models import ObjectSummary
from pydantic import BaseModel, ValidationError

from ads.aqua.common.entities import GPUShapesIndex
from ads.aqua.common.enums import (
    CONTAINER_FAMILY_COMPATIBILITY,
    InferenceContainerParamType,
    InferenceContainerType,
    RqsAdditionalDetails,
    TextEmbeddingInferenceContainerParams,
)
from ads.aqua.common.errors import (
    AquaFileNotFoundError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.constants import (
    AQUA_GA_LIST,
    CONSOLE_LINK_RESOURCE_TYPE_MAPPING,
    DEPLOYMENT_CONFIG,
    FINE_TUNING_CONFIG,
    HF_LOGIN_DEFAULT_TIMEOUT,
    LICENSE,
    MAXIMUM_ALLOWED_DATASET_IN_BYTE,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    README,
    SERVICE_MANAGED_CONTAINER_URI_SCHEME,
    SUPPORTED_FILE_FORMATS,
    TEI_CONTAINER_DEFAULT_HOST,
    TGI_INFERENCE_RESTRICTED_PARAMS,
    UNKNOWN_JSON_STR,
    VLLM_INFERENCE_RESTRICTED_PARAMS,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.common import auth as authutil
from ads.common.auth import AuthState, default_signer
from ads.common.decorator.threaded import threaded
from ads.common.extended_enum import ExtendedEnum
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import (
    UNKNOWN,
    copy_file,
    get_console_link,
    read_file,
    upload_to_os,
)
from ads.config import (
    AQUA_MODEL_DEPLOYMENT_FOLDER,
    AQUA_SERVICE_MODELS_BUCKET,
    CONDA_BUCKET_NAME,
    CONDA_BUCKET_NS,
    TENANCY_OCID,
)
from ads.model import DataScienceModel, ModelVersionSet

logger = logging.getLogger("ads.aqua")


DEFINED_METADATA_TO_FILE_MAP = {
    "readme": README,
    "license": LICENSE,
    "finetuneconfiguration": FINE_TUNING_CONFIG,
    "deploymentconfiguration": DEPLOYMENT_CONFIG,
}


class LifecycleStatus(ExtendedEnum):
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
            if job_run_status in {
                JobRun.LIFECYCLE_STATE_IN_PROGRESS,
                JobRun.LIFECYCLE_STATE_ACCEPTED,
            }:
                status = JobRun.LIFECYCLE_STATE_IN_PROGRESS
            elif job_run_status in {
                JobRun.LIFECYCLE_STATE_FAILED,
                JobRun.LIFECYCLE_STATE_NEEDS_ATTENTION,
            }:
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
        A list of custom metadata of OCI model.

    Returns
    -------
    str:
        The artifact path from model.
    """
    try:
        for custom_metadata in custom_metadata_list:
            if custom_metadata.key == MODEL_BY_REFERENCE_OSS_PATH_KEY:
                if ObjectStorageDetails.is_oci_path(custom_metadata.value):
                    artifact_path = custom_metadata.value
                else:
                    artifact_path = ObjectStorageDetails(
                        AQUA_SERVICE_MODELS_BUCKET,
                        CONDA_BUCKET_NS,
                        custom_metadata.value,
                    ).path
                return artifact_path
    except Exception as ex:
        logger.debug(ex)

    logger.debug("Failed to get artifact path from custom metadata.")
    return UNKNOWN


@threaded()
def load_config(file_path: str, config_file_name: str, **kwargs) -> dict:
    artifact_path = f"{file_path.rstrip('/')}/{config_file_name}"
    signer = default_signer() if artifact_path.startswith("oci://") else {}
    config = json.loads(
        read_file(file_path=artifact_path, auth=signer, **kwargs) or UNKNOWN_JSON_STR
    )
    if not config:
        raise AquaFileNotFoundError(
            f"Config file `{config_file_name}` is either empty or missing at {artifact_path}",
            500,
        )
    return config


def list_os_files_with_extension(oss_path: str, extension: str) -> List[str]:
    """
    List files in the specified directory with the given extension.

    Parameters:
    - oss_path: The path to the directory where files are located.
    - extension: The file extension to filter by (e.g., 'txt' for text files).

    Returns:
    - A list of file paths matching the specified extension.
    """

    oss_client = ObjectStorageDetails.from_path(oss_path)

    # Ensure the extension is prefixed with a dot if not already
    if not extension.startswith("."):
        extension = "." + extension
    files: List[ObjectSummary] = oss_client.list_objects().objects

    return [
        file.name[len(oss_client.filepath) :].lstrip("/")
        for file in files
        if file.name.endswith(extension)
    ]


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

    if not ocid:
        return False
    return ocid.lower().startswith("ocid")


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
    logger.debug(query)

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
    logger.debug(query)
    logger.debug(f"tenant_id=`{TENANCY_OCID}`")

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
        logger.debug(
            f"Failed to construct AquaResourceIdentifier from given id=`{id}`, and name=`{name}`, {str(e)}"
        )
        return AquaResourceIdentifier()


def _get_experiment_info(
    model: Union[oci.resource_search.models.ResourceSummary, DataScienceModel],
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


def get_max_version(versions):
    """Takes in a list of versions and returns the higher version."""
    if not versions:
        return UNKNOWN

    def compare_versions(version1, version2):
        # split version strings into parts and convert to int values for comparison
        parts1 = list(map(int, version1.split(".")))
        parts2 = list(map(int, version2.split(".")))

        # compare each part
        for idx in range(min(len(parts1), len(parts2))):
            if parts1[idx] < parts2[idx]:
                return version2
            elif parts1[idx] > parts2[idx]:
                return version1

        # if all parts are equal up to this point, return the longer version string
        return version1 if len(parts1) > len(parts2) else version2

    max_version = versions[0]
    for version in versions[1:]:
        max_version = compare_versions(max_version, version)

    return max_version


def fire_and_forget(func):
    """Decorator to push execution of methods to the background."""

    @wraps(func)
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, func, *args, *kwargs)

    return wrapped


def extract_id_and_name_from_tag(tag: str):
    base_model_ocid = UNKNOWN
    base_model_name = UNKNOWN
    try:
        base_model_ocid, base_model_name = tag.split("#")
    except Exception:
        pass

    if not (is_valid_ocid(base_model_ocid) and base_model_name):
        logger.debug(
            f"Invalid {tag}. Specify tag in the format as <service_model_id>#<service_model_name>."
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
        name = resource.display_name if resource else UNKNOWN
    except Exception:
        name = UNKNOWN
    return name


def get_model_by_reference_paths(
    model_file_description: dict, is_ft_model_v2: bool = False
):
    """Reads the model file description json dict and returns the base model path and fine-tuned path for
        models created by reference.

    Parameters
    ----------
    model_file_description: dict
        json dict containing model paths and objects for models created by reference.
    is_ft_model_v2: bool
        Flag to indicate if it's fine tuned model v2. Defaults to False.

    Returns
    -------
        a tuple with base_model_path and fine_tune_output_path
    """
    base_model_path = UNKNOWN
    fine_tune_output_path = UNKNOWN
    models = model_file_description["models"]

    if not models:
        raise AquaValueError(
            "Model path is not available in the model json artifact. "
            "Please check if the model created by reference has the correct artifact."
        )

    if is_ft_model_v2:
        # model_file_description json for fine tuned model v2 contains only fine tuned model artifacts
        # so first model is always the fine tuned model
        ft_model_artifact = models[0]
        fine_tune_output_path = f"oci://{ft_model_artifact['bucketName']}@{ft_model_artifact['namespace']}/{ft_model_artifact['prefix']}".rstrip(
            "/"
        )

        return UNKNOWN, fine_tune_output_path

    if len(models) > 0:
        # since the model_file_description json for legacy fine tuned model does not have a flag to identify the base model, we consider
        # the first instance to be the base model.
        base_model_artifact = models[0]
        base_model_path = f"oci://{base_model_artifact['bucketName']}@{base_model_artifact['namespace']}/{base_model_artifact['prefix']}".rstrip(
            "/"
        )
    if len(models) > 1:
        # second model is considered as fine-tuned model
        ft_model_artifact = models[1]
        fine_tune_output_path = f"oci://{ft_model_artifact['bucketName']}@{ft_model_artifact['namespace']}/{ft_model_artifact['prefix']}".rstrip(
            "/"
        )

    return base_model_path, fine_tune_output_path


def _is_valid_mvs(mvs: ModelVersionSet, target_tag: str) -> bool:
    """Returns whether the given model version sets has the target tag.

    Parameters
    ----------
    mvs: str
        The instance of `ads.model.ModelVersionSet`.
    target_tag: list
        Target tag expected to be in MVS.

    Returns
    -------
    bool:
        Return True if the given model version sets is valid.
    """
    if mvs.freeform_tags is None:
        return False

    return target_tag in mvs.freeform_tags


def known_realm():
    """This helper function returns True if the Aqua service is available by default in the given namespace.
    Returns
    -------
    bool:
        Return True if aqua service is available.

    """
    return os.environ.get("CONDA_BUCKET_NS") in AQUA_GA_LIST


def get_ocid_substring(ocid: str, key_len: int) -> str:
    """This helper function returns the last n characters of the ocid specified by key_len parameter.
    If ocid is None or length is less than key_len, it returns an empty string."""
    return ocid[-key_len:] if ocid and len(ocid) > key_len else ""


def upload_folder(
    os_path: str, local_dir: str, model_name: str, exclude_pattern: str = None
) -> str:
    """Upload the local folder to the object storage

    Args:
        os_path (str): object storage URI with prefix. This is the path to upload
        local_dir (str): Local directory where the object is downloaded
        model_name (str): Name of the huggingface model
        exclude_pattern (optional, str): The matching pattern of files to be excluded from uploading.
    Retuns:
        str: Object name inside the bucket
    """
    os_details: ObjectStorageDetails = ObjectStorageDetails.from_path(os_path)
    if not os_details.is_bucket_versioned():
        raise ValueError(f"Version is not enabled at object storage location {os_path}")
    auth_state = AuthState()
    object_path = os_details.filepath.rstrip("/") + "/" + model_name + "/"
    command = f"oci os object bulk-upload --src-dir {local_dir} --prefix {object_path} -bn {os_details.bucket} -ns {os_details.namespace} --auth {auth_state.oci_iam_type} --profile {auth_state.oci_key_profile} --no-overwrite"
    if exclude_pattern:
        command += f" --exclude {exclude_pattern}"
    try:
        logger.info(f"Running: {command}")
        subprocess.check_call(shlex.split(command))
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error uploading the object. Exit code: {e.returncode} with error {e.stdout}"
        )

    return f"oci://{os_details.bucket}@{os_details.namespace}" + "/" + object_path


def cleanup_local_hf_model_artifact(
    model_name: str,
    local_dir: str = None,
):
    """
    Helper function that deletes local artifacts downloaded from Hugging Face to free up disk space.
    Parameters
    ----------
    model_name (str): Name of the huggingface model
    local_dir (str): Local directory where the object is downloaded

    """
    if local_dir and os.path.exists(local_dir):
        model_dir = os.path.join(local_dir, model_name)
        model_dir = (
            os.path.dirname(model_dir)
            if "/" in model_name or os.sep in model_name
            else model_dir
        )
        shutil.rmtree(model_dir, ignore_errors=True)
        if os.path.exists(model_dir):
            logger.debug(
                f"Could not delete local model artifact directory: {model_dir}"
            )
        else:
            logger.debug(f"Deleted local model artifact directory: {model_dir}.")

    hf_local_path = os.path.join(
        HF_HUB_CACHE, repo_folder_name(repo_id=model_name, repo_type="model")
    )
    shutil.rmtree(hf_local_path, ignore_errors=True)

    if os.path.exists(hf_local_path):
        logger.debug(
            f"Could not clear the local Hugging Face cache directory {hf_local_path} for the model {model_name}."
        )
    else:
        logger.debug(
            f"Cleared contents of local Hugging Face cache directory {hf_local_path} for the model {model_name}."
        )


def is_service_managed_container(container):
    return container and container.startswith(SERVICE_MANAGED_CONTAINER_URI_SCHEME)


def get_params_list(params: str) -> List[str]:
    """
    Parses a string of CLI-style double-dash parameters and returns them as a list.

    Parameters
    ----------
    params : str
        A single string containing parameters separated by the `--` delimiter.

    Returns
    -------
    List[str]
        A list of parameter strings, each starting with `--`.

    Example
    -------
    >>> get_params_list("--max-model-len 65536 --enforce-eager")
    ['--max-model-len 65536', '--enforce-eager']
    """
    if not params:
        return []
    return [f"--{param.strip()}" for param in params.split("--") if param.strip()]


def get_params_dict(params: Union[str, List[str]]) -> dict:
    """
    Converts CLI-style double-dash parameters (as string or list) into a dictionary.

    Parameters
    ----------
    params : Union[str, List[str]]
        Parameters provided either as:
        - a single string: "--key1 val1 --key2 val2"
        - a list of strings: ["--key1 val1", "--key2 val2"]

    Returns
    -------
    dict
        A dictionary mapping parameter names to their values. If no value is found, uses `UNKNOWN`.

    Example
    -------
    >>> get_params_dict("--max-model-len 65536 --enforce-eager")
    {'--max-model-len': '65536', '--enforce-eager': ''}
    """
    params_list = get_params_list(params) if isinstance(params, str) else params
    return {
        split_result[0]: " ".join(split_result[1:])
        if len(split_result) > 1
        else UNKNOWN
        for split_result in (x.split() for x in params_list)
    }


def get_combined_params(
    params1: Optional[str] = None, params2: Optional[str] = None
) -> str:
    """
    Merges two double-dash parameter strings (`--param value`) into one, with values from `params2`
    overriding any duplicates from `params1`.

    Parameters
    ----------
    params1 : Optional[str]
        The base parameter string. Can be None.
    params2 : Optional[str]
        The override parameter string. Parameters in this string will override those in `params1`.

    Returns
    -------
    str
        A combined parameter string with deduplicated keys and overridden values from `params2`.
    """
    if not params1 and not params2:
        return ""

    # If only one string is provided, return it directly
    if not params1:
        return params2.strip()
    if not params2:
        return params1.strip()

    # Combine both dictionaries, with params2 overriding params1
    merged_dict = {**get_params_dict(params1), **get_params_dict(params2)}

    # Reconstruct the string
    combined = " ".join(
        f"{key} {value}" if value else key for key, value in merged_dict.items()
    )

    return combined.strip()


def find_restricted_params(
    default_params: Union[str, List[str]],
    user_params: Union[str, List[str]],
    container_family: str,
) -> List[str]:
    """Returns a list of restricted params that user chooses to override when creating an Aqua deployment.
    The default parameters coming from the container index json file cannot be overridden.

    Parameters
    ----------
    default_params:
        Inference container parameter string with default values.
    user_params:
        Inference container parameter string with user provided values.
    container_family: str
        The image family of model deployment container runtime.

    Returns
    -------
        A list with params keys common between params1 and params2.

    """
    restricted_params = []
    if default_params and user_params:
        default_params_dict = get_params_dict(default_params)
        user_params_dict = get_params_dict(user_params)

        restricted_params_set = get_restricted_params_by_container(container_family)
        for key, _items in user_params_dict.items():
            if key in default_params_dict or key in restricted_params_set:
                restricted_params.append(key.lstrip("-"))

    return restricted_params


def build_params_string(params: Optional[Dict[str, Any]]) -> str:
    """
    Converts a dictionary of CLI parameters into a command-line friendly string.

    This is typically used to transform framework-specific model parameters (e.g., vLLM or TGI flags)
    into a space-separated string that can be passed to container startup commands.

    Parameters
    ----------
    params : Optional[Dict[str, Any]]
        Dictionary containing parameter name as keys (e.g., "--max-model-len") and their corresponding values.
        If a parameter does not require a value (e.g., a boolean flag), its value can be None or an empty string.

    Returns
    -------
    str
        A space-separated string of CLI arguments.
        Returns "<unknown>" if the input dictionary is None or empty.

    Example
    -------
    >>> build_params_string({"--max-model-len": 4096, "--enforce-eager": None})
    '--max-model-len 4096 --enforce-eager'
    """
    if not params:
        return UNKNOWN

    parts = []
    for key, value in params.items():
        if value is None or value == "":
            parts.append(str(key))
        else:
            parts.append(f"{key} {value}")

    return " ".join(parts).strip()


def copy_model_config(
    artifact_path: str, os_path: str, auth: Optional[Dict[str, Any]] = None
):
    """Copies the aqua model config folder from the artifact path to the user provided object storage path.
    The config folder is overwritten if the files already exist at the destination path.

    Parameters
    ----------
    artifact_path:
        Path of the aqua model where config folder is available.
    os_path:
        User provided path where config folder will be copied.
    auth: (Dict, optional). Defaults to None.
        The default authentication is set using `ads.set_auth` API. If you need to override the
        default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
        authentication signer and kwargs required to instantiate IdentityClient object.

    Returns
    -------
    None
        Nothing.
    """

    try:
        source_dir = ObjectStorageDetails(
            AQUA_SERVICE_MODELS_BUCKET,
            CONDA_BUCKET_NS,
            f"{os.path.dirname(artifact_path).rstrip('/')}/config",
        ).path
        dest_dir = f"{os_path.rstrip('/')}/config"

        oss_details = ObjectStorageDetails.from_path(source_dir)
        objects = oss_details.list_objects(fields="name").objects

        for obj in objects:
            source_path = ObjectStorageDetails(
                AQUA_SERVICE_MODELS_BUCKET, CONDA_BUCKET_NS, obj.name
            ).path
            destination_path = os.path.join(dest_dir, os.path.basename(obj.name))
            copy_file(
                uri_src=source_path,
                uri_dst=destination_path,
                force_overwrite=True,
                auth=auth,
            )
    except Exception as ex:
        logger.debug(ex)
        logger.debug(f"Failed to copy config folder from {artifact_path} to {os_path}.")


def get_container_params_type(container_type_name: str) -> str:
    """The utility function accepts the deployment container type name and returns the corresponding params name.
    Parameters
    ----------
    container_type_name: str
        type of deployment container, like odsc-vllm-serving or odsc-tgi-serving.

    Returns
    -------
        InferenceContainerParamType value

    """
    # check substring instead of direct match in case container_type_name changes in the future
    if InferenceContainerType.CONTAINER_TYPE_VLLM in container_type_name.lower():
        return InferenceContainerParamType.PARAM_TYPE_VLLM
    elif InferenceContainerType.CONTAINER_TYPE_TGI in container_type_name.lower():
        return InferenceContainerParamType.PARAM_TYPE_TGI
    elif InferenceContainerType.CONTAINER_TYPE_LLAMA_CPP in container_type_name.lower():
        return InferenceContainerParamType.PARAM_TYPE_LLAMA_CPP
    else:
        return UNKNOWN


def get_container_env_type(container_type_name: Optional[str]) -> str:
    """
    Determine the container environment type based on the container type name.

    This function matches the provided container type name against the known
    values of `InferenceContainerType`. The check is case-insensitive and
    allows for partial matches so that changes in container naming conventions
    (e.g., prefixes or suffixes) will still be matched correctly.

    Examples:
        >>> get_container_env_type("odsc-vllm-serving")
        'vllm'
        >>> get_container_env_type("ODSC-TGI-Serving")
        'tgi'
        >>> get_container_env_type("custom-unknown-container")
        'UNKNOWN'

    Args:
        container_type_name (Optional[str]):
            The deployment container type name (e.g., "odsc-vllm-serving").

    Returns:
        str:
            - A matching `InferenceContainerType` value string (e.g., "VLLM", "TGI", "LLAMA-CPP").
            - `"UNKNOWN"` if no match is found or the input is empty/None.
    """
    if not container_type_name:
        return UNKNOWN

    needle = container_type_name.strip().casefold()

    for container_type in InferenceContainerType.values():
        if container_type and container_type.casefold() in needle:
            return container_type.upper()

    return UNKNOWN


def get_restricted_params_by_container(container_type_name: str) -> set:
    """The utility function accepts the deployment container type name and returns a set of restricted params
        for that container.
    Parameters
    ----------
    container_type_name: str
        type of deployment container, like odsc-vllm-serving or odsc-tgi-serving.

    Returns
    -------
        Set of restricted params based on container type

    """
    # check substring instead of direct match in case container_type_name changes in the future
    if InferenceContainerType.CONTAINER_TYPE_VLLM in container_type_name.lower():
        return VLLM_INFERENCE_RESTRICTED_PARAMS
    elif InferenceContainerType.CONTAINER_TYPE_TGI in container_type_name.lower():
        return TGI_INFERENCE_RESTRICTED_PARAMS
    else:
        return set()


def get_huggingface_login_timeout() -> int:
    """This helper function returns the huggingface login timeout, returns default if not set via
    env var.
    Returns
    -------
    timeout: int
        huggingface login timeout.

    """
    timeout = HF_LOGIN_DEFAULT_TIMEOUT
    try:
        timeout = int(
            os.environ.get("HF_LOGIN_DEFAULT_TIMEOUT", HF_LOGIN_DEFAULT_TIMEOUT)
        )
    except ValueError:
        pass
    return timeout


def format_hf_custom_error_message(error: HfHubHTTPError):
    """
    Formats a custom error message based on the Hugging Face error response.

    Parameters
    ----------
    error (HfHubHTTPError): The caught exception.

    Raises
    ------
    AquaRuntimeError: A user-friendly error message.
    """
    # Extract the repository URL from the error message if present
    match = re.search(r"(https://huggingface.co/[^\s]+)", str(error))
    url = match.group(1) if match else "the requested Hugging Face URL."

    if isinstance(error, RepositoryNotFoundError):
        raise AquaRuntimeError(
            reason=f"Failed to access `{url}`. Please check if the provided repository name is correct. "
            "If the repo is private, make sure you are authenticated and have a valid HF token registered. "
            "To register your token, run this command in your terminal: `huggingface-cli login`",
            service_payload={"error": "RepositoryNotFoundError"},
        )

    if isinstance(error, GatedRepoError):
        raise AquaRuntimeError(
            reason=f"Access denied to `{url}` "
            "This repository is gated. Access is restricted to authorized users. "
            "Please request access or check with the repository administrator. "
            "If you are trying to access a gated repository, ensure you have a valid HF token registered. "
            "To register your token, run this command in your terminal: `huggingface-cli login`",
            service_payload={"error": "GatedRepoError"},
        )

    if isinstance(error, RevisionNotFoundError):
        raise AquaRuntimeError(
            reason=f"The specified revision could not be found at `{url}` "
            "Please check the revision identifier and try again.",
            service_payload={"error": "RevisionNotFoundError"},
        )
                
    raise AquaRuntimeError(
        reason=f"An error occurred while accessing `{url}` "
        "Please check your network connection and try again. "
        "If you are trying to access a gated repository, ensure you have a valid HF token registered. "
        "To register your token, run this command in your terminal: `huggingface-cli login`",
        service_payload={"error": "Error"},
    )


@cached(cache=TTLCache(maxsize=1, ttl=timedelta(hours=5), timer=datetime.now))
def get_hf_model_info(repo_id: str) -> ModelInfo:
    """Gets the model information object for the given model repository name. For models that requires a token,
    this method assumes that the token validation is already done.

    Parameters
    ----------
    repo_id: str
        hugging face model repository name

    Returns
    -------
        instance of ModelInfo object

    """
    try:
        return HfApi().model_info(repo_id=repo_id)
    except HfHubHTTPError as err:
        raise format_hf_custom_error_message(err) from err


@cached(cache=TTLCache(maxsize=1, ttl=timedelta(hours=5), timer=datetime.now))
def list_hf_models(query: str) -> List[str]:
    try:
        models = HfApi().list_models(
            model_name=query,
            sort="downloads",
            direction=-1,
            limit=500,
        )
        return [model.id for model in models if model.disabled is None]
    except HfHubHTTPError as err:
        raise format_hf_custom_error_message(err) from err


def generate_tei_cmd_var(os_path: str) -> List[str]:
    """This utility functions generates CMD params for Text Embedding Inference container. Only the
    essential parameters for OCI model deployment are added, defaults are used for the rest.
    Parameters
    ----------
    os_path: str
        OCI bucket path where the model artifacts are uploaded - oci://bucket@namespace/prefix

    Returns
    -------
        cmd_var:
            List of command line arguments
    """

    cmd_prefix = "--"
    cmd_var = [
        f"{cmd_prefix}{TextEmbeddingInferenceContainerParams.MODEL_ID}",
        f"{AQUA_MODEL_DEPLOYMENT_FOLDER}{ObjectStorageDetails.from_path(os_path.rstrip('/')).filepath}/",
        f"{cmd_prefix}{TextEmbeddingInferenceContainerParams.PORT}",
        TEI_CONTAINER_DEFAULT_HOST,
    ]

    return cmd_var


def parse_cmd_var(cmd_list: List[str]) -> dict:
    """Helper functions that parses a list into a key-value dictionary. The list contains keys separated by the prefix
    '--' and the value of the key is the subsequent element.
    """
    parsed_cmd = {}

    for i, cmd in enumerate(cmd_list):
        if cmd.startswith("--"):
            if i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("--"):
                parsed_cmd[cmd] = cmd_list[i + 1]
                i += 1
            else:
                parsed_cmd[cmd] = None
    return parsed_cmd


def validate_cmd_var(
    cmd_var: Optional[List[str]], overrides: Optional[List[str]]
) -> List[str]:
    """
    Validates and combines two lists of command-line parameters. Raises an error if any parameter
    key in the `overrides` list already exists in the `cmd_var` list, preventing unintended overrides.

    Parameters
    ----------
    cmd_var : Optional[List[str]]
        The default list of command-line parameters (e.g., ["--param1", "value1", "--flag"]).
    overrides : Optional[List[str]]
        The list of overriding command-line parameters.

    Returns
    -------
    List[str]
        A validated and combined list of parameters, with overrides appended.

    Raises
    ------
    AquaValueError
        If `overrides` contain any parameter keys that already exist in `cmd_var`.
    """
    cmd_var = [str(x).strip() for x in cmd_var or []]
    overrides = [str(x).strip() for x in overrides or []]

    cmd_dict = parse_cmd_var(cmd_var)
    overrides_dict = parse_cmd_var(overrides)

    # Check for conflicting keys
    conflicting_keys = set(cmd_dict.keys()) & set(overrides_dict.keys())
    if conflicting_keys:
        raise AquaValueError(
            f"Cannot override the following model deployment parameters: {', '.join(sorted(conflicting_keys))}"
        )

    combined_params = cmd_var + overrides
    return combined_params


def build_pydantic_error_message(ex: ValidationError):
    """
    Added to handle error messages from pydantic model validator.
    Combine both loc and msg for errors where loc (field) is present in error details, else only build error
    message using msg field.
    """

    return {
        ".".join(map(str, e["loc"])): e["msg"]
        for e in ex.errors()
        if "loc" in e and e["loc"]
    } or "; ".join(e["msg"] for e in ex.errors())


def is_pydantic_model(obj: object) -> bool:
    """
    Returns True if obj is a Pydantic model class or an instance of a Pydantic model.

    Args:
        obj: The object or class to check.

    Returns:
        bool: True if obj is a subclass or instance of BaseModel, False otherwise.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return issubclass(cls, BaseModel)


@cached(cache=TTLCache(maxsize=1, ttl=timedelta(minutes=5), timer=datetime.now))
def load_gpu_shapes_index(
    auth: Optional[Dict[str, Any]] = None,
) -> GPUShapesIndex:
    """
    Load the GPU shapes index, merging based on freshness.

    Compares last-modified timestamps of local and remote files,
    merging the shapes from the fresher file on top of the older one.

    Parameters
    ----------
    auth
        Optional auth dict (as returned by `ads.common.auth.default_signer()`)
        to pass through to `fsspec.open()`.

    Returns
    -------
    GPUShapesIndex
        Merged index where any shape present remotely supersedes the local entry.

    Raises
    ------
    json.JSONDecodeError
        If any of the JSON is malformed.
    """
    file_name = "gpu_shapes_index.json"

    # Try remote load
    local_data, remote_data = {}, {}
    local_mtime, remote_mtime = None, None

    if CONDA_BUCKET_NS:
        try:
            auth = auth or authutil.default_signer()
            storage_path = (
                f"oci://{CONDA_BUCKET_NAME}@{CONDA_BUCKET_NS}/service_pack/{file_name}"
            )
            logger.debug(
                "Loading GPU shapes index from Object Storage: %s", storage_path
            )

            fs = fsspec.filesystem("oci", **auth)
            with fs.open(storage_path, mode="r") as f:
                remote_data = json.load(f)

            remote_info = fs.info(storage_path)
            remote_mtime_str = remote_info.get("timeModified", None)
            if remote_mtime_str:
                # Convert OCI timestamp (e.g., 'Mon, 04 Aug 2025 06:37:13 GMT') to epoch time
                remote_mtime = datetime.strptime(
                    remote_mtime_str, "%a, %d %b %Y %H:%M:%S %Z"
                ).timestamp()

                logger.debug(
                    "Remote GPU shapes last-modified time: %s",
                    datetime.fromtimestamp(remote_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                )

            logger.debug(
                "Loaded %d shapes from Object Storage",
                len(remote_data.get("shapes", {})),
            )
        except Exception as ex:
            logger.debug("Remote load failed (%s); falling back to local", ex)

    # Load local copy
    local_path = os.path.join(os.path.dirname(__file__), "../resources", file_name)
    try:
        logger.debug("Loading GPU shapes index from local file: %s", local_path)
        with open(local_path) as f:
            local_data = json.load(f)

        local_mtime = os.path.getmtime(local_path)

        logger.debug(
            "Local GPU shapes last-modified time: %s",
            datetime.fromtimestamp(local_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        )

        logger.debug(
            "Loaded %d shapes from local file", len(local_data.get("shapes", {}))
        )
    except Exception as ex:
        logger.debug("Local load GPU shapes index failed (%s)", ex)

    # Merge: remote shapes override local
    local_shapes = local_data.get("shapes", {})
    remote_shapes = remote_data.get("shapes", {})
    merged_shapes = {}

    if local_mtime and remote_mtime:
        if remote_mtime >= local_mtime:
            logger.debug("Remote data is fresher or equal; merging remote over local.")
            merged_shapes = {**local_shapes, **remote_shapes}
        else:
            logger.debug("Local data is fresher; merging local over remote.")
            merged_shapes = {**remote_shapes, **local_shapes}
    elif remote_shapes:
        logger.debug("Only remote shapes available.")
        merged_shapes = remote_shapes
    elif local_shapes:
        logger.debug("Only local shapes available.")
        merged_shapes = local_shapes
    else:
        logger.error("No GPU shapes data found in either source.")
        merged_shapes = {}

    return GPUShapesIndex(shapes=merged_shapes)


def get_preferred_compatible_family(selected_families: set[str]) -> str:
    """
    Determines the preferred container family from a given set of container families.

    This method is used in the context of multi-model deployment to handle cases
    where models selected for deployment use different, but compatible, container families.

    It checks the input `families` set against the `CONTAINER_FAMILY_COMPATIBILITY` map.
    If a compatibility group exists that fully includes all the families in the input,
    the corresponding key (i.e., the preferred family) is returned.

    Parameters
    ----------
    families : set[str]
        A set of container family identifiers.

    Returns
    -------
    Optional[str]
        The preferred container family if all families are compatible within one group;
        otherwise, returns `None` indicating that no compatible family group was found.

    Example
    -------
    >>> get_preferred_compatible_family({"odsc-vllm-serving", "odsc-vllm-serving-v1"})
    'odsc-vllm-serving-v1'

    >>> get_preferred_compatible_family({"odsc-vllm-serving", "odsc-tgi-serving"})
    None  # Incompatible families
    """
    for preferred, compatible_list in CONTAINER_FAMILY_COMPATIBILITY.items():
        if selected_families.issubset(set(compatible_list)):
            return preferred

    return None

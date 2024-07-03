#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA utils and constants."""

import asyncio
import base64
import json
import logging
import os
import random
import re
from functools import wraps
from pathlib import Path
from string import Template
from typing import List, Union

import fsspec
import oci
from oci.data_science.models import JobRun, Model

from ads.aqua.common.enums import (
    InferenceContainerParamType,
    InferenceContainerType,
    RqsAdditionalDetails,
)
from ads.aqua.common.errors import (
    AquaFileNotFoundError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.constants import (
    AQUA_GA_LIST,
    COMPARTMENT_MAPPING_KEY,
    CONSOLE_LINK_RESOURCE_TYPE_MAPPING,
    CONTAINER_INDEX,
    MAXIMUM_ALLOWED_DATASET_IN_BYTE,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    SERVICE_MANAGED_CONTAINER_URI_SCHEME,
    SUPPORTED_FILE_FORMATS,
    TGI_INFERENCE_RESTRICTED_PARAMS,
    UNKNOWN,
    UNKNOWN_JSON_STR,
    VLLM_INFERENCE_RESTRICTED_PARAMS,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.common.auth import default_signer
from ads.common.decorator.threaded import threaded
from ads.common.extended_enum import ExtendedEnumMeta
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import copy_file, get_console_link, upload_to_os
from ads.config import AQUA_SERVICE_MODELS_BUCKET, CONDA_BUCKET_NS, TENANCY_OCID
from ads.model import DataScienceModel, ModelVersionSet

logger = logging.getLogger("ads.aqua")


class LifecycleStatus(str, metaclass=ExtendedEnumMeta):
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


def read_file(file_path: str, **kwargs) -> str:
    try:
        with fsspec.open(file_path, "r", **kwargs.get("auth", {})) as f:
            return f.read()
    except Exception as e:
        logger.debug(f"Failed to read file {file_path}. {e}")
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


def container_config_path():
    return f"oci://{AQUA_SERVICE_MODELS_BUCKET}@{CONDA_BUCKET_NS}/service_models/config"


def get_container_config():
    config = load_config(
        file_path=container_config_path(),
        config_file_name=CONTAINER_INDEX,
    )

    return config


def get_container_image(
    config_file_name: str = None, container_type: str = None
) -> str:
    """Gets the image name from the given model and container type.
    Parameters
    ----------
    config_file_name: str
        name of the config file
    container_type: str
        type of container, can be either deployment-container, finetune-container, evaluation-container

    Returns
    -------
    Dict:
        A dict of allowed configs.
    """

    config = config_file_name or get_container_config()
    config_file_name = container_config_path()

    if container_type not in config:
        raise AquaValueError(
            f"{config_file_name} does not have config details for model: {container_type}"
        )

    container_image = None
    mapping = config[container_type]
    versions = [obj["version"] for obj in mapping]
    # assumes numbered versions, update if `latest` is used
    latest = get_max_version(versions)
    for obj in mapping:
        if obj["version"] == str(latest):
            container_image = f"{obj['name']}:{obj['version']}"
            break

    if not container_image:
        raise AquaValueError(
            f"{config_file_name} is missing name and/or version details."
        )

    return container_image


def fetch_service_compartment() -> Union[str, None]:
    """
    Loads the compartment mapping json from service bucket.
    This json file has a service-model-compartment key which contains a dictionary of namespaces
    and the compartment OCID of the service models in that namespace.
    """
    config_file_name = (
        f"oci://{AQUA_SERVICE_MODELS_BUCKET}@{CONDA_BUCKET_NS}/service_models/config"
    )

    try:
        config = load_config(
            file_path=config_file_name,
            config_file_name=CONTAINER_INDEX,
        )
    except Exception as e:
        logger.debug(
            f"Config file {config_file_name}/{CONTAINER_INDEX} to fetch service compartment OCID "
            f"could not be found. \n{str(e)}."
        )
        return
    compartment_mapping = config.get(COMPARTMENT_MAPPING_KEY)
    if compartment_mapping:
        return compartment_mapping.get(CONDA_BUCKET_NS)


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


def get_model_by_reference_paths(model_file_description: dict):
    """Reads the model file description json dict and returns the base model path and fine-tuned path for
        models created by reference.

    Parameters
    ----------
    model_file_description: dict
        json dict containing model paths and objects for models created by reference.

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

    if len(models) > 0:
        # since the model_file_description json does not have a flag to identify the base model, we consider
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


def is_service_managed_container(container):
    return container and container.startswith(SERVICE_MANAGED_CONTAINER_URI_SCHEME)


def get_params_list(params: str) -> List[str]:
    """Parses the string parameter and returns a list of params.

    Parameters
    ----------
    params
        string parameters by separated by -- delimiter

    Returns
    -------
        list of params

    """
    if not params:
        return []
    return ["--" + param.strip() for param in params.split("--")[1:]]


def get_params_dict(params: Union[str, List[str]]) -> dict:
    """Accepts a string or list of string of double-dash parameters and returns a dict with the parameter keys and values.

    Parameters
    ----------
    params:
        List of parameters or parameter string separated by space.

    Returns
    -------
        dict containing parameter keys and values

    """
    params_list = get_params_list(params) if isinstance(params, str) else params
    return {
        split_result[0]: split_result[1] if len(split_result) > 1 else UNKNOWN
        for split_result in (x.split() for x in params_list)
    }


def get_combined_params(params1: str = None, params2: str = None) -> str:
    """
    Combines string of double-dash parameters, and overrides the values from the second string in the first.
    Parameters
    ----------
    params1:
        Parameter string with values
    params2:
        Parameter string with values that need to be overridden.

    Returns
    -------
        A combined list with overridden values from params2.
    """
    if not params1:
        return params2
    if not params2:
        return params1

    # overwrite values from params2 into params1
    combined_params = [
        f"{key} {value}" if value else key
        for key, value in {
            **get_params_dict(params1),
            **get_params_dict(params2),
        }.items()
    ]

    return " ".join(combined_params)


def copy_model_config(artifact_path: str, os_path: str, auth: dict = None):
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
    else:
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

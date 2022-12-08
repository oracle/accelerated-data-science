#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import tempfile
import yaml
from typing import Any, Dict, Optional, Union
from zipfile import ZipFile

from ads.common import utils
from ads.config import OCI_REGION_METADATA


def _extract_locals(
    locals: Dict[str, Any], filter_out_nulls: Optional[bool] = True
) -> Dict[str, Any]:
    """Extract arguments from local variables.
    If input dictionary contains `kwargs`, then it will not be included to
    the result dictionary.

    Properties
    ----------
    locals: Dict[str, Any]
        A dictionary, the result of `locals()` method.
        However can be any dictionary.
    filter_out_nulls: (bool, optional). Defaults to `True`.
        Whether `None` values should be filtered out from the result dict or not.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with the result values.
    """
    result = {}
    keys_to_filter_out = ("kwargs",)
    consolidated_dict = {**locals.get("kwargs", {}), **locals}
    for key, value in consolidated_dict.items():
        if key not in keys_to_filter_out and not (filter_out_nulls and value is None):
            result[key] = value
    return result


def _is_json_serializable(data: Any) -> bool:
    """Check is data input is json serialization.

    Parameters
    ----------
    data: (Any)
        data to be passed to model for prediction.

    Returns
    -------
    bool
        Whether data is json serializable.
    """
    result = True
    try:
        json.dumps(data)
    except:
        result = False
    return result


def fetch_manifest_from_conda_location(env_location: str):
    """
    Convenience method to fetch the manifest file from a conda environment.

    :param env_location: Absolute path to the environment.
    :type env_location: str
    """
    manifest_location = None
    for file in os.listdir(env_location):
        if file.endswith("_manifest.yaml"):
            manifest_location = f"{env_location}/{file}"
            break
    env = {}
    if not manifest_location:
        raise Exception(
            f"Could not locate manifest file in the provided conda environment: {env_location}. Dir Listing - "
            f"{os.listdir(env_location)}"
        )

    with open(manifest_location) as mlf:
        env = yaml.load(mlf, Loader=yaml.FullLoader)
    manifest = env["manifest"]
    return manifest


def extract_region() -> Union[str, None]:
    """Extracts region information from the environment variables.

    Returns
    -------
    Union[str, None]
        The region identifier. For example: `us-ashburn-1`.
        Returns `None` if region not extracted.
    """
    region = None
    try:
        region = json.loads(OCI_REGION_METADATA)["regionIdentifier"]
    except:
        pass
    return region


def zip_artifact(artifact_dir: str) -> str:
    """Prepares model artifacts ZIP archive.

    Parameters
    ----------
    artifact_dir: str
        Path to the model artifact.

    Returns
    -------
    str
        Path to the model artifact ZIP archive file.
    """
    if not artifact_dir:
        raise ValueError("The `artifact_dir` must be provided.")
    if not os.path.exists(artifact_dir):
        raise ValueError(f"The {artifact_dir} not exists.")
    if not os.path.isdir(artifact_dir):
        raise ValueError("The `artifact_dir` must be a folder.")

    files_to_upload = utils.get_files(artifact_dir)
    # Set delete=False when creating NamedTemporaryFile,
    # Otherwise, the file will be delete when download_artifact() close the file.
    artifact = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    # Close the file since NamedTemporaryFile() opens the file by default.
    artifact.close()
    artifact_zip_path = artifact.name

    with ZipFile(artifact_zip_path, "w") as zf:
        for matched_file in files_to_upload:
            zf.write(
                os.path.join(artifact_dir, matched_file),
                arcname=matched_file,
            )

    return artifact_zip_path

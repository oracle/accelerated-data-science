#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Dict, Optional
from zipfile import ZipFile

import yaml

from ads.common import utils
from ads.common.extended_enum import ExtendedEnum

DEPRECATE_AS_ONNX_WARNING = "This attribute `as_onnx` will be deprecated in the future. You can choose specific format by setting `model_save_serializer`."
DEPRECATE_USE_TORCH_SCRIPT_WARNING = "This attribute `use_torch_script` will be deprecated in the future. You can choose specific format by setting `model_save_serializer`."
MODEL_ARTIFACT_MANIFEST_FILE = "model_artifact_manifest.json"
MODEL_ARTIFACT_MANIFEST_VERSION = 1


class MetadataArtifactPathType(ExtendedEnum):
    """
    Enum for defining metadata artifact path type.
    Can be either local path or OSS path. It can also be the content itself.
    """

    LOCAL = "local"
    OSS = "oss"
    CONTENT = "content"


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
        env = yaml.safe_load(mlf)
    manifest = env["manifest"]
    return manifest


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
            if matched_file == MODEL_ARTIFACT_MANIFEST_FILE:
                continue
            archive_name = _normalize_artifact_path(matched_file)
            zf.write(
                os.path.join(artifact_dir, matched_file),
                arcname=archive_name,
            )
        zf.writestr(
            MODEL_ARTIFACT_MANIFEST_FILE,
            json.dumps(
                generate_model_artifact_manifest(artifact_dir, files_to_upload),
                indent=2,
                sort_keys=True,
            ),
        )

    return artifact_zip_path


def _normalize_artifact_path(name: str) -> str:
    return name.replace(os.sep, "/")


def _validate_archive_member_name(name: str):
    if not name or "\\" in name:
        raise ValueError(f"Invalid model artifact archive member: {name}")

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid model artifact archive member: {name}")


def _get_manifest_file_records(manifest: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("Model artifact manifest must be a JSON object.")

    file_records = manifest.get("files")
    if not isinstance(file_records, dict):
        raise ValueError("Model artifact manifest is missing a valid `files` map.")

    for relative_path, expected in file_records.items():
        _validate_archive_member_name(relative_path)
        if not isinstance(expected, dict):
            raise ValueError(
                f"Model artifact manifest entry `{relative_path}` is invalid."
            )
        if not isinstance(expected.get("sha256"), str) or not isinstance(
            expected.get("size"), int
        ):
            raise ValueError(
                f"Model artifact manifest entry `{relative_path}` is missing hash or size metadata."
            )

    return file_records


def _read_model_artifact_manifest(zip_file: ZipFile):
    manifest_members = [
        member
        for member in zip_file.infolist()
        if member.filename == MODEL_ARTIFACT_MANIFEST_FILE
    ]
    if not manifest_members:
        return None
    if len(manifest_members) > 1:
        raise ValueError(
            f"Model artifact archive has multiple `{MODEL_ARTIFACT_MANIFEST_FILE}` entries."
        )
    return json.loads(zip_file.read(MODEL_ARTIFACT_MANIFEST_FILE).decode("utf-8"))


def _archive_file_names(members):
    return {
        member.filename
        for member in members
        if not member.is_dir() and member.filename != MODEL_ARTIFACT_MANIFEST_FILE
    }


def _validate_archive_matches_manifest(members, manifest: Dict[str, Any]):
    expected_files = set(_get_manifest_file_records(manifest))
    archive_files = _archive_file_names(members)
    if archive_files != expected_files:
        missing = sorted(expected_files - archive_files)
        extra = sorted(archive_files - expected_files)
        details = []
        if missing:
            details.append(f"missing files: {missing}")
        if extra:
            details.append(f"extra files: {extra}")
        raise ValueError(
            "Model artifact manifest does not match archive contents"
            + (f" ({'; '.join(details)})" if details else ".")
        )


def _extract_zip_members(zip_file: ZipFile, members, target_dir: str):
    target_path = os.path.realpath(target_dir)
    for member in members:
        _validate_archive_member_name(member.filename)
        destination = os.path.realpath(os.path.join(target_dir, member.filename))
        if destination != target_path and not destination.startswith(
            target_path + os.sep
        ):
            raise ValueError(
                f"Invalid model artifact archive member: {member.filename}"
            )

    for member in members:
        destination = os.path.realpath(os.path.join(target_dir, member.filename))
        if member.is_dir():
            os.makedirs(destination, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with zip_file.open(member) as source, open(destination, "wb") as target:
            shutil.copyfileobj(source, target)
        mode = (member.external_attr >> 16) & 0o777
        if mode:
            os.chmod(destination, mode)


def safe_extract_zip(zip_file: ZipFile, target_dir: str):
    """Extracts a model artifact archive after validating member paths."""
    members = zip_file.infolist()
    for member in members:
        _validate_archive_member_name(member.filename)

    manifest = _read_model_artifact_manifest(zip_file)
    if manifest is not None:
        _validate_archive_matches_manifest(members, manifest)

    target_path = os.path.realpath(target_dir)
    parent_dir = os.path.dirname(target_path) or os.curdir
    os.makedirs(parent_dir, exist_ok=True)
    tmp_extract_dir = tempfile.mkdtemp(prefix=".ads-model-artifact-", dir=parent_dir)

    try:
        _extract_zip_members(zip_file, members, tmp_extract_dir)
        verify_model_artifact_manifest(
            tmp_extract_dir,
            strict=manifest is not None,
        )
        os.makedirs(target_path, exist_ok=True)
        for name in os.listdir(tmp_extract_dir):
            source = os.path.join(tmp_extract_dir, name)
            destination = os.path.join(target_path, name)
            if os.path.exists(destination):
                if os.path.isdir(destination) and not os.path.islink(destination):
                    shutil.rmtree(destination)
                else:
                    os.remove(destination)
            shutil.move(source, destination)
    finally:
        shutil.rmtree(tmp_extract_dir, ignore_errors=True)


def _sha256_file(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_model_artifact_manifest(artifact_dir: str, files) -> Dict[str, Any]:
    """Builds a manifest for files included in an ADS-generated artifact."""
    file_records = {}
    for matched_file in sorted(set(files)):
        if matched_file == MODEL_ARTIFACT_MANIFEST_FILE:
            continue
        archive_name = _normalize_artifact_path(matched_file)
        _validate_archive_member_name(archive_name)
        file_path = os.path.join(artifact_dir, matched_file)
        if os.path.isfile(file_path):
            file_records[archive_name] = {
                "sha256": _sha256_file(file_path),
                "size": os.path.getsize(file_path),
            }

    return {
        "version": MODEL_ARTIFACT_MANIFEST_VERSION,
        "generated_by": "oracle-ads",
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "files": file_records,
    }


def _iter_model_artifact_files(artifact_dir: str):
    for root, _, files in os.walk(artifact_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, artifact_dir).replace(os.sep, "/")
            if relative_path != MODEL_ARTIFACT_MANIFEST_FILE:
                yield relative_path


def verify_model_artifact_manifest(artifact_dir: str, strict: bool = False) -> bool:
    """Verifies artifact files when an ADS manifest is present."""
    manifest_path = os.path.join(artifact_dir, MODEL_ARTIFACT_MANIFEST_FILE)
    if not os.path.exists(manifest_path):
        if strict:
            raise ValueError(
                f"Model artifact manifest `{MODEL_ARTIFACT_MANIFEST_FILE}` was not found."
            )
        return False

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    file_records = _get_manifest_file_records(manifest)
    expected_files = set(file_records)
    actual_files = set(_iter_model_artifact_files(artifact_dir))
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        details = []
        if missing:
            details.append(f"missing files: {missing}")
        if extra:
            details.append(f"extra files: {extra}")
        raise ValueError(
            "Model artifact manifest does not match extracted files"
            + (f" ({'; '.join(details)})" if details else ".")
        )

    for relative_path, expected in file_records.items():
        file_path = os.path.join(artifact_dir, relative_path)
        if not os.path.exists(file_path):
            raise ValueError(f"Model artifact file `{relative_path}` is missing.")
        if os.path.getsize(file_path) != expected.get("size"):
            raise ValueError(f"Model artifact file `{relative_path}` size mismatch.")
        if _sha256_file(file_path) != expected.get("sha256"):
            raise ValueError(f"Model artifact file `{relative_path}` hash mismatch.")

    return True

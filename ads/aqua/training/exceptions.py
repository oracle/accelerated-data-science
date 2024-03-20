#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""Contains exceptions classes for training job.

Exit codes are defined in the exceptions.
1       : Unhandled errors
30-39   : Error for setting up training.
40-59   : Specific errors for model training/fine-tuning.
60      : Unhandled error with OCI API calls.
61      : Unhandled error with OCI API calls to Data Science service.
62-79   : Errors with OCI API calls to Data Science service.
80-98   : Errors with OCI API calls to other services.
100     : GPU out of memory.
101-119 : Other infrastructure related errors.
"""

import getpass
import inspect
import logging
import os
import traceback

import oci


USER_HOME = os.environ.get("HOME", f"/home/{getpass.getuser()}")
EXIT_CODE_FILENAME = os.path.abspath(
    os.path.expanduser(os.path.join(USER_HOME, "aqua_ft_exit_code"))
)
logger = logging.getLogger(__name__)


class TrainingExit(SystemExit):
    """Base exception for exiting training process.
    Make sure traceback is logged/printed.
    Unlike regular Exception, traceback of SystemExit is not printed by default.
    """

    code = 1
    reason = ""

    def __init__(self, code: int = None) -> None:
        if isinstance(code, int):
            self.code = code
        self.write_exit_code()
        super().__init__(self.code)

    def write_exit_code(self):
        """Writes the exit code to a file."""
        # Write the exist code only if the file does not exist.
        if os.path.exists(EXIT_CODE_FILENAME):
            logger.debug(
                "File %s exists. Skipped writting exit code.", EXIT_CODE_FILENAME
            )
            return
        try:
            logger.debug("Writting exit code to %s", EXIT_CODE_FILENAME)
            with open(EXIT_CODE_FILENAME, "w", encoding="utf-8") as f:
                f.write(str(self.code))
        except Exception:
            logger.error(
                "Error occurred when saving exit code to %s", EXIT_CODE_FILENAME
            )
            traceback.print_exc()


class TrustRemoteCodeErrorExit(TrainingExit):
    """Error when the model contains custom code while `trust_remote_code` is not set to True."""

    code = 30
    reason = (
        "The model contains custom code which must be executed in order to correctly load the model."
        "However, `trust_remote_code` is not enabled."
    )


class DistributedTrainingLoggingError(TrainingExit):
    """Error when fetching job run logs for setting up distributed training."""

    code = 31

    reason = {
        "Unable to get logs for distributed training. "
        "Logging is required for distributed training. "
        "Please make sure the job run has permission to access logs."
    }


class ObjectStoragePathInvalid(TrainingExit):
    """Error when object storage path is not valid or not authorized."""

    code = 32

    reason = {
        "Unable to parse object storage path. "
        "Please make sure the path is formatted as oci://bucket@namespace/prefix, "
        "and the job run has permission to access it."
    }


class TrainingDataNotSpecified(TrainingExit):
    """Error when training data is not specified."""

    code = 40

    reason = "Training data is not specified."


class DataNotFoundErrorExit(TrainingExit):
    """Error when reading training data."""

    code = 41
    reason = (
        "Unable to read training data. "
        "Please make sure the file exist and the training job has permission to access the file."
    )

    def __init__(self, code: int = None, filename: str = None) -> None:
        if filename:
            self.reason += f"\n{filename}"
        super().__init__(code)


class NotEnoughDataErrorExit(TrainingExit):
    """Error when training data does not have enough rows."""

    code = 42

    MIN_ROWS = 50

    reason = (
        "There is not enough training data. "
        + f"Please use training data with at least {MIN_ROWS} rows."
    )


class InvalidDataFormatErrorExit(TrainingExit):
    """Error when training data format is invalid."""

    code = 43

    reason = (
        "Unable to load training data. "
        "Please provide training data in JSONL format with 'prompt' and 'completion' fields/columns."
    )


class MissingDataColumnErrorExit(TrainingExit):
    """Error when training data is missing required columns."""

    code = 44

    reason = (
        "Missing required field/column in training data. "
        "Please provide training data in JSONL with 'prompt' and 'completion' fields/columns."
    )


class InvalidDataValueErrorExit(TrainingExit):
    """Error when training data contain non-string values."""

    code = 45

    reason = (
        "Training data contain non-string values. "
        "Please provide training data with only strings in JSONL format."
    )


class OutputPathInvalidExit(TrainingExit):
    """Error when ouptut path is not OCI object storage path."""

    code = 50

    reason = "Output URI must be oci://bucket@namespace/prefix path."


class OutputPathWriteErrorExit(TrainingExit):
    """Error when output path is not writable."""

    code = 51

    reason = (
        "Unable to write the the output path. "
        "Please make sure the job run has permission to write to the ouput path."
    )


class OutputBucketNotVersionedExit(TrainingExit):
    """Error when versioning is not enabled in output bucket."""

    code = 52

    reason = (
        "The bucket for storing training output is not versioned. "
        "Please enable versioning on the bucket."
    )


class DownloadArtifactErrorExit(TrainingExit):
    """Error when downloading the model artifacts."""

    code = 53
    reason = (
        "There is an error downloading the model artifact from object storage."
        "Please make sure the job run has access to it."
    )


class ServiceErrorExit(TrainingExit):
    """Error with by OCI API call."""

    code = 60
    operation_name = ""
    target_service = ""
    reason = (
        "Error occurred when calling OCI API. "
        "Please make sure the job run has permission to access the resource."
    )

    def __init__(
        self, code: int = None, service: str = None, operation: str = None
    ) -> None:
        if operation:
            self.operation_name = operation
        if service:
            self.target_service = service
        super().__init__(code)


class DataScienceServiceErrorExit(ServiceErrorExit):
    """Error with by OCI API call to data science service."""

    code = 61
    target_service = "data_science"
    reason = (
        "Error occurred when calling OCI Data Science API. "
        "Please make sure the job run has permission to access the resource."
    )


class GetBaseModelErrorExit(DataScienceServiceErrorExit):
    """Error when reading base model."""

    code = 62
    reason = (
        "Failed to read base model from Model Catalog."
        "Please make sure the job run has permission to read the model."
    )


class GetOutputModelErrorExit(DataScienceServiceErrorExit):
    """Error when reading output model."""

    code = 63
    reason = (
        "Failed to read output model from Model Catalog."
        "Please make sure the job run has permission to read the model."
    )


class CreateModelErrorExit(DataScienceServiceErrorExit):
    """Error when creating the model."""

    code = 64
    operation_name = "create_model"
    reason = (
        "Failed to create a new model in Model Catalog."
        "Please make sure the job run has permission to create new model."
    )


class UpdateModelErrorExit(DataScienceServiceErrorExit):
    """Error when updating the model."""

    code = 65
    operation_name = "update_model"
    reason = (
        "Failed to update model in Model Catalog."
        "Please make sure the job run has permission to update the model."
    )


class UploadModelArtifactErrorExit(DataScienceServiceErrorExit):
    """Error when uploading model artifacts."""

    code = 66
    operation_name = "create_model_artifact"
    reason = (
        "Failed to upload model artifact to Model Catalog."
        "Please make sure the model is in active state without existing artifact, "
        "and the job run has permission to upload the model artifact."
    )


class GetJobErrorExit(DataScienceServiceErrorExit):
    """Error when getting job details."""

    code = 67
    operation_name = "get_job"
    reason = (
        "Failed to get job details."
        "Please make sure job run has permission to read job details."
    )


class GetJobRunErrorExit(DataScienceServiceErrorExit):
    """Error when getting job details."""

    code = 68
    operation_name = "get_job_run"
    reason = (
        "Failed to get job run details."
        "Please make sure job run has permission to read job run details."
    )


class GetSubnetErrorExit(ServiceErrorExit):
    """Error when getting subnet details."""

    code = 81
    operation_name = "get_subnet"
    reason = (
        "Failed to get subnet details. "
        "Subnet details are required for distributed training. "
        "Please make sure the job run has permission to read subnet details."
    )


class SearchLogErrorExit(ServiceErrorExit):
    """Error when searching logs."""

    code = 82
    target_service = "log_search"
    operation_name = "search_logs"
    reason = (
        "Failed to search logs. Logs are required for distributed training. "
        "Please make sure the job run has permission to search logs."
    )


class LogManagementErrorExit(ServiceErrorExit):
    """Error when calling log management APIs."""

    code = 91
    target_service = "logging_management"
    reason = (
        "Failed to access logging service. Logs are required for distributed training. "
        "Please make sure the job run has permission to use logging service."
    )


class ObjectStorageErrorExit(ServiceErrorExit):
    """Error when calling log management APIs."""

    code = 92
    target_service = "object_storage"
    reason = (
        "Failed to access object storage service. "
        "Please make sure the job run has permission to access the object storage."
    )


class OutOfMemoryErrorExit(TrainingExit):
    """Error when GPU Out-Of-Memory."""

    code = 100
    reason = (
        "CUDA out of memory. "
        "GPU does not have enough memory to train the model. "
        "Please use a shape with more GPU memory."
    )


class NetworkTimeoutErrorExit(TrainingExit):
    """Error due to network connection timeout."""

    code = 101
    reason = (
        "Network connection timeout. "
        "If you are using custom networking, "
        "please check the VCN/subnet setting to make sure service or NAT gateway is configured correctly. "
        "This could also be a temporary issue."
    )


class GPUNotAvailableErrorExit(TrainingExit):
    """Error when GPU is not available."""

    code = 102
    reason = "GPU is not available."


def exception_list():
    """Returns a list of exceptions that may cause the training to exit with specific code."""
    return [
        obj
        for obj in globals().values()
        if inspect.isclass(obj) and issubclass(obj, TrainingExit)
    ]


def exception_dict():
    """Returns a dict of exceptions that may cause the training to exit with specific code."""
    return {
        key: obj
        for key, obj in globals().items()
        if inspect.isclass(obj) and issubclass(obj, TrainingExit)
    }


def exit_code_dict():
    """Returns a dict of exceptions that may cause the training to exit with specific code.
    The keys will be the exit codes.
    The values will be the exception class.
    """
    return {
        ex.code: ex
        for ex in globals().values()
        if inspect.isclass(ex) and issubclass(ex, TrainingExit)
    }


def operation_dict():
    """Returns a dict of ServiceErrorExit exceptions with non-empty operation name,
    that may cause the training to exit with specific code.

    The keys will be the operation_name of the expcetion.
    The values will be the exception class.
    """
    return {
        ex.operation_name: ex
        for ex in globals().values()
        if inspect.isclass(ex)
        and issubclass(ex, ServiceErrorExit)
        and ex.operation_name
    }


def service_dict():
    """Returns a dict of ServiceErrorExit exceptions that are not for specific operation.
    The keys will be the target_service of the expcetion.
    The values will be the exception class.
    """
    return {
        ex.target_service: ex
        for ex in globals().values()
        if inspect.isclass(ex)
        and issubclass(ex, ServiceErrorExit)
        and ex.target_service
        and not ex.operation_name
    }


def log_reason_and_print_exc(exit_ex: TrainingExit = None):
    """Log the exit reason and raise a TrainingExit exception to exit with exit code."""
    logger.error(traceback.format_exc())
    logger.critical(exit_ex.reason)


def prepare_service_exit(ex: oci.exceptions.ServiceError):
    """Raise ServiceErrorExit base on OCI ServiceError."""
    operations = operation_dict()
    services = service_dict()
    if ex.operation_name in operations:
        exit_ex = operations[ex.operation_name]()
    elif ex.target_service in services:
        exit_ex = services[ex.target_service]()
    else:
        exit_ex = ServiceErrorExit(
            service=ex.target_service, operation=ex.operation_name
        )
    log_reason_and_print_exc(exit_ex=exit_ex)
    return exit_ex

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to minimize the number of errors of the model post-deployment process.
The model provides a simple testing harness to ensure that model artifacts are
thoroughly tested before being saved to the model catalog.

Classes
--------
ModelIntrospect
    Class to introspect model artifacts.

Examples
--------
>>> model_introspect = ModelIntrospect(artifact=model_artifact)
>>> model_introspect()
... Test key         Test name            Result              Message
... ----------------------------------------------------------------------------
... test_key_1       test_name_1          Passed              test passed
... test_key_2       test_name_2          Not passed          some error occured
>>> model_introspect.status
... Passed
"""
from enum import Enum
import errno
import importlib
import json
import os
from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import List

import pandas as pd
from ads.model.model_metadata import MetadataTaxonomyKeys
from ads.common.object_storage_details import ObjectStorageDetails


class IntrospectionNotPassed(ValueError):
    pass


class TEST_STATUS(str):
    PASSED = "Passed"
    NOT_PASSED = "Failed"
    NOT_TESTED = "Skipped"


_PATH_TO_MODEL_ARTIFACT_VALIDATOR = "ads.model.model_artifact_boilerplate.artifact_introspection_test.model_artifact_validate"
_INTROSPECT_METHOD_NAME = "validate_artifact"
_INTROSPECT_RESULT_FILE_NAME = "test_json_output.json"


class _PRINT_COLUMNS(Enum):
    KEY = "Test key"
    CASE = "Test name"
    RESULT = "Result"
    MESSAGE = "Message"


class _TEST_COLUMNS(str):
    CATEGORY = "category"
    DESCRIPTION = "description"
    ERROR_MSG = "error_msg"
    SUCCESS = "success"


_TEST_STATUS_MAP = {
    True: TEST_STATUS.PASSED,
    False: TEST_STATUS.NOT_PASSED,
    None: TEST_STATUS.NOT_TESTED,
}


class _ERROR_MESSAGES(str):
    MODEL_ARTIFACT_NOT_SET = "A model artifact is required."
    MODEL_ARTIFACT_INVALID_TYPE = (
        "The model artifact must be an instance of the class ModelArtifact."
    )


@dataclass
class PrintItem:
    """Class represents the model introspection print item."""

    key: str = ""
    case: str = ""
    result: str = ""
    message: str = ""

    def to_list(self) -> List[str]:
        """Converts instance to a list representation.

        Returns
        -------
        List[str]
            The instance in a list representation.
        """
        return [self.key, self.case, self.result, self.message]


class Introspectable(ABC):
    """Base class that represents an introspectable object."""

    pass


class ModelIntrospect:
    """Class to introspect model artifacts.

    Parameters
    ----------
    status: str
        Returns the current status of model introspection.
        The possible variants: `Passed`, `Not passed`, `Not tested`.
    failures: int
        Returns the number of failures of introspection result.

    Methods
    -------
    run(self) -> None
        Invokes model artifacts introspection.
    to_dataframe(self) -> pd.DataFrame
        Serializes model introspection result into a DataFrame.

    Examples
    --------
    >>> model_introspect = ModelIntrospect(artifact=model_artifact)
    >>> result = model_introspect()
    ... Test key         Test name            Result              Message
    ... ----------------------------------------------------------------------------
    ... test_key_1       test_name_1          Passed              test passed
    ... test_key_2       test_name_2          Not passed          some error occured
    """

    def __init__(self, artifact: Introspectable):
        """Initializes the Model Introspect.

        Parameters
        ----------
        artifact: Introspectable
            The instance of ModelArtifact object.

        Raises
        ------
            ValueError: If model artifact object not provided.
            TypeError: If provided input paramater not a ModelArtifact instance.
        """
        if not artifact:
            raise ValueError(_ERROR_MESSAGES.MODEL_ARTIFACT_NOT_SET)

        if not isinstance(artifact, Introspectable):
            raise TypeError(_ERROR_MESSAGES.MODEL_ARTIFACT_INVALID_TYPE)

        self._artifact = artifact
        self._reset()

    def _reset(self) -> None:
        """Resets test result to initial state."""
        self._status = TEST_STATUS.NOT_TESTED
        self._result = None
        self._prepared_result = []

    def _save_result_to_artifacts(self) -> None:
        """Saves introspection result into the model artifacts folder.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
            FileNotFoundError: If path to model artifacts does not exist.
        """
        artifact_dir = (
            self._artifact.artifact_dir
            if not ObjectStorageDetails.is_oci_path(self._artifact.artifact_dir)
            else self._artifact.local_copy_dir
        )
        if not os.path.isdir(artifact_dir):
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), artifact_dir
            )

        output_file = f"{artifact_dir}/{_INTROSPECT_RESULT_FILE_NAME}"
        with open(output_file, "w") as f:
            json.dump(self._result, f, indent=4)

    def _save_result_to_metadata(self) -> None:
        """Saves the result of introspection to the model metadata."""
        self._artifact.metadata_taxonomy[
            MetadataTaxonomyKeys.ARTIFACT_TEST_RESULT
        ].update(value=self._result)

    def _import_and_run_validator(self) -> None:
        """Imports and run model artifact validator.

        The validator provided as one of the modules of model artifacts boilerplate.
        The importlib API is used to load validator and to invoke test method.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        FileNotFoundError: If path to model artifacts does not exist.
        """
        artifact_dir = (
            self._artifact.artifact_dir
            if not ObjectStorageDetails.is_oci_path(self._artifact.artifact_dir)
            else self._artifact.local_copy_dir
        )
        if not os.path.isdir(artifact_dir):
            raise FileNotFoundError(
                errno.ENOENT,
                os.strerror(errno.ENOENT),
            )

        module = importlib.import_module(_PATH_TO_MODEL_ARTIFACT_VALIDATOR)
        importlib.reload(module)
        method = getattr(module, _INTROSPECT_METHOD_NAME)
        params = {"artifact": artifact_dir}
        test_result, _ = method(**params)

        self._status = _TEST_STATUS_MAP.get(test_result)
        self._result = copy(module.TESTS)
        self._prepared_result = self._prepare_result()

    @property
    def status(self) -> str:
        """Gets the current status of model introspection."""
        return self._status

    def run(self) -> pd.DataFrame:
        """Invokes introspection.

        Returns
        -------
        pd.DataFrame
           The introspection result in a DataFrame format.
        """
        self._reset()
        self._import_and_run_validator()
        self._save_result_to_metadata()
        self._save_result_to_artifacts()
        return self.to_dataframe()

    def _prepare_result(self) -> List[PrintItem]:
        """Prepares introspection result information to display to user.

        Returns
        -------
        List[PrintItem]
            The list of prepared to print data items.
        """
        if not self._result:
            return []

        result = []
        for key, item in self._result.items():
            error_msg = (
                item.get(_TEST_COLUMNS.ERROR_MSG)
                if (
                    item.get(_TEST_COLUMNS.SUCCESS) == False
                    or (
                        item.get(_TEST_COLUMNS.SUCCESS) == None
                        and "WARNING" in item.get(_TEST_COLUMNS.ERROR_MSG, "")
                    )
                )
                else ""
            )
            result.append(
                PrintItem(
                    key,
                    item.get(_TEST_COLUMNS.DESCRIPTION, ""),
                    _TEST_STATUS_MAP.get(item.get(_TEST_COLUMNS.SUCCESS)),
                    error_msg,
                )
            )
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Serializes model introspection result into a DataFrame.

        Returns
        -------
        `pandas.DataFrame`
            The model introspection result in a DataFrame representation.
        """
        return (
            pd.DataFrame(
                (item.to_list() for item in self._prepared_result),
                columns=[item.value for item in _PRINT_COLUMNS],
            )
            .sort_values(by=[_PRINT_COLUMNS.KEY.value, _PRINT_COLUMNS.CASE.value])
            .reset_index(drop=True)
        )

    @property
    def failures(self) -> int:
        """Calculates the number of failures.

        Returns
        -------
        int
            The number of failures.
        """
        return len(
            [
                item
                for item in self._prepared_result
                if item.result == TEST_STATUS.NOT_PASSED
            ]
        )

    def __call__(self) -> pd.DataFrame:
        """Invokes introspection.

        Returns
        -------
        pd.DataFrame
           The introspection result in a DataFrame format.
        """
        return self.run()

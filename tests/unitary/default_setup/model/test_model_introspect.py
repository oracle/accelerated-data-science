#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit Tests for model introspection module."""

from unittest import TestCase
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
from ads.common.model_artifact import ModelArtifact
from ads.model.model_introspect import (
    _ERROR_MESSAGES,
    _INTROSPECT_METHOD_NAME,
    _INTROSPECT_RESULT_FILE_NAME,
    _PRINT_COLUMNS,
    _TEST_STATUS_MAP,
    TEST_STATUS,
    ModelIntrospect,
    PrintItem,
)
from ads.model.model_metadata import MetadataTaxonomyKeys


class TestModelIntrospect(TestCase):
    """Unittests for ModelIntrospect class."""

    def setUp(self) -> None:
        """Sets up the test case."""
        super().setUp()

        self.mock_path_to_model_artifacts = "/path/to/model_artifacts/"
        self.mock_model_artifact = ModelArtifact(
            self.mock_path_to_model_artifacts,
            reload=False,
            create=False,
            ignore_deployment_error=True,
        )

        self.mock_prepared_result = [
            PrintItem("key1", "description1", _TEST_STATUS_MAP.get(True), ""),
            PrintItem(
                "key2", "description2", _TEST_STATUS_MAP.get(False), "error_msg2"
            ),
            PrintItem("key3", "description3", _TEST_STATUS_MAP.get(None), ""),
        ]

        self.mock_model_introspect_result = {
            "key1": {
                "category": "category1",
                "description": "description1",
                "error_msg": "error_msg1",
                "success": True,
            },
            "key2": {
                "category": "category2",
                "description": "description2",
                "error_msg": "error_msg2",
                "success": False,
            },
            "key3": {
                "category": "category3",
                "description": "description3",
                "error_msg": "error_msg3",
            },
        }

    def tearDown(self):
        super().tearDown()

    def test_init_fail(self):
        """Ensures initializing model introspection fails when wrong input parameters provided."""

        with pytest.raises(ValueError) as err:
            ModelIntrospect(artifact=None)
        assert _ERROR_MESSAGES.MODEL_ARTIFACT_NOT_SET == str(err.value)

        with pytest.raises(TypeError) as err:
            ModelIntrospect(artifact={"key"})
        assert _ERROR_MESSAGES.MODEL_ARTIFACT_INVALID_TYPE == str(err.value)

    @patch.object(ModelIntrospect, "_reset")
    def test_init_success(self, mock_reset):
        """Ensures initializing model introspection passes when valid input parameters provided."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        assert self.mock_model_artifact == mock_model_introspect._artifact
        mock_reset.assert_called()

    def test_reset(self):
        """Tests reseting introspection result to initial state."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        mock_model_introspect._status = TEST_STATUS.PASSED
        mock_model_introspect._result = {}
        mock_model_introspect._reset()
        assert mock_model_introspect._status == TEST_STATUS.NOT_TESTED
        assert mock_model_introspect._result == None
        assert mock_model_introspect._prepared_result == []

    def test_status(self):
        """Tests getting the current status of model introspection."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        assert mock_model_introspect.status == TEST_STATUS.NOT_TESTED

    @patch("os.path.isdir", return_value=False)
    def test_save_result_to_artifacts_fail(self, mock_isdir):
        """Ensures saving introspection result fails when invalid destination folder provided."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        with pytest.raises(FileNotFoundError):
            mock_model_introspect._save_result_to_artifacts()

    @patch("os.path.isdir", return_value=True)
    def test_save_result_to_artifacts_success(self, mock_isdir):
        """Ensures saving introspection result passes when valid destination folder provided."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        mock_result_data = {"test": True}
        mock_output_file = f"{mock_model_introspect._artifact.artifact_dir}/{_INTROSPECT_RESULT_FILE_NAME}"
        mock_model_introspect._result = mock_result_data

        open_mock = mock_open()
        with patch("ads.model.model_introspect.open", open_mock, create=True):
            mock_model_introspect._save_result_to_artifacts()

        open_mock.assert_called_with(mock_output_file, "w")
        open_mock.return_value.write.assert_called_with("}")

    def test_save_result_to_metadata(self):
        """Tests saving the result of introspection to the model metadata."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)

        mock_result_data = {"test": True}
        mock_model_introspect._result = mock_result_data
        mock_model_introspect._save_result_to_metadata()

        assert (
            self.mock_model_artifact.metadata_taxonomy[
                MetadataTaxonomyKeys.ARTIFACT_TEST_RESULT
            ].value
            == mock_result_data
        )

    @patch.object(ModelIntrospect, "_reset")
    @patch.object(ModelIntrospect, "_import_and_run_validator")
    @patch.object(ModelIntrospect, "_save_result_to_metadata")
    @patch.object(ModelIntrospect, "_save_result_to_artifacts")
    def test_introspect(
        self,
        mock_reset,
        mock_import_and_run_validator,
        mock_save_result_to_metadata,
        mock_save_result_to_artifacts,
    ):
        """Tests invoking model artifacts introspection."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        mock_model_introspect._status = TEST_STATUS.PASSED
        mock_model_introspect._prepared_result = self.mock_prepared_result
        expected_result = (
            pd.DataFrame(
                (item.to_list() for item in self.mock_prepared_result),
                columns=[item.value for item in _PRINT_COLUMNS],
            )
            .sort_values(by=[_PRINT_COLUMNS.KEY.value, _PRINT_COLUMNS.CASE.value])
            .reset_index(drop=True)
        )
        result = mock_model_introspect()
        mock_reset.assert_called()
        mock_import_and_run_validator.assert_called()
        mock_save_result_to_artifacts.assert_called()
        mock_save_result_to_metadata.assert_called()
        assert pd.DataFrame.equals(expected_result, result)

    @patch.object(ModelIntrospect, "run")
    def test_callable(self, mock_introspect):
        """Ensures model introspect instance is callable."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        mock_model_introspect._status = TEST_STATUS.PASSED
        mock_model_introspect()
        mock_introspect.assert_called()

    def test_prepare_result(self):
        """Tests preparing introspection result information to display to user."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)

        mock_model_introspect._result = None
        test_result = mock_model_introspect._prepare_result()
        assert test_result == []

        mock_model_introspect._result = self.mock_model_introspect_result
        test_result = mock_model_introspect._prepare_result()
        assert test_result == self.mock_prepared_result

    @patch("os.path.isdir", return_value=False)
    def test_import_and_run_validator_fail(self, mock_isdir):
        """Ensures that importing and runing model artifact validator fails."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        with pytest.raises(FileNotFoundError):
            mock_model_introspect._import_and_run_validator()

    @patch("os.path.isdir", return_value=True)
    @patch("importlib.reload")
    def test_import_and_run_validator_success(self, mock_isdir, mock_importlib_reload):
        """Tests importing and runing model artifact validator."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)

        mock_module = MagicMock()
        method = MagicMock(return_value=[False, True])
        mock_module.configure_mock(**{_INTROSPECT_METHOD_NAME: method})
        mock_module.TESTS = self.mock_model_introspect_result

        with patch(
            "importlib.import_module", return_value=mock_module
        ) as mock_import_module:
            mock_model_introspect._import_and_run_validator()
            mock_import_module.assert_called()
            mock_importlib_reload.assert_called()
            params = {"artifact": self.mock_path_to_model_artifacts[:-1]}
            method.assert_called_with(**params)
            assert mock_model_introspect._status == TEST_STATUS.NOT_PASSED
            assert mock_model_introspect._result == self.mock_model_introspect_result

    def test_to_dataframe(self):
        """Tests serializing model introspection result into a DataFrame."""
        mock_model_introspect = ModelIntrospect(artifact=self.mock_model_artifact)
        mock_model_introspect._prepared_result = self.mock_prepared_result
        expected_result = (
            pd.DataFrame(
                (item.to_list() for item in self.mock_prepared_result),
                columns=[item.value for item in _PRINT_COLUMNS],
            )
            .sort_values(by=[_PRINT_COLUMNS.KEY.value, _PRINT_COLUMNS.CASE.value])
            .reset_index(drop=True)
        )
        assert pd.DataFrame.equals(
            expected_result, mock_model_introspect.to_dataframe()
        )


class TestPrintItem(TestCase):
    """Unittests for PrintItem class."""

    def setUp(self) -> None:
        super().setUp()
        self.mock_print_item = PrintItem(
            "test_key", "test_case", "test_result", "test_message"
        )

    def test_to_list(self):
        """Tests converting instance to a list representation."""
        expected_result = ["test_key", "test_case", "test_result", "test_message"]
        assert self.mock_print_item.to_list() == expected_result

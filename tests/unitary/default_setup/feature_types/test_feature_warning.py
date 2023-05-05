#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit Tests for feature warning module."""
from unittest import mock
from unittest import TestCase
import pytest
import pandas as pd
from ads.feature_engineering.feature_type.handler.feature_warning import (
    FeatureWarning,
    _validate_warning_handler,
)
from ads.feature_engineering.exceptions import WarningAlreadyExists, WarningNotFound


class TestFeatureWarning(TestCase):
    """Unittest for FeatureWarning class."""

    def setUp(self):
        """Sets up the test case."""
        super(TestFeatureWarning, self).setUp()
        self.mock_handler_result = pd.DataFrame(
            [["warning", "message", "metric", "value"]],
            columns=["Warning", "Message", "Metric", "Value"],
        )
        self.mock_handler_result1 = pd.DataFrame(
            [["message", "warning", "value", "metric"]],
            columns=["Message", "Warning", "Value", "Metric"],
        )
        self.mock_warning = FeatureWarning()
        self.mock_warning_name1 = "test_warning_name1"
        self.mock_warning_name2 = "test_warning_name2"
        self.mock_warning_name3 = "test_warning_name3"
        self.mock_warning_handler1 = mock.MagicMock(
            return_value=self.mock_handler_result
        )
        self.mock_warning_handler1.__name__ = "mock_warning_handler1"
        self.mock_warning_handler2 = mock.MagicMock(
            return_value=self.mock_handler_result1
        )
        self.mock_warning_handler2.__name__ = "mock_warning_handler2"
        self.mock_data = pd.Series((1, 2, 3))

    def test_init(self):
        """Tests initialization of FeatureWarning instance."""
        warning = FeatureWarning()
        assert isinstance(warning, FeatureWarning)
        assert warning._data is None
        assert warning._handlers == {}

    def test_register_success(self):
        """Ensures new warning with provided handler can be registered."""
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        test_handlers = {self.mock_warning_name1: self.mock_warning_handler1}
        assert self.mock_warning._handlers == test_handlers

    def test_register_many_success(self):
        """Ensures a few warnings with provided handlers can be registered."""
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        self.mock_warning.register(self.mock_warning_name2, self.mock_warning_handler2)
        test_handlers = {
            self.mock_warning_name1: self.mock_warning_handler1,
            self.mock_warning_name2: self.mock_warning_handler2,
        }
        assert self.mock_warning._handlers == test_handlers

    def test_register_success_with_replace_flag(self):
        """Ensures new warning with provided handler can be overwritten with replace flag."""
        test_handlers = {self.mock_warning_name1: self.mock_warning_handler1}
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        self.mock_warning.register(
            self.mock_warning_name1, self.mock_warning_handler1, replace=True
        )
        assert self.mock_warning._handlers == test_handlers

    def test_register_fail(self):
        """Ensures registering warning fails when input parameters not provided or have wrong format."""
        with pytest.raises(ValueError) as exc:
            self.mock_warning.register(name=None, handler=None)
        assert str(exc.value) == "Warning name is not provided."

        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        with pytest.raises(WarningAlreadyExists) as exc:
            self.mock_warning.register(
                self.mock_warning_name1, self.mock_warning_handler1
            )
        assert str(exc.value) == str(WarningAlreadyExists(self.mock_warning_name1))

        with pytest.raises(ValueError) as exc:
            self.mock_warning.register(self.mock_warning_name2, handler=None)
        assert str(exc.value) == "Handler is not provided."

        with pytest.raises(TypeError) as exc:
            self.mock_warning.register(
                self.mock_warning_name2, handler="not a function"
            )
        assert str(exc.value) == "Handler should be a function."

    def test_unregister_success(self):
        """Ensures a warning can be unregistered."""
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        test_handlers = {self.mock_warning_name1: self.mock_warning_handler1}
        assert self.mock_warning._handlers == test_handlers
        self.mock_warning.unregister(self.mock_warning_name1)
        assert self.mock_warning._handlers == {}

    def test_unregister_fail(self):
        """Ensures unregistering warning fails when input parameters not provided or have wrong format."""
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        with pytest.raises(ValueError) as exc:
            self.mock_warning.unregister(name=None)
        assert str(exc.value) == "Warning name is not provided."
        with pytest.raises(ValueError) as exc:
            self.mock_warning.unregister(name=self.mock_warning_name2)
        assert str(exc.value) == str(WarningNotFound(self.mock_warning_name2))

    def test_registered(self):
        """Tests getting the list of registered warnings."""
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)

        expected_result = pd.DataFrame(
            [(self.mock_warning_name1, self.mock_warning_handler1.__name__)],
            columns=["Warning", "Handler"],
        )
        assert pd.DataFrame.equals(expected_result, self.mock_warning.registered())

    def test_bind_data(self):
        """Tests binding data to the feature warning instance."""
        test_data = (1, 2, 3)
        self.mock_warning._bind_data(test_data)
        assert test_data == self.mock_warning._data

    def test_proccess_warning_registered_fail(self):
        """Ensures processing of registered warnings fails if data not provided."""
        with pytest.raises(ValueError) as exc:
            self.mock_warning._process()
        assert str(exc.value) == "Data is not provided."

        self.mock_warning._bind_data(self.mock_data)
        with pytest.raises(ValueError) as exc:
            self.mock_warning.register(
                self.mock_warning_name1,
                mock.MagicMock(side_effect=ValueError("division by zero")),
            )
            self.mock_warning._process()
        assert str(exc.value) == (
            f"An error occurred while executing the '{self.mock_warning_name1}'. "
            f"Details: division by zero."
        )

        self.mock_warning.unregister(self.mock_warning_name1)
        expected_columns = ["Warning", "Message", "Metric", "Value"]
        with pytest.raises(ValueError) as exc:
            self.mock_warning.register(
                self.mock_warning_name1, mock.MagicMock(return_value=True)
            )
            self.mock_warning._process()
        assert str(exc.value) == (
            f"An error occurred while executing the '{self.mock_warning_name1}'. "
            f"Details: '{self.mock_warning_name1}' should return a DataFrame "
            f"with columns: {expected_columns}."
        )

    def test_proccess_warning_registered_success(self):
        """Tests processing of registered warnings."""
        self.mock_warning._bind_data(self.mock_data)
        assert self.mock_warning._process() is None
        self.mock_warning.register(self.mock_warning_name1, self.mock_warning_handler1)
        # registering warning which returns None
        self.mock_warning.register(self.mock_warning_name2, lambda x: None)
        # registering warning which returns empty DataFrame
        self.mock_warning.register(self.mock_warning_name3, lambda x: pd.DataFrame())
        assert pd.DataFrame.equals(
            self.mock_handler_result, self.mock_warning._process()
        )

    def test_instance_callable_with_direct_data(self):
        """Ensures that feature warning instance is callable with directly provided data."""
        with mock.patch.object(FeatureWarning, "_process"):
            self.mock_warning(self.mock_data)
            self.mock_warning._process.assert_called_with()

    def test_instance_callable_fail_when_data_not_provided(self):
        """Ensures that feature warning instance is callable and fails if data not provided."""
        with pytest.raises(ValueError) as exc:
            self.mock_warning()
        assert str(exc.value) == "Data is not provided."

    def test_instance_callable_fail_when_data_has_wrong_format(self):
        """Ensures that feature warning instance is callable and fails if data not provided."""
        with pytest.raises(TypeError) as exc:
            self.mock_warning("wrong data")
        assert str(exc.value) == "Wrong data format. Data should be Series."

    def test_get_attr(self):
        """Ensures registered warnings can be invoked like a regular methods."""
        self.mock_warning.register("good_quality_data", self.mock_warning_handler1)
        self.mock_warning.good_quality_data(self.mock_data)
        self.mock_warning_handler1.assert_called_with(self.mock_data)

    def test_validate_warning_handler_success(self):
        """Ensures feature warning validator returns True if registered handler compatible."""
        result_df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
        mock_handler = mock.MagicMock(return_value=result_df)
        assert _validate_warning_handler(mock_handler) is True

    def test_validate_warning_handler_fail(self):
        """Ensures feature warning validator returns False if registered handler not compatible."""
        result_df = pd.DataFrame([], columns=["Warning", "Message", "Metric", "Value"])
        assert _validate_warning_handler(lambda x: "not valid") is False
        assert _validate_warning_handler(lambda x: None) is False

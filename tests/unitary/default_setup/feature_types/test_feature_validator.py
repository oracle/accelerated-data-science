#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Unit Tests for feature validator."""
from logging import Handler
from unittest import TestCase, mock
from unittest.mock import patch

import pandas as pd
import pytest
from ads.feature_engineering.feature_type.handler.feature_validator import (
    FeatureValidator,
    FeatureValidatorMethod,
    ValidatorAlreadyExists,
    ValidatorNotFound,
    ValidatorWithConditionAlreadyExists,
    WrongHandlerMethodSignature,
    _prepare_condition,
)


class TestFeatureValidator(TestCase):
    """Unittest for FeatureValidator class."""

    def setUp(self):
        """Sets up the test case."""
        super(TestFeatureValidator, self).setUp()

        self.mock_result_data = pd.Series([True, True, True])
        self.mock_default_handler = mock.MagicMock(return_value=self.mock_result_data)
        self.mock_feature_validator = FeatureValidator()

    @patch.object(FeatureValidatorMethod, "register")
    def test_register_success(self, mock_register):
        """Tests registering new handlers for a feature type."""
        self.mock_feature_validator.register(
            name="test_validator", handler=self.mock_default_handler
        )
        assert "test_validator" in self.mock_feature_validator._validators

        self.mock_feature_validator.register(
            name="test_validator",
            handler=self.mock_default_handler,
            condition={"key": "value"},
        )
        mock_register.assert_called_with(
            condition={"key": "value"}, handler=self.mock_default_handler
        )

    @patch.object(FeatureValidatorMethod, "_has_condition", return_value=True)
    def test_register_fail(self, mock_has_condition):
        """Ensures registering new handlers fails in case of wrong parameters."""
        self.mock_feature_validator.register(
            name="test_validator", handler=self.mock_default_handler
        )

        with pytest.raises(ValueError) as exc:
            self.mock_feature_validator.register(name=None, handler=None)
        assert str(exc.value) == "Validator name is not provided."

        with pytest.raises(TypeError) as exc:
            self.mock_feature_validator.register(name={"key": "value"}, handler=None)
        assert str(exc.value) == "Validator name should be a string."

        with pytest.raises(ValidatorAlreadyExists) as exc:
            self.mock_feature_validator.register(name="test_validator", handler=None)
        assert str(exc.value) == "Validator test_validator already exists."

        with pytest.raises(ValueError) as exc:
            self.mock_feature_validator.register(
                name="new_test_validator", handler=None
            )
        assert str(exc.value) == "Handler is not provided."

        with pytest.raises(TypeError) as exc:
            self.mock_feature_validator.register(
                name="new_test_validator", handler="not callable"
            )
        assert str(exc.value) == "Handler should be a function."

        with pytest.raises(ValidatorWithConditionAlreadyExists):
            self.mock_feature_validator._validators[
                "new_test_validator"
            ] = FeatureValidatorMethod(self.mock_default_handler)
            self.mock_feature_validator.register(
                name="new_test_validator",
                handler="not callable",
                condition={"key": "value"},
            )
            mock_has_condition.assert_called()

    def test_unregister_success(self):
        """Tests unregistering validators."""
        self.mock_feature_validator.register(
            name="test_validator", handler=self.mock_default_handler
        )
        assert "test_validator" in self.mock_feature_validator._validators
        self.mock_feature_validator.unregister("test_validator")
        assert "test_validator" not in self.mock_feature_validator._validators

    def test_unregister_fail(self):
        """Tests unregistering validators."""
        self.mock_feature_validator.register(
            name="test_validator", handler=self.mock_default_handler
        )

        with pytest.raises(ValueError) as exc:
            self.mock_feature_validator.unregister(name=None)
        assert str(exc.value) == "Validator name is not provided."

        with pytest.raises(TypeError) as exc:
            self.mock_feature_validator.unregister(name={"key": "value"})
        assert str(exc.value) == "Validator name should be a string."

        with pytest.raises(ValidatorNotFound) as exc:
            self.mock_feature_validator.unregister(name="not_test_validator")
        assert str(exc.value) == "Validator not_test_validator not found."

    def test_registered(self):
        """Tests getting list of registered validators."""

        def default_handler(data):
            return data

        self.mock_feature_validator.register(
            name="test_validator", handler=default_handler
        )
        expected_result = pd.DataFrame(
            [
                (
                    "test_validator",
                    "()",
                    "default_handler",
                )
            ],
            columns=["Validator", "Condition", "Handler"],
        )
        assert pd.DataFrame.equals(
            expected_result, self.mock_feature_validator.registered()
        )

    def test_bind_data(self):
        """Tests binding data to the all registered validators."""
        self.mock_feature_validator.register(
            name="test_validator", handler=self.mock_default_handler
        )
        test_data = (1, 2, 3)
        self.mock_feature_validator._bind_data(test_data)
        for item in self.mock_feature_validator._validators.values():
            assert item._data == test_data


class TestFeatureValidatorMethod(TestCase):
    """Unittest for FeatureValidatorMethod class."""

    def setUp(self):
        "Sets up the test case."
        super(TestFeatureValidatorMethod, self).setUp()

        self.mock_imput_data = pd.Series(
            ["+1-202-555-0141", "+1-202-555-0198", "+1-202-555-0199"]
        )
        self.mock_result_data = pd.Series([True, True, True])
        self.mock_default_handler = mock.MagicMock(return_value=self.mock_result_data)

    def test_init_success(self):
        """Tests initialization of FeatureValidatorMethod instance with default handler."""
        feature_validator_inst = FeatureValidatorMethod(self.mock_default_handler)
        assert isinstance(feature_validator_inst, FeatureValidatorMethod)
        assert self.mock_default_handler == feature_validator_inst._default_handler

    def test_init_fail(self):
        """Tests initialization of FeatureValidatorMethod instance without default handler.

        Ensures the error occures in case of registering instance without default handler.
        """
        error_msg = "Default handler is not specified."
        with pytest.raises(ValueError) as exc:
            FeatureValidatorMethod(None)
        assert error_msg == str(exc.value)

    def test_bind_data(self):
        """Tests binding data to the feature validator instance."""
        test_data = (1, 2, 3)
        feature_validator_inst = FeatureValidatorMethod(handler=lambda: True)
        feature_validator_inst._bind_data(test_data)
        assert test_data == feature_validator_inst._data

    def test_instance_callable_with_direct_data(self):
        """Ensures that feature validator instance is callable with directly provided data."""
        with mock.patch.object(FeatureValidatorMethod, "_process"):
            feature_validator_inst = FeatureValidatorMethod(
                handler=self.mock_default_handler
            )
            feature_validator_inst(self.mock_imput_data, testparam="test_value")
            assert pd.Series.equals(feature_validator_inst._data, self.mock_imput_data)
            feature_validator_inst._process.assert_called_with(testparam="test_value")

    def test_instance_callable_with_bound_data(self):
        """Ensures that feature validator instance is callable with bound data."""
        with mock.patch.object(FeatureValidatorMethod, "_process"):
            feature_validator_inst = FeatureValidatorMethod(
                handler=self.mock_default_handler
            )
            feature_validator_inst._bind_data(self.mock_imput_data)
            feature_validator_inst(testparam="test_value")
            assert pd.Series.equals(feature_validator_inst._data, self.mock_imput_data)
            feature_validator_inst._process.assert_called_with(testparam="test_value")

    def test_instance_callable_fail_when_data_not_provided(self):
        """Ensures that feature validator instance is callable and fails if data not provided."""
        error_msg = "Data is not provided."
        with pytest.raises(ValueError) as exc:
            feature_validator_inst = FeatureValidatorMethod(
                handler=self.mock_default_handler
            )
            feature_validator_inst(testparam="test_value")
        assert error_msg == str(exc.value)

    def test_instance_callable_fail_when_data_has_wrong_format(self):
        """Ensures that feature validator instance is callable and fails if data not provided."""
        error_msg = "Wrong data format. Data should be Series."
        with pytest.raises(TypeError) as exc:
            feature_validator_inst = FeatureValidatorMethod(
                handler=self.mock_default_handler
            )
            feature_validator_inst("wrong_data", testparam="test_value")
        assert error_msg == str(exc.value)

    def test_proccess_registered(self):
        """Tests processing of registered handlers.
        Ensures suitable handler can be found and invoked for the given arguments.
        """

        def test_handler(data, param1, param2):
            return data

        test_handler1 = mock.create_autospec(test_handler)
        test_handler2 = mock.create_autospec(test_handler)

        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )

        feature_validator_inst.register(("param1", "param2"), test_handler1)
        feature_validator_inst.register(
            {"param1": "value1", "param2": "value2"}, test_handler2
        )

        feature_validator_inst(self.mock_imput_data, param1="any", param2="any")
        test_handler1.assert_called_with(
            self.mock_imput_data, param1="any", param2="any"
        )
        feature_validator_inst(self.mock_imput_data, param1="value1", param2="value2")
        test_handler2.assert_called_with(
            self.mock_imput_data, param1="value1", param2="value2"
        )

    def test_proccess_default_handler_without_args(self):
        """Tests processing of registered handlers without arguments.
        Ensures default handler used if suitable handler not found.
        """
        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )
        feature_validator_inst(self.mock_imput_data)
        self.mock_default_handler.assert_called_with(self.mock_imput_data)

    def test_proccess_default_handler_with_args(self):
        """Tests processing of registered handlers with arguments.
        Ensures default handler used if suitable handler not found.
        """

        def default_handler(series, test_arg):
            return test_arg

        test_data = "test_data"
        feature_validator_inst = FeatureValidatorMethod(handler=default_handler)
        test_result = feature_validator_inst(self.mock_imput_data, test_arg=test_data)
        assert test_result == test_data

        with pytest.raises(TypeError):
            test_result = feature_validator_inst(self.mock_imput_data)

    def test_handlers(self):
        "Tests getting the list of registered handlers"

        def default_handler(data):
            return data

        feature_validator_inst = FeatureValidatorMethod(handler=default_handler)

        assert pd.DataFrame.equals(
            pd.DataFrame(
                [
                    (
                        "()",
                        "default_handler",
                    )
                ],
                columns=["Condition", "Handler"],
            ),
            feature_validator_inst.registered(),
        )

        def test_handler1(data, param1, param2):
            return data

        def test_handler2(data, param1, param2):
            return data

        feature_validator_inst.register(("param1", "param2"), test_handler1)
        feature_validator_inst.register(
            {"param1": "value1", "param2": "value2"}, test_handler2
        )

        expected_result = pd.DataFrame(
            [
                (
                    "()",
                    "default_handler",
                ),
                ("('param1', 'param2')", "test_handler1"),
                ("{'param1': 'value1', 'param2': 'value2'}", "test_handler2"),
            ],
            columns=["Condition", "Handler"],
        )
        assert pd.DataFrame.equals(expected_result, feature_validator_inst.registered())

    def test_register_fail(self):
        """Ensures registering fails in case of parameters not provided or have wrong format."""
        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )
        with pytest.raises(ValueError) as exc:
            feature_validator_inst.register(condition=None, handler=None)
        assert str(exc.value) == "Condition not provided."

        with pytest.raises(ValueError) as exc:
            feature_validator_inst.register(condition="params", handler=None)
        assert (
            str(exc.value)
            == "Wrong format for the condition. Condition should be dict or list."
        )
        with pytest.raises(ValueError) as exc:
            feature_validator_inst.register(condition={"key": "value"}, handler=None)
        assert str(exc.value) == "Handler not provided. Handler should be a function."

    def test_register_success(self):
        """Ensures registering pass if parameters provided and have valid format."""

        def test_handler1(data, param1, param2):
            return data

        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )
        feature_validator_inst.register(
            condition=("param1", "param2"), handler=test_handler1
        )
        result_handlers = {("param1", "param2"): test_handler1}
        assert feature_validator_inst._handlers == result_handlers

    def test_unregister_fail(self):
        """Ensures unregistering fails in case of parameters not provided or have wrong format."""

        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )
        with pytest.raises(ValueError) as exc:
            feature_validator_inst.unregister(condition=None)
        assert str(exc.value) == "Condition not provided."

        with pytest.raises(ValueError) as exc:
            feature_validator_inst.unregister(condition="params")
        assert (
            str(exc.value)
            == "Wrong format for the condition. Condition should be dict or list."
        )

        with pytest.raises(ValueError) as exc:
            feature_validator_inst.unregister(condition={"key": "value"})
        assert str(exc.value) == "Condition not registered."

    def test_unregister_success(self):
        """Ensures unregistering pass if parameters provided and have valid format."""

        def test_handler1(data, param1, param2):
            return data

        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )
        feature_validator_inst.register(
            condition=("param1", "param2"), handler=test_handler1
        )

        result_handlers = {("param1", "param2"): test_handler1}
        assert feature_validator_inst._handlers == result_handlers

        feature_validator_inst.unregister(condition=("param1", "param2"))

        assert feature_validator_inst._handlers == {}

    def test_validate_handler_signature(self):
        """Tests validating registered handler."""

        def test_handler(data, param1, param2):
            return data

        feature_validator_inst = FeatureValidatorMethod(
            handler=self.mock_default_handler
        )

        assert (
            feature_validator_inst._validate_handler_signature(
                ("param1", "param2"), test_handler
            )
            is True
        )

        expected_error = WrongHandlerMethodSignature(
            test_handler.__name__,
            str(["data", "param1", "param3"]),
            str(["data", "param1", "param2"]),
        )
        with pytest.raises(WrongHandlerMethodSignature) as exc:
            feature_validator_inst._validate_handler_signature(
                ("param1", "param3"), test_handler
            )
        assert str(expected_error) == str(exc.value)

        expected_error = WrongHandlerMethodSignature(
            handler_name=test_handler.__name__,
            condition=str(["data", "param1", "param3"]),
            handler_signature=str(["data", "param1", "param2"]),
        )
        with pytest.raises(WrongHandlerMethodSignature) as exc:
            feature_validator_inst._validate_handler_signature(
                {"param1": "value1", "param3": "value2"}, test_handler
            )
        assert str(expected_error) == str(exc.value)

    def test_prepare_condition(self):
        """Tests converting provided parameters to Tuple."""
        test_params1 = ("param1", "param2")
        result_param1 = ("param1", "param2")
        test_param2 = {"param1": "value1", "param2": "value2"}
        result_param2 = (("param1", "value1"), ("param2", "value2"))
        assert _prepare_condition(test_params1) == result_param1
        assert _prepare_condition(test_param2) == result_param2

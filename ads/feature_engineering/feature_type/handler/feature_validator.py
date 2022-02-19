#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to register custom validators for the feature types and
extending registered validators with dispatching based on the specific arguments.

Classes
-------
    FeatureValidator
        The Feature Validator class to manage custom validators.
    FeatureValidatorMethod
        The Feature Validator Method class. Extends methods which requires
        dispatching based on the specific arguments.
"""
import inspect
from typing import Any, Callable, Dict, Tuple, Union

import pandas as pd


class WrongHandlerMethodSignature(ValueError):
    def __init__(self, handler_name: str, condition: str, handler_signature: str):
        super().__init__(
            f"The registered condition {condition} is not compatible "
            f"with the provided {handler_name} method. "
            f"Expected parameters: {handler_signature}"
        )


class ValidatorNotFound(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Validator {name} not found.")


class ValidatorWithConditionNotFound(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Validator {name} with provided condition not found.")


class ValidatorAlreadyExists(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Validator {name} already exists.")


class ValidatorWithConditionAlreadyExists(ValueError):
    def __init__(self, name: str):
        super().__init__(f"Validator {name} with provided condition already exists.")


def _prepare_condition(params: Union[Tuple, Dict[str, Any]]) -> Tuple:
    """Converts provided parameters to Tuple.

    Parameters
    -----------
    params: (Union[Tuple, Dict[str, Any]])
        The condition which will be used to register a new validator.

    Returns
    -------
    Tuple
        Prepared condition.

    Raises
    ------
    ValueError
        If condition not provided or provided in the wrong format.
    """
    if not params:
        raise ValueError("Condition not provided.")
    if not isinstance(params, (dict, tuple)):
        raise ValueError(
            "Wrong format for the condition. Condition should be dict or list."
        )
    if not isinstance(params, tuple):
        return tuple((key, params[key]) for key in params)
    return params


class FeatureValidator:
    """The Feature Validator class to manage custom validators.

    Methods
    -------
    register(self, name: str, handler: Callable, condition: Union[Tuple, Dict[str, Any]] = None, replace: bool = False) -> None
        Registers new validator.
    unregister(self, name: str, condition: Union[Tuple, Dict[str, Any]] = None) -> None
        Unregisters validator.
    registered(self) -> pd.DataFrame
        Gets the list of registered validators.


    Examples
    --------
    >>> series = pd.Series(['+1-202-555-0141', '+1-202-555-0142'], name='Phone Number')

    >>> def phone_number_validator(data: pd.Series) -> pd.Series:
    ...    print("phone_number_validator")
    ...    return data

    >>> def universal_phone_number_validator(data: pd.Series, country_code) -> pd.Series:
    ...    print("universal_phone_number_validator")
    ...    return data

    >>> def us_phone_number_validator(data: pd.Series, country_code) -> pd.Series:
    ...    print("us_phone_number_validator")
    ...    return data

    >>> PhoneNumber.validator.register(name="is_phone_number", handler=phone_number_validator, replace=True)
    >>> PhoneNumber.validator.register(name="is_phone_number", handler=universal_phone_number_validator, condition = ('country_code',))
    >>> PhoneNumber.validator.register(name="is_phone_number", handler=us_phone_number_validator, condition = {'country_code':'+1'})

    >>> PhoneNumber.validator.is_phone_number(series)
        phone_number_validator
        0     +1-202-555-0141
        1     +1-202-555-0142

    >>> PhoneNumber.validator.is_phone_number(series, country_code = '+7')
        universal_phone_number_validator
        0     +1-202-555-0141
        1     +1-202-555-0142

    >>> PhoneNumber.validator.is_phone_number(series, country_code = '+1')
        us_phone_number_validator
        0     +1-202-555-0141
        1     +1-202-555-0142

    >>> PhoneNumber.validator.registered()
                   Validator                 Condition                            Handler
        ---------------------------------------------------------------------------------
        0    is_phone_number                        ()             phone_number_validator
        1    is_phone_number          ('country_code')   universal_phone_number_validator
        2    is_phone_number    {'country_code': '+1'}          us_phone_number_validator

    >>> series.ads.validator.is_phone_number()
        phone_number_validator
            0     +1-202-555-0141
            1     +1-202-555-0142

    >>> series.ads.validator.is_phone_number(country_code = '+7')
        universal_phone_number_validator
            0     +1-202-555-0141
            1     +1-202-555-0142

    >>> series.ads.validator.is_phone_number(country_code = '+1')
        us_phone_number_validator
        0     +1-202-555-0141
        1     +1-202-555-0142
    """

    def __init__(self):
        """Initializes the FeatureValidator."""
        self._validators = {}

    def register(
        self,
        name: str,
        handler: Callable,
        condition: Union[Tuple, Dict[str, Any]] = None,
        replace: bool = False,
    ) -> None:
        """Registers new validator.

        Parameters
        ----------
        name : str
            The validator name.
        handler: callable
            The handler.
        condition: Union[Tuple, Dict[str, Any]]
            The condition for the validator.
        replace: bool
            The flag indicating if the registered validator should be replaced with the new one.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            The name is empty or handler is not provided.
        TypeError
            The handler is not callable.
            The name of the validator is not a string.
        ValidatorAlreadyExists
            The validator is already registered.
        """
        if not name:
            raise ValueError("Validator name is not provided.")
        if not isinstance(name, str):
            raise TypeError("Validator name should be a string.")
        if not replace and name in self._validators:
            if not condition:
                raise ValidatorAlreadyExists(name)
            if self._validators[name]._has_condition(condition):
                raise ValidatorWithConditionAlreadyExists(name)
        if not handler:
            raise ValueError("Handler is not provided.")
        if not callable(handler):
            raise TypeError("Handler should be a function.")

        if condition:
            self._validators[name].register(condition=condition, handler=handler)
        else:
            self._validators[name] = FeatureValidatorMethod(handler)

    def unregister(
        self, name: str, condition: Union[Tuple, Dict[str, Any]] = None
    ) -> None:
        """Unregisters validator.

        Parameters
        ----------
        name: str
            The name of the validator to be unregistered.
        condition: Union[Tuple, Dict[str, Any]]
            The condition for the validator to be unregistered.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        TypeError
            The name of the validator is not a string.
        ValidatorNotFound
            The validator not found.
        ValidatorWIthConditionNotFound
            The validator with provided condition not found.
        """
        if not name:
            raise ValueError("Validator name is not provided.")
        if not isinstance(name, str):
            raise TypeError("Validator name should be a string.")
        if name not in self._validators:
            raise ValidatorNotFound(name)
        if condition and not self._validators[name]._has_condition(condition):
            raise ValidatorWithConditionNotFound(name)

        if condition:
            self._validators[name].unregister(condition)
        else:
            del self._validators[name]

    def registered(self) -> pd.DataFrame:
        """Gets the list of registered validators.

        Returns
        -------
        pd.DataFrame
            The list of registerd validators.
        """
        result_df = pd.DataFrame((), columns=["Validator", "Condition", "Handler"])
        for key, feature_validator in self._validators.items():
            feature_validators_df = feature_validator.registered()
            feature_validators_df.insert(0, "Validator", key)
            result_df = result_df.append(feature_validators_df)
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    def _bind_data(self, data: pd.Series) -> None:
        """Binds the data to the all registered validators.

        Parameters
        ----------
        data: pd.Series
            The data to be processed.
        """
        for validator in self._validators.values():
            validator._bind_data(data)

    def __getattr__(self, attr):
        """Makes it possible to invoke registered validators as a regular method."""
        if attr in self._validators:
            return self._validators[attr]
        raise AttributeError(attr)


class FeatureValidatorMethod:
    """The Feature Validator Method class.

    Extends methods which requires dispatching based on the specific arguments.

    Methods
    -------
    register(self, condition: Union[Tuple, Dict[str, Any]], handler: Callable) -> None
        Registers new handler.
    unregister(self, condition: Union[Tuple, Dict[str, Any]]) -> None
        Unregisters existing handler.
    registered(self) -> pd.DataFrame
        Gets the list of registered handlers.
    """

    def __init__(self, handler: Callable):
        """Initializes the Feature Validator Method.

        Parameters
        ----------
        handler: Callable
            The handler that will be called by default if suitable one not found.
        """
        if not handler:
            raise ValueError("Default handler is not specified.")

        self._default_handler = handler
        self._handlers = {}
        self._data = None

    def register(
        self, condition: Union[Tuple, Dict[str, Any]], handler: Callable
    ) -> None:
        """Registers new handler.

        Parameters
        -----------
        condition: Union[Tuple, Dict[str, Any]]
            The condition which will be used to register a new handler.
        handler: Callable
            The handler to be registered.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If condition not provided or provided in the wrong format.
            If handler not provided or has wrong format.
        """
        if not condition:
            raise ValueError("Condition not provided.")
        if not isinstance(condition, (dict, tuple)):
            raise ValueError(
                "Wrong format for the condition. Condition should be dict or list."
            )
        if not handler or not callable(handler):
            raise ValueError("Handler not provided. Handler should be a function.")

        prepared_condition = _prepare_condition(condition)
        # self.__validate_handler_signature(handler)
        self._handlers[prepared_condition] = handler
        self._data = None

    def unregister(self, condition: Union[Tuple, Dict[str, Any]]) -> None:
        """Unregisters existing handler.

        Parameters
        -----------
        condition: Union[Tuple, Dict[str, Any]]
            The condition which will be used to unregister a handler.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If condition not provided or provided in the wrong format.
            If condition not registered.
        """
        if not condition:
            raise ValueError("Condition not provided.")
        if not isinstance(condition, (dict, tuple)):
            raise ValueError(
                "Wrong format for the condition. Condition should be dict or list."
            )
        prepared_condition = _prepare_condition(condition)
        if prepared_condition not in self._handlers:
            raise ValueError("Condition not registered.")
        del self._handlers[prepared_condition]

    def registered(self) -> pd.DataFrame:
        """Gets the list of registered handlers.

        Returns
        -------
        pd.DataFrame
            The list of registerd handlers.
        """
        result = [("()", self._default_handler.__name__)]
        for key, value in self._handlers.items():
            try:
                str_key = str(dict(key))
            except ValueError:
                str_key = str(key)
            result.append((str_key, value.__name__))
        return pd.DataFrame(result, columns=["Condition", "Handler"])

    def _process(self, *args, **kwargs) -> pd.Series:
        """Finds and invokes a suitable handler for the provided condition.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments. Parameters to search suitable handler.

        Returns
        -------
        pd.Series
            The result of invoked handler.
        """
        if kwargs:
            for key in (
                tuple((key, kwargs[key]) for key in kwargs),
                tuple(kwargs.keys()),
            ):
                if key in self._handlers:
                    return self._handlers[key](self._data, *args, **kwargs)

        return self._default_handler(self._data, *args, **kwargs)

    def _bind_data(self, data: pd.Series) -> None:
        """Binds the data to the validator.

        Parameters
        ----------
        data: pd.Series
            The data to be processed.
        """
        self._data = data

    def _validate_handler_signature(
        self, condition: Union[Tuple, Dict[str, Any]], handler: Callable
    ) -> bool:
        """Validates handler signature.

        Parameters
        ----------
        condition: Union[Tuple, Dict[str, Any]]
            The condition to validate.
        handler: Callabe
            The hanlder to validate.

        Returns
        -------
        bool
            True if provided condition and handler arguments compatible.

        Raises
        -------
        WrongHandlerMethodSignature
            If provided condition and handler arguments not compatible.
        """
        prepared_condition = _prepare_condition(condition)
        handler_args = list(inspect.signature(handler).parameters.keys())
        params_args = ["data"] + (
            list(arg[0] for arg in prepared_condition)
            if isinstance(prepared_condition[0], tuple)
            else list(prepared_condition)
        )
        if handler_args != params_args:
            raise WrongHandlerMethodSignature(
                handler.__name__, str(params_args), str(handler_args)
            )
        return True

    def _has_condition(self, condition: Union[Tuple, Dict[str, Any]]) -> bool:
        """Checks whether provided condition registered or not.

        Parameters
        ----------
        condition: Union[Tuple, Dict[str, Any]]
                The condition to check.
        Returns
        -------
        bool
            True if condition registered, False othervise.

        Raises
        ------
        ValueError
            If condition not provided or has wrong format.
        """
        if not condition:
            raise ValueError("Condition not provided.")
        if not isinstance(condition, (dict, tuple)):
            raise ValueError(
                "Wrong format for the condition. Condition should be dict or list."
            )
        prepared_condition = _prepare_condition(condition)
        return prepared_condition in self._handlers

    def __call__(self, *args, **kwargs) -> pd.Series:
        """Makes class instance callable.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        pd.Series
            The result of processing data.

        Raises
        ------
        ValueError
            If data is not provided,
        TypeError
            If data has wrong format.
        """
        if args and len(args) > 0:
            self._data = args[0]
        if self._data is None:
            raise ValueError("Data is not provided.")
        if not isinstance(self._data, pd.Series):
            raise TypeError("Wrong data format. Data should be Series.")
        return self._process(**kwargs)

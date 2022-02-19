#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that helps to register custom warnings for the feature types.

Classes
-------
    FeatureWarning
        The Feature Warning class. Provides functionality to register
        warning handlers and invoke them.

Examples
--------
    >>> warning = FeatureWarning()
    >>> def warning_handler_zeros_count(data):
    ...    return pd.DataFrame(
    ...        [['Zeros', 'Age has 38 zeros', 'Count', 38]],
    ...        columns=['Warning', 'Message', 'Metric', 'Value'])
    >>> def warning_handler_zeros_percentage(data):
    ...    return pd.DataFrame(
    ...        [['Zeros', 'Age has 12.2% zeros', 'Percentage', '12.2%']],
    ...        columns=['Warning', 'Message', 'Metric', 'Value'])
    >>> warning.register(name="zeros_count", handler=warning_handler_zeros_count)
    >>> warning.register(name="zeros_percentage", handler=warning_handler_percentage)
    >>> warning.registered()
                        Name                               Handler
        ----------------------------------------------------------
        0         zeros_count          warning_handler_zeros_count
        1    zeros_percentage     warning_handler_zeros_percentage

    >>> warning.zeros_percentage(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros      Age has 38 zeros          Count         38

    >>> warning.zeros_count(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        1          Zeros   Age has 12.2% zeros     Percentage      12.2%

    >>> warning(data_series)
            Warning                    Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros      Age has 38 zeros          Count         38
        1          Zeros   Age has 12.2% zeros     Percentage      12.2%

    >>> warning.unregister('zeros_count')
    >>> warning(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros   Age has 12.2% zeros     Percentage      12.2%
"""
from typing import Callable
import pandas as pd
from ads.feature_engineering.exceptions import WarningNotFound, WarningAlreadyExists


def _validate_warning_handler(handler: Callable) -> bool:
    """Validates warning handler.

    Handler should get pd.Series as a parameter and return pd.DataFrame as result.
    Dataframe should have four columns: Warning, Message, Metric and Value.

    Parameters
    ----------
    handler: Callable
        The handler to validate.

    Returns
    -------
    bool
        True if handler compatible with Feature Warning, False otherwise.
    """
    result = True
    try:
        handler_result = handler(pd.Series([]))
        assert isinstance(handler_result, pd.DataFrame)
        assert list(handler_result.columns) == ["Warning", "Message", "Metric", "Value"]
    except AssertionError:
        result = False
    return result


class FeatureWarning:
    """The Feature Warning class.

    Provides functionality to register warning handlers and invoke them.

    Methods
    -------
    register(self, name: str, handler: Callable) -> None
        Registers a new warning for the feature type.
    unregister(self, name: str) -> None
        Unregisters warning.
    registered(self) -> pd.DataFrame
        Gets the list of registered warnings.

    Examples
    --------
    >>> warning = FeatureWarning()
    >>> def warning_handler_zeros_count(data):
    ...    return pd.DataFrame(
    ...        [['Zeros', 'Age has 38 zeros', 'Count', 38]],
    ...        columns=['Warning', 'Message', 'Metric', 'Value'])
    >>> def warning_handler_zeros_percentage(data):
    ...    return pd.DataFrame(
    ...        [['Zeros', 'Age has 12.2% zeros', 'Percentage', '12.2%']],
    ...        columns=['Warning', 'Message', 'Metric', 'Value'])
    >>> warning.register(name="zeros_count", handler=warning_handler_zeros_count)
    >>> warning.register(name="zeros_percentage", handler=warning_handler_percentage)
    >>> warning.registered()
                      Warning                              Handler
        ----------------------------------------------------------
        0         zeros_count          warning_handler_zeros_count
        1    zeros_percentage     warning_handler_zeros_percentage

    >>> warning.zeros_percentage(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros      Age has 38 zeros          Count         38

    >>> warning.zeros_count(data_series)
                  Warning              Message         Metric      Value
        ----------------------------------------------------------------
        1          Zeros   Age has 12.2% zeros     Percentage      12.2%

    >>> warning(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros      Age has 38 zeros          Count         38
        1          Zeros   Age has 12.2% zeros     Percentage      12.2%

    >>> warning.unregister('zeros_count')
    >>> warning(data_series)
                 Warning               Message         Metric      Value
        ----------------------------------------------------------------
        0          Zeros   Age has 12.2% zeros     Percentage      12.2%
    """

    def __init__(self):
        """Initializes the FeatureWarning."""
        self._data = None
        self._handlers = {}

    def register(self, name: str, handler: Callable, replace: bool = False) -> None:
        """Registers a new warning.

        Parameters
        ----------
        name : str
            The warning name.
        handler: callable
            The handler associated with the warning.
        replace: bool
            The flag indicating if the registered warning should be replaced with the new one.

        Returns
        -------
        None
            Nothing

        Raises
        ------
        ValueError
            If warning name is empty or handler not defined.
        TypeError
            If handler is not callable.
        WarningAlreadyExists
            If warning is already registered.
        """
        if not name:
            raise ValueError("Warning name is not provided.")
        if name in self._handlers and not replace:
            raise WarningAlreadyExists(name)
        if not handler:
            raise ValueError("Handler is not provided.")
        if not callable(handler):
            raise TypeError("Handler should be a function.")
        self._handlers[name] = handler

    def unregister(self, name: str) -> None:
        """Unregisters warning.

        Parameters
        -----------
        name: str
            The name of warning to be unregistered.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If warning name is not provided or empty.
        WarningNotFound
            If warning not found.
        """
        if not name:
            raise ValueError("Warning name is not provided.")
        if name not in self._handlers:
            raise WarningNotFound(name)
        del self._handlers[name]

    def registered(self) -> pd.DataFrame:
        """Gets the list of registered warnings.

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>>    The list of registerd warnings in DataFrame format.
                             Name                               Handler
            -----------------------------------------------------------
            0         zeros_count           warning_handler_zeros_count
            1    zeros_percentage      warning_handler_zeros_percentage
        """
        result = []
        for name, handler in self._handlers.items():
            result.append((name, handler.__name__))
        return pd.DataFrame(result, columns=["Warning", "Handler"])

    def _bind_data(self, data: pd.Series) -> None:
        """Binds data to the feature warning.

        Parameters
        ----------
        data: pd.Series
            The data to be bound.
        """
        self._data = data

    def _process(self) -> pd.DataFrame:
        """Invokes the all registered warnings.

        Returns
        -------
        pd.DataFrame
        >>>    The result of invoked warning handlers.
                 Warning               Message       Metric    Value
            --------------------------------------------------------
                   Zeros      Age has 38 zeros        Count       38
                   Zeros   Age has 12.2% zeros   Percentage    12.2%
        Raises
        ------
        ValueError
            If data is not provided or result of warning has a wrong format.
        """
        if self._data is None:
            raise ValueError("Data is not provided.")

        if not self._handlers:
            return None

        expected_columns = ["Warning", "Message", "Metric", "Value"]
        result_df = pd.DataFrame([], columns=expected_columns)
        for name, handler in self._handlers.items():
            try:
                handler_result = handler(self._data)
            except Exception as ex:
                raise ValueError(
                    f"An error occurred while executing the '{name}'. "
                    f"Details: {str(ex)}."
                ) from ex
            if handler_result is not None:
                if not isinstance(handler_result, pd.DataFrame) or (
                    not handler_result.empty
                    and set(list(handler_result.columns)) != set(expected_columns)
                ):
                    raise ValueError(
                        f"An error occurred while executing the '{name}'. "
                        f"Details: '{name}' should return a DataFrame "
                        f"with columns: {expected_columns}."
                    )
                result_df = result_df.append(handler_result)
        result_df.reset_index(drop=True, inplace=True)
        return result_df

    def __call__(self, *args) -> pd.DataFrame:
        """Makes class instance callable.

        Parameters
        ----------
        *args
            Variable length argument list.

        Returns
        -------
        pd.DataFrame
            The result of processing the all registered warnings.

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
        return self._process()

    def __getattr__(self, attr):
        """Makes it possible to invoke registered warning as a regular method."""
        if attr in self._handlers:
            return self._handlers[attr]
        raise AttributeError(attr)

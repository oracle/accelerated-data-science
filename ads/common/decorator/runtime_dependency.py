#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that provides the decorator helping to add runtime dependencies in functions.

Examples
--------
>>> @runtime_dependency(module="pandas", short_name="pd")
... def test_function()
...     print(pd)

>>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df")
... def test_function()
...     print(df)

>>> @runtime_dependency(module="pandas", short_name="pd")
... @runtime_dependency(module="pandas", object="DataFrame", short_name="df")
... def test_function()
...     print(df)
...     print(pd)

>>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df", install_from="ads[optional]")
... def test_function()
...     pass

>>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df", err_msg="Custom error message.")
... def test_function()
...     pass
"""

import importlib
import logging
from functools import wraps
from typing import Any, Callable

from .utils import _get_original_func

logger = logging.getLogger(__name__)

MODULE_NOT_FOUND_ERROR = (
    "The `{module}` module was not found. Please run `pip install {package}`."
)
IMPORT_ERROR = "Cannot import name `{object}` from `{module}`."


class OptionalDependency:
    LABS = "oracle-ads[labs]"
    BOOSTED = "oracle-ads[boosted]"
    NOTEBOOK = "oracle-ads[notebook]"
    TEXT = "oracle-ads[text]"
    VIZ = "oracle-ads[viz]"
    DATA = "oracle-ads[data]"
    OPCTL = "oracle-ads[opctl]"
    MYSQL = "oracle-ads[mysql]"
    BDS = "oracle-ads[bds]"
    PYTORCH = "oracle-ads[torch]"
    TENSORFLOW = "oracle-ads[tensorflow]"
    GEO = "oracle-ads[geo]"
    ONNX = "oracle-ads[onnx]"
    OPTUNA = "oracle-ads[optuna]"
    SPARK = "oracle-ads[spark]"
    HUGGINGFACE = "oracle-ads[huggingface]"


def runtime_dependency(
    module: str,
    short_name: str = "",
    object: str = None,
    install_from: str = None,
    err_msg: str = "",
    is_for_notebook_only=False,
):
    """The decorator which is helping to add runtime dependencies to functions.

    Parameters
    ----------
    module: str
        The module name to be imported.
    short_name: (str, optional). Defaults to empty string.
        The short name for the imported module.
    object: (str, optional). Defaults to None.
        The name of the object to be imported. Can be a function or a class, or
        any variable provided by module.
    install_from: (str, optional). Defaults to None.
        The parameter helping to answer from where the required dependency can be installed.
    err_msg: (str, optional). Defaults to empty string.
        The custom error message.
    is_for_notebook_only: (bool, optional). Defaults to False.
        If the value of this flag is set to True, the dependency will be added only
        in case when the current environment is a jupyter notebook.

    Raises
    ------
    ModuleNotFoundError
        In case if requested module not found.
    ImportError
        In case if object cannot be imported from the module.

    Examples
    --------
    >>> @runtime_dependency(module="pandas", short_name="pd")
    ... def test_function()
    ...     print(pd)

    >>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df")
    ... def test_function()
    ...     print(df)

    >>> @runtime_dependency(module="pandas", short_name="pd")
    ... @runtime_dependency(module="pandas", object="DataFrame", short_name="df")
    ... def test_function()
    ...     print(df)
    ...     print(pd)

    >>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df", install_from="ads[optional]")
    ... def test_function()
    ...     pass

    >>> @runtime_dependency(module="pandas", object="DataFrame", short_name="df", err_msg="Custom error message.")
    ... def test_function()
    ...     pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            from ads.common.utils import is_notebook

            if not is_for_notebook_only or is_notebook():
                assert module, "The parameter `module` must be provided."
                assert isinstance(
                    module, str
                ), "The parameter `module` must be a string."
                try:
                    _package_name = module.split(".")[0]
                    # importing module
                    _module = importlib.import_module(module)
                    # checking if object parameter is specified then inporting
                    # the object from the module. The object can be a function
                    # or a class or any module vatiable.
                    _module = getattr(_module, object) if object else _module
                    # retrieving the original function from the decorated one.
                    # This is necessary when the chain of decorators is used.
                    # The only original function should be injected with the
                    # dependency.
                    original_func = _get_original_func(func)
                    _module_name = module.split(".")[-1]
                    # Injecting the imported module into global variables of the
                    # original function.
                    original_func.__globals__[
                        short_name or object or _module_name
                    ] = _module
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        err_msg
                        or MODULE_NOT_FOUND_ERROR.format(
                            module=module, package=install_from or _package_name
                        )
                    )
                except AttributeError:
                    raise ImportError(
                        err_msg or IMPORT_ERROR.format(object=object, module=module)
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator

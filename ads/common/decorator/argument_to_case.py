#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
The module that provides the decorator helping to convert function arguments
to specific case (lower/upper).

Examples
--------
>>> @argument_to_case("lower", ["name"])
... def test_function(name: str, last_name: str)
...     print(name)
...
>>> test_function("myname", "mylastname")
... MYNAME
"""

import inspect
from copy import copy
from enum import Enum
from functools import wraps
from typing import Callable, List

from .utils import _get_original_func


class ArgumentCase(Enum):
    LOWER = "lower"
    UPPER = "upper"


def argument_to_case(case: ArgumentCase, arguments: List[str]):
    """The decorator helping to convert function arguments to specific case.

    Parameters
    ----------
    case: ArgumentCase
        The case to convert specific arguments.
    arguments: List[str]
        The list of arguments to convert to specific case.

    Examples
    --------
    >>> @argument_to_case("lower", ["name"])
    ... def test_function(name: str, last_name: str)
    ...     print(name)
    ...
    >>> test_function("myname", "mylastname")
    ... MYNAME
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieving the original function from the decorated one.
            # This is necessary when the chain of decorators is used.
            # The only original function arguments should be processed.
            original_func = _get_original_func(func)
            # Getting the original function arguments
            func_args = inspect.getfullargspec(original_func).args
            # Saving original args and kwargs.
            new_args = list(copy(args))
            new_kwargs = copy(kwargs)
            # Converting args and kwargs to the specific case.
            for func_arg in func_args:
                if func_arg in arguments:
                    if func_arg in new_kwargs:
                        new_kwargs[func_arg] = (
                            new_kwargs[func_arg].lower()
                            if case == ArgumentCase.LOWER
                            else new_kwargs[func_arg].upper()
                        )
                    else:
                        arg_index = func_args.index(func_arg)
                        if arg_index >= 0:
                            new_args[arg_index] = (
                                new_args[arg_index].lower()
                                if case == ArgumentCase.LOWER
                                else new_args[arg_index].upper()
                            )
            return func(*new_args, **new_kwargs)

        return wrapper

    return decorator

#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import inspect
from functools import wraps
from typing import Any, Callable, List, Union

from ads.common.decorator.utils import _get_original_func


def require_nonempty_arg(
    arg_name: Union[str, List[str]], error_msg: str = "A required argument is empty"
) -> Callable:
    """
    A decorator to ensure that a specific argument of a function is not empty.

    Parameters
    ----------
    arg_name (Union[str, List[str]]): The name of the argument or the list of the arguments to check.
    error_msg (str, optional)
        The error message to raise if the check fails.

    Returns
    -------
    Callable
        A wrapped function that includes the check.

    Raises
    ------
    ValueError
        If the specified argument is empty.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Retrieving the original function from the decorated one.
            # This is necessary when the chain of decorators is used.
            # The only original function arguments should be processed.
            original_func = _get_original_func(func)

            # Get the signature of the function
            sig = inspect.signature(original_func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check if the argument is present and not empty
            if isinstance(arg_name, str):
                arguments_to_check = [arg_name]
            else:
                arguments_to_check = arg_name

            if not any(
                check_name in bound_args.arguments and bound_args.arguments[check_name]
                for check_name in arguments_to_check
            ):
                raise ValueError(error_msg)

            return func(*args, **kwargs)

        return wrapper

    return decorator

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings
from enum import Enum
from functools import wraps


class TARGET_TYPE(Enum):
    CLASS = "Class"
    METHOD = "Method"
    ATTRIBUTE = "Attribute"


def deprecated(
    deprecated_in: str,
    removed_in: str = None,
    details: str = None,
    target_type: str = None,
):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    deprecated_in: `str`
        Version of ADS where this function deprecated.
    removed_in: `str`
        Future version where this function will be removed.
    details: `str`
        More information to be shown.
    """

    def decorator(target):
        @wraps(target)
        def wrapper(*args, **kwargs):
            if target.__name__ == "__init__":
                _target_type = target_type or TARGET_TYPE.CLASS.value
                target_name = target.__qualname__.split(".")[0]
            else:
                _target_type = target_type or TARGET_TYPE.METHOD.value
                target_name = target.__name__

            msg = (
                f"{_target_type} {target_name} is "
                f"deprecated in {deprecated_in} and will be "
                f"removed in {removed_in if removed_in else 'a future release'}."
                f"{'' if not details else ' ' + details}"
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return target(*args, **kwargs)

        return wrapper

    return decorator

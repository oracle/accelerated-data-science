#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import Callable


def _get_original_func(func: callable) -> Callable:
    """The helper to retrieve the original function from the decorated one."""
    if func and hasattr(func, "__wrapped__"):
        return _get_original_func(func.__wrapped__)
    return func

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from typing import Any, Callable

import cloudpickle

logger = logging.getLogger(__name__)


def serialize_function_to_hex(func: Callable[..., Any]) -> str:
    """
    Serialize a function to a hexadecimal string.

    Parameters
    ----------
    func (Callable[..., Any]): The function to serialize.

    Returns
    -------
    str: The serialized function as a hexadecimal string.

    Raises
    ------
    ValueError: If serialization fails.
    """
    try:
        return cloudpickle.dumps(func).hex()
    except Exception as e:
        raise ValueError(f"Failed to serialize function: {e}")

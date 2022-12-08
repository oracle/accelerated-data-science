#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import copy

ALIAS_MAP = {
    "file_uri": "script_uri",
}


def normalize_config(config: dict) -> dict:
    """
    Normalize property names in a configuration so that they work directly with ADS.

    Parameters
    ----------
    config: dict
        input configuration, usually coming directly from an OCI response

    Returns
    -------
    dict
        output configuration
    """
    normalized = {}
    for k, v in config.items():
        k = ALIAS_MAP[k] if k in ALIAS_MAP else k
        if isinstance(v, dict):
            normalized[k] = normalize_config(v)
        else:
            normalized[k] = v
    return normalized


def get_value(obj, attr, default=None):
    """Gets a copy of the value from a nested dictionary of an object with nested attributes.

    Parameters
    ----------
    obj :
        An object or a dictionary
    attr :
        Attributes as a string seprated by dot(.)
    default :
        Default value to be returned if attribute is not found.

    Returns
    -------
    Any:
        A copy of the attribute value. For dict or list, a deepcopy will be returned.

    """
    keys = attr.split(".")
    val = default
    for key in keys:
        if hasattr(obj, key):
            val = getattr(obj, key)
        elif hasattr(obj, "get"):
            val = obj.get(key, default)
        else:
            return default
        obj = val
    return copy.deepcopy(val)

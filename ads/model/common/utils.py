#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2022 Oracle and its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, Optional


def _extract_locals(
    locals: Dict[str, Any], filter_out_nulls: Optional[bool] = True
) -> Dict[str, Any]:
    """Extract arguments from local variables.
    If input dictionary contains `kwargs`, then it will not be included to
    the result dictionary.

    Properties
    ----------
    locals: Dict[str, Any]
        A dictionary, the result of `locals()` method.
        However can be any dictionary.
    filter_out_nulls: (bool, optional). Defaults to `True`.
        Whether `None` values should be filtered out from the result dict or not.

    Returns
    -------
    Dict[str, Any]
        A new dictionary with the result values.
    """
    result = {}
    keys_to_filter_out = ("kwargs",)
    consolidated_dict = {**locals.get("kwargs", {}), **locals}
    for key, value in consolidated_dict.items():
        if key not in keys_to_filter_out and not (filter_out_nulls and value is None):
            result[key] = value
    return result

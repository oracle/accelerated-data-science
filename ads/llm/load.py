#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict, List, Optional
from langchain.load.load import load as lc_load


def load(
    obj: Any,
    *,
    secrets_map: Optional[Dict[str, str]] = None,
    valid_namespaces: Optional[List[str]] = None,
) -> Any:
    """Revive a LangChain class from a JSON object. Use this if you already
    have a parsed JSON object, eg. from `json.load` or `json.loads`.

    This is a drop in replacement for load() in langchain.load.load to support loading ADS LangChain compatible component.

    Args:
        obj: The object to load.
        secrets_map: A map of secrets to load.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.

    Returns:
        Revived LangChain objects.
    """
    if not valid_namespaces:
        valid_namespaces = []
    if "ads" not in valid_namespaces:
        valid_namespaces.append("ads")
    return lc_load(obj, secrets_map=secrets_map, valid_namespaces=valid_namespaces)

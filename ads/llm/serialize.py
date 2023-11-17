#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from typing import Any, Dict, List, Optional

import fsspec
import yaml
from langchain import llms
from langchain.llms import loading
from langchain.chains.loading import load_chain_from_config
from langchain.load.load import load as __lc_load
from langchain.load.serializable import Serializable

from ads.llm.langchain.plugins.llm_gen_ai import GenerativeAI
from ads.common.auth import default_signer

# This is a temp solution for supporting custom LLM in legacy load_chain
__lc_llm_dict = llms.get_type_to_cls_dict()
__lc_llm_dict["GenerativeAI"] = lambda: GenerativeAI


def __new_type_to_cls_dict():
    return __lc_llm_dict


llms.get_type_to_cls_dict = __new_type_to_cls_dict
loading.get_type_to_cls_dict = __new_type_to_cls_dict


def load(
    obj: Any,
    *,
    secrets_map: Optional[Dict[str, str]] = None,
    valid_namespaces: Optional[List[str]] = None,
    **kwargs,
) -> Any:
    """Revive an ADS/LangChain class from a JSON object. Use this if you already
    have a parsed JSON object, eg. from `json.load` or `json.loads`.

    This is a drop in replacement for load() in langchain.load.load to support loading compatible class from ADS.

    Args:
        obj: The object to load.
        secrets_map: A map of secrets to load.
        valid_namespaces: A list of additional namespaces (modules)
            to allow to be deserialized.

    Returns:
        Revived LangChain objects.
    """
    if isinstance(obj, dict) and "_type" in obj:
        # Legacy chain
        return load_chain_from_config(obj, **kwargs)

    if not valid_namespaces:
        valid_namespaces = []
    if "ads" not in valid_namespaces:
        valid_namespaces.append("ads")
    return __lc_load(obj, secrets_map=secrets_map, valid_namespaces=valid_namespaces)


def load_from_yaml(
    uri: str,
    *,
    secrets_map: Optional[Dict[str, str]] = None,
    valid_namespaces: Optional[List[str]] = None,
    **kwargs,
):
    class __SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node):
            return node.value[0].value

    __SafeLoaderIgnoreUnknown.add_constructor(
        None, __SafeLoaderIgnoreUnknown.ignore_unknown
    )

    if uri.startswith("oci://"):
        storage_options = default_signer()
    else:
        storage_options = {}
    with fsspec.open(uri, **storage_options) as f:
        config = yaml.load(f, Loader=__SafeLoaderIgnoreUnknown)
    return load(
        config, secrets_map=secrets_map, valid_namespaces=valid_namespaces, **kwargs
    )


def default(obj: Any) -> Any:
    """Calls the to_json() method to serialize the object.

    Parameters
    ----------
    obj : Any
        The object to be serialized.

    Returns
    -------
    Any
        The serialized representation of the object.

    Raises
    ------
    TypeError
        If the object is not LangChain serializable.
    """
    if isinstance(obj, Serializable) and obj.is_lc_serializable():
        return obj.to_json()
    raise TypeError(f"Serialization of {type(obj)} is not supported.")


def dump(obj: Any) -> Dict[str, Any]:
    """Return a json dict representation of an object.

    This is a drop in replacement of the dumpd() in langchain.load.dump,
    except that this method will raise TypeError when the object is not serializable.
    """
    return json.loads(json.dumps(obj, default=default))

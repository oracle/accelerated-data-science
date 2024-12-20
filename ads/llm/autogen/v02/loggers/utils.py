#!/usr/bin/env python
# Copyright (c) 2024, Oracle and/or its affiliates.  All rights reserved.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import inspect
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union


def is_json_serializable(obj: Any) -> bool:
    """Checks if an object is JSON serializable.

    Parameters
    ----------
    obj : Any
        Any object.

    Returns
    -------
    bool
        True if the object is JSON serializable, otherwise False.
    """
    try:
        json.dumps(obj)
    except Exception:
        return False
    return True


def serialize_response(response) -> dict:
    """Serializes the LLM response to dictionary."""
    if isinstance(response, SimpleNamespace) or is_json_serializable(response):
        # Convert simpleNamespace to dict
        return json.loads(json.dumps(response, default=vars))
    elif hasattr(response, "dict") and callable(response.dict):
        return json.loads(json.dumps(response.dict(), default=str))
    elif hasattr(response, "model") and hasattr(response, "choices"):
        return {
            "model": response.model,
            "choices": [
                {"message": {"content": choice.message.content}}
                for choice in response.choices
            ],
            "response": str(response),
        }
    return {
        "model": "",
        "choices": [{"message": {"content": response}}],
        "response": str(response),
    }


def serialize(
    obj: Union[int, float, str, bool, Dict[Any, Any], List[Any], Tuple[Any, ...], Any],
    exclude: Tuple[str, ...] = ("api_key", "__class__"),
    no_recursive: Tuple[Any, ...] = (),
) -> Any:
    """Serializes an object for logging purpose."""
    try:
        if isinstance(obj, (int, float, str, bool)):
            return obj
        elif callable(obj):
            return inspect.getsource(obj).strip()
        elif isinstance(obj, dict):
            return {
                str(k): (
                    serialize(str(v))
                    if isinstance(v, no_recursive)
                    else serialize(v, exclude, no_recursive)
                )
                for k, v in obj.items()
                if k not in exclude
            }
        elif isinstance(obj, (list, tuple)):
            return [
                (
                    serialize(str(v))
                    if isinstance(v, no_recursive)
                    else serialize(v, exclude, no_recursive)
                )
                for v in obj
            ]
        else:
            return str(obj)
    except Exception:
        return str(obj)

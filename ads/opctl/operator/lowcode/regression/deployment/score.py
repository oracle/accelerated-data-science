#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def load_model():
    """Loads the packaged regression artifact bundle from `models.pickle`."""
    import cloudpickle

    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_dir, "models.pickle")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"models.pickle is not found in {model_dir}")

    with open(model_path, "rb") as f:
        return cloudpickle.load(f)


def _resolve_model_and_spec(model_or_bundle):
    if isinstance(model_or_bundle, dict) and "models" in model_or_bundle:
        return model_or_bundle["models"], model_or_bundle.get("spec", {}) or {}
    return model_or_bundle, {}


def _is_list_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray, pd.Series))


def _to_frame(raw_data) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()

    if isinstance(raw_data, dict):
        if any(_is_list_like(value) for value in raw_data.values()):
            return pd.DataFrame(raw_data)
        return pd.DataFrame([raw_data])

    return pd.DataFrame(raw_data)


def _build_input_frame(data, target_column=None) -> pd.DataFrame:
    if "data" not in data:
        raise ValueError("Input payload must include `data`.")

    raw_data = data["data"]
    x = _to_frame(raw_data)

    if target_column and target_column in x.columns:
        x = x.drop(columns=[target_column])

    return x


def predict(data, model=None) -> dict:
    """Runs prediction on raw regression input rows.

    Expected payload:
    {
      "data": dataset_df
    }
    """
    model_obj, spec = _resolve_model_and_spec(model or load_model())
    target_column = spec.get("target_column")

    x = _build_input_frame(data, target_column=target_column)
    print(x)
    yhat = model_obj.predict(x)

    if isinstance(yhat, pd.Series):
        yhat = yhat.tolist()
    elif isinstance(yhat, np.ndarray):
        yhat = yhat.tolist()

    return {"prediction": json.dumps(yhat)}

#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def load_model():
    """Loads serialized model pipeline from `model.pkl`."""
    import cloudpickle

    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl is not found in {model_dir}")

    with open(model_path, "rb") as f:
        return cloudpickle.load(f)


def predict(data, model=None) -> dict:
    """Returns prediction given the model and JSON payload.

    Expected payload:
    {
      "data": [
        {"feature_a": 1.0, "feature_b": "x"},
        ...
      ]
    }
    """
    model = model or load_model()
    if "data" not in data:
        raise ValueError("Input payload must include `data`.")

    x = pd.DataFrame(data["data"])
    yhat = model.predict(x)

    if isinstance(yhat, np.ndarray):
        yhat = yhat.tolist()

    return {"prediction": json.dumps(yhat)}

#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

# THIS IS A CUSTOM SCORE.PY

model_name = "model.pkl"


def load_model(model_file_name=model_name):
    return model_file_name


def predict(data, model=load_model()):
    return {"prediction": "This is a custom score.py."}

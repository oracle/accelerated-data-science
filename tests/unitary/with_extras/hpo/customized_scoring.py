#!/usr/bin/env python

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from sklearn.metrics import make_scorer

import pytest

pytest.skip(allow_module_level=True)


def customerize_score(y_true, y_pred, sample_weight=None):
    score = y_true == y_pred
    return np.average(score, weights=sample_weight)


scoring = make_scorer(customerize_score)

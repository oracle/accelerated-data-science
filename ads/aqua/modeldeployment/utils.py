#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA model deployment utils"""

import itertools


def get_combinations(input_dict: dict):
    """Finds all unique combinations within input dict.

    The input is a dict of {model:[gpu_count]} on a specific shape and this method will
    return a list of all unique combinations of gpu allocation of each model.

    For example:

    input: {'model_a': [2, 4], 'model_b': [1, 2, 4], 'model_c': [1, 2, 8]}
    output:
    [
        {'model_a': 2, 'model_b': 1, 'model_c': 1},
        {'model_a': 2, 'model_b': 1, 'model_c': 2},
        {'model_a': 2, 'model_b': 1, 'model_c': 8},
        {'model_a': 2, 'model_b': 2, 'model_c': 1},
        {'model_a': 2, 'model_b': 2, 'model_c': 2},
        {'model_a': 2, 'model_b': 2, 'model_c': 8},
        {'model_a': 2, 'model_b': 4, 'model_c': 1},
        {'model_a': 2, 'model_b': 4, 'model_c': 2},
        {'model_a': 2, 'model_b': 4, 'model_c': 8},
        {'model_a': 4, 'model_b': 1, 'model_c': 1},
        {'model_a': 4, 'model_b': 1, 'model_c': 2},
        {'model_a': 4, 'model_b': 1, 'model_c': 8},
        {'model_a': 4, 'model_b': 2, 'model_c': 1},
        {'model_a': 4, 'model_b': 2, 'model_c': 2},
        {'model_a': 4, 'model_b': 2, 'model_c': 8},
        {'model_a': 4, 'model_b': 4, 'model_c': 1},
        {'model_a': 4, 'model_b': 4, 'model_c': 2},
        {'model_a': 4, 'model_b': 4, 'model_c': 8}
    ]

    Parameters
    ----------
    input_dict: dict
        A dict of {model:[gpu_count]} on a specific shape

    Returns
    -------
    list:
        A list of all unique combinations of gpu allocation of each model.
    """
    keys, values = zip(*input_dict.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

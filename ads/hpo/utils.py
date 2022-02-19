#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from numbers import Integral, Number

import numpy as np
import oci
import scipy as sp
from ads.common import utils
from oci.object_storage import ObjectStorageClient
from optuna import logging  # NOQA
from sklearn.utils import _safe_indexing as sklearn_safe_indexing

logger = logging.get_logger(__name__)


def _update_space_name(search_space, **kwargs):
    """
    name:
        step name
    search_space:
        search space
    """
    step_name = kwargs.pop("step_name", None)
    if step_name:
        param_distributions = {}
        for param_name, distributions in search_space.items():
            param_name = "__".join([step_name, param_name])
            param_distributions[param_name] = distributions
        return param_distributions
    else:
        return search_space


def _initialize_ociclient():
    """
    Creates and initializes oci object storage client.
    """
    try:
        rps = oci.auth.signers.get_resource_principals_signer()
        oci_client = ObjectStorageClient(
            {}, signer=rps, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY
        )
    except:
        if os.path.exists(utils.oci_config_file()):
            config = utils.oci_config_file()
            profile = utils.oci_config_profile()
            config = oci.config.from_file(config, profile)
            oci_client = ObjectStorageClient(
                config, retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY
            )
        else:
            logger.error(
                f"OCI key configuration was not found at {utils.oci_config_file()}. Either Resource principal or OCI keys are required to access the conda environments"
            )
    return oci_client


def _extract_uri(file_uri):
    bucketname, rest = file_uri.split("oci://")[1].split("@")
    namespace = rest.split("/")[0]
    filename = rest.replace(namespace + "/", "")
    return bucketname, namespace, filename


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L131-L135
def _is_arraylike(x):
    # type: (Any) -> bool

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


# NOTE Original implementation:
# https://github.com/scikit-learn/scikit-learn/blob/ \
# 8caa93889f85254fc3ca84caa0a24a1640eebdd1/sklearn/utils/validation.py#L217-L234
def _make_indexable(iterable):  # pragma: no cover
    # type: (IterableType) -> (IndexableType)

    tocsr_func = getattr(iterable, "tocsr", None)
    if tocsr_func is not None and sp.sparse.issparse(iterable):
        return tocsr_func(iterable)
    elif (
        hasattr(iterable, "__getitem__")
        or hasattr(iterable, "iloc")
        or iterable is None
    ):
        return iterable
    return np.array(iterable)


def _num_samples(x):
    # type: (ArrayLikeType) -> int

    x_shape = getattr(x, "shape", None)
    if x_shape is not None:
        if isinstance(x_shape[0], Integral):
            return int(x_shape[0])

    try:
        return len(x)
    except TypeError:
        raise TypeError("Expected sequence or array-like, got %s." % type(x))


def _safe_indexing(
    X,  # type: Union[OneDimArrayLikeType, TwoDimArrayLikeType]
    indices,  # type: OneDimArrayLikeType
):
    # type: (...) -> Union[OneDimArrayLikeType, TwoDimArrayLikeType]
    return X if X is None else sklearn_safe_indexing(X, indices)

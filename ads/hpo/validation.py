#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common import logger
from ads.hpo.ads_search_space import get_model2searchspace
from ads.hpo.distributions import *
from ads.hpo.utils import _is_arraylike, _make_indexable, _num_samples, _safe_indexing
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline


ADS_DISTRIBUTIONS = (
    UniformDistribution,
    LogUniformDistribution,
    DiscreteUniformDistribution,
    IntUniformDistribution,
    IntLogUniformDistribution,
    CategoricalDistribution,
)


def assert_model_is_supported(estimator):
    if estimator.__class__ not in get_model2searchspace().keys():
        raise NotImplementedError(
            "{} is not supported.".format(estimator.__class__.__name__)
        )


def assert_tuner_is_fitted(estimator, msg=None):
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this method."
        )

    if not hasattr(estimator, "tune"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    attributes = ["_n_splits", "_sample_indices", "_scorer", "_study"]

    if not all([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def assert_is_estimator(estimator):
    assert hasattr(estimator, "fit"), "Estimator must implement fit"


def validate_pipeline(model):
    if isinstance(model, list):
        assert all((isinstance(m, tuple) for m in model)), "Model is not a pipeline."
        return Pipeline(model)
    return model


def validate_search_space(params, param_distributions):
    assert isinstance(
        param_distributions, dict
    ), "Must pass a dictionary to <code>search_space</code>!"

    params_to_delete = []
    for param, distribution in param_distributions.items():
        if not param in params:
            logger.warning(
                f"Ignoring {param} as it is not a hyperparameter of the model."
            )
            params_to_delete.append(param)

        if (
            isinstance(distribution, float)
            or isinstance(distribution, int)
            or isinstance(distribution, str)
        ):
            param_distributions.update({param: CategoricalDistribution([distribution])})
        elif isinstance(distribution, list):
            param_distributions.update({param: CategoricalDistribution(distribution)})
        elif not isinstance(distribution, ADS_DISTRIBUTIONS):
            raise ValueError(
                "<code>search_space</code> only accept numbers, strings, list or distributions."
            )
    for param in params_to_delete:
        del param_distributions[param]


def assert_strategy_valid(param_distributions, new_strategy, old_strategy=None):
    if isinstance(new_strategy, str):
        assert new_strategy in [
            "perfunctory",
            "detailed",
        ], "Valid values of <code>strategy</code> are `perfunctory` and `detailed`."
    else:
        assert isinstance(
            new_strategy, dict
        ), "Valid <code>strategy</code> type are string and dictionary."
        if old_strategy:
            for name, distribution in new_strategy.items():
                if name in param_distributions.keys():
                    assert isinstance(
                        distribution, param_distributions[name].__class__
                    ), "Cannot change the distribution of existing params."
                    if isinstance(distribution, CategoricalDistribution):
                        assert set(param_distributions[name].choices) == set(
                            distribution.choices
                        ), "Does not support updating the list of values for categorical distributions."


def validate_fit_params(
    X,  # type: TwoDimArrayLikeType
    fit_params,  # type: Dict
    indices,  # type: OneDimArrayLikeType
):
    # type: (...) -> Dict

    fit_params_validated = {}
    for key, value in fit_params.items():

        # NOTE Original implementation:
        # https://github.com/scikit-learn/scikit-learn/blob/ \
        # 2467e1b84aeb493a22533fa15ff92e0d7c05ed1c/sklearn/utils/validation.py#L1324-L1328
        # Scikit-learn does not accept non-iterable inputs.
        # This line is for keeping backward compatibility.
        # (See: https://github.com/scikit-learn/scikit-learn/issues/15805)
        if not _is_arraylike(value) or _num_samples(value) != _num_samples(X):
            fit_params_validated[key] = value

        else:
            fit_params_validated[key] = _make_indexable(value)
            fit_params_validated[key] = _safe_indexing(
                fit_params_validated[key], indices
            )
    return fit_params_validated


def validate_params_for_plot(params, param_distributions):
    params = [] if params is None else params
    if not set(params).issubset(set(param_distributions.keys())):
        raise ValueError("Not all the params are in the search space.")

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

try:
    from mlx import LimeExplainer
    from mlx import FDExplainer, AleExplainer
    from mlx import PermutationImportance
    from mlx.whatif import WhatIf
except:
    pass
from ads.common import logger
from ads.dataset.helper import is_text_data

import pandas as pd
from ads.common.decorator.deprecate import deprecated


def _reset_index(x):
    assert isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)
    return x.reset_index(drop=True)


def check_tabular_or_text(est, X):
    """
    Returns "text" if a text dataset, "tabular" otherwise.

    Parameters
    ----------
    est : ADSModel
        Model to explain.
    X : pandas.DataFrame
        Dataset.

    Return
    ------
    str
        "text" or "tabular"
    """
    return "text" if is_text_data(X) else "tabular"


@deprecated(
    details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
    raise_error=True,
)
def init_lime_explainer(
    explainer,
    est,
    X_train,
    y_train,
    mode,
    class_names=None,
    selected_features=None,
    client=None,
    batch_size=16,
    surrogate_model="linear",
    num_samples=5000,
    exp_sorting="absolute",
    discretization="decile",
    scale_weight=True,
):
    """
    Initializes a local LIME Explainer. Also supports aggregate local
    explanations, which approximates a global behavior based on multiple local explanations.

    Supports both tabular and text datasets. The explainer is initialized to the defaults
    and can be updated with `MLXGlobalExplainer.configure_feature_importance()` or
    `MLXLocalExplainer.configure_local_explainer()`.

    Parameters
    ----------
    explainer : LimeExplainer, None
        If the explainer has previously been initialized, it can be passed in to avoid
        creating a new explainer object. If `None`, a new `LimeExplainer` instance will be created.
    est : ADSModel
        Model to explain.
    X_train : pandas.DataFrame
        Training dataset.
    y_train : pandas.DataFrame/Series
        Training labels.
    mode : str
        'classification' or 'regression'.
    class_names : list
        List of target names.
    selected_features : list[int], optional
        Pass the selected features information to the explainer. Defaults value is `None`.
    client : Dask Client
        Specifies that Dask Client object to use in MLX. If None, no parallelization.
    batch_size : int
        Number of local explanations that are batched and processed by each Dask worker
        in parallel.
    surrogate_model : str
        Surrogate model to approximate the local behavior of the ML model. Can be
        'linear' or 'decision_tree'.
    num_samples : int
        Number of samples the local explainer generates in the local neighborhood
        around the sample to explain to fit the surrogate model.
    exp_sorting : str
        Order of how to sort the feature importances. Can be 'absolute' or 'ordered'.
        Absolute ordering orders based on the absolute values, while ordered considers
        the sign of the feature importance values.
    discretizer : str
        Method to discretize continuous features in the local explainer. Supports 'decile',
        'quartile', 'entropy', and `None`. If `None`, the continuous feature values are
        used directly. If not None, each continuous feature is discretized and treated
        as a categorical feature.
    scale_weight : bool
        Normalizes the feature importance coefficients from the local explainer to sum to one.

    Return
    ------
    :class:`mlx.LimeExplainer`
    """
    if explainer is None:
        exp = LimeExplainer()
    else:
        if not isinstance(explainer, LimeExplainer):
            raise TypeError(
                "Invalid explainer provided to "
                "init_lime_explainer: {}".format(type(explainer))
            )
        exp = explainer
    exp_type = check_tabular_or_text(est, X_train)
    exp.set_config(
        type=exp_type,
        mode=mode,
        discretizer=discretization,
        client=client,
        batch_size=batch_size,
        scale_weight=scale_weight,
        surrogate_model=surrogate_model,
        num_samples=num_samples,
        exp_sorting=exp_sorting,
        kernel_width="dynamic",
    )
    exp.fit(
        est,
        X_train,
        y=y_train,
        target_names=class_names,
        selected_features=selected_features,
    )
    return exp


@deprecated(
    details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
    raise_error=True,
)
def init_permutation_importance_explainer(
    explainer,
    est,
    X_train,
    y_train,
    mode,
    class_names=None,
    selected_features=None,
    client=None,
    random_state=42,
):
    """
    Initializes a Global Feature Permutation Importance Explainer.

    Supported for tabular datasets only.

    The explainer is initialized to the defaults and can be updated with
    MLXGlobalExplainer.configure_feature_importance().

    Parameters
    ----------
    explainer : PermutationImportance, None
        If the explainer has previously been initialized, it can be passed in to avoid
        creating a new explainer object. If `None`, a new `PermutationImportance` explainer
        will be created.
    est : ADSModel
        Model to explain.
    X_train : pandas.DataFrame
        Training dataset.
    y_train : pandas.DataFrame/Series
        Training labels.
    mode : str
        'classification' or 'regression'.
    class_names : list, optional
        List of target names. Default value is `None`
    selected_features : list[str], list[int], optional
        Pass the selected features list to the explainer. Defaults value is `None`.
    client : Dask Client, optional
        Specifies that Dask Client object to use in MLX. If `None`, no parallelization.
    random_state : int, optional
        Random seed, by default 42.

    Return
    ------
    :class:`mlx.PermutationImportance`
    """
    if explainer is None:
        exp = PermutationImportance()
    else:
        if not isinstance(explainer, PermutationImportance):
            raise TypeError(
                "Invalid explainer provided to "
                "init_permutation_importance_explainer: {}".format(type(explainer))
            )
        exp = explainer
    if check_tabular_or_text(est, X_train) == "text":
        raise TypeError(
            "Global feature importance explainers are currently not "
            "supported for text datasets."
        )
    exp.set_config(mode=mode, client=client, random_state=random_state)
    try:
        exp.fit(
            est,
            X_train,
            y=y_train,
            target_names=class_names,
            selected_features=selected_features,
        )
    except Exception as e:
        logger.error(
            f"Unable to construct the PermutationImportance explainer based on the MLX config "
            f"and fit to the train data due to: {e}."
        )
        raise e
    return exp


@deprecated(
    details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
    raise_error=True,
)
def init_partial_dependence_explainer(
    explainer, est, X_train, y_train, mode, class_names=None, client=None
):
    """
    Initializes a Global feature dependence explainer.

    Supports one and two feature partial dependence plots (PDP) and one feature individual
    conditional expectation plots (ICE). Currently only supported for tabular datasets
    (text is not supported).

    The explainer is initialized to the defaults and can be updated with
    `MLXGlobalExplainer.configure_partial_dependence()`.

    Parameters
    ----------
    explainer : FDExplainer
        If the explainer has previously been initialized, it can be passed in to avoid
        creating a new explainer object. If None, a new `FDExplainer` will be created.
    est : ADSModel
        Model to explain.
    X_train : pandas.DataFrame
        Training dataset.
    y_train : pandas.DataFrame/Series
        Training labels.
    mode : str
        'classification' or 'regression'.
    class_names : list
        List of target names.
    client : Dask Client
        Specifies that Dask Client object to use in MLX. If None, no parallelization.

    Return
    ------
    :class:`mlx.FDExplainer`
    """
    if explainer is None:
        exp = FDExplainer()
    else:
        if not isinstance(explainer, FDExplainer):
            raise TypeError(
                "Invalid explainer provided to "
                "init_partial_dependence_explainer: {}".format(type(explainer))
            )
        exp = explainer
    if check_tabular_or_text(est, X_train) == "text":
        raise TypeError(
            "Global partial dependence explainers are currently not "
            "supported for text datasets."
        )
    exp.set_config(mode=mode, client=client)
    exp.fit(est, X_train, y=y_train, target_names=class_names)
    return exp


@deprecated(
    details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
    raise_error=True,
)
def init_ale_explainer(
    explainer, est, X_train, y_train, mode, class_names=None, client=None
):
    """
    Initializes a Global Accumulated Local Effects(ALE) Explainer.

    Supports one feature ALE plots. Supported for tabular datasets
    (text is not supported).

    The explainer is initialized to the defaults and can be updated with
    `MLXGlobalExplainer.configure_accumulated_local_effects()`.

    Parameters
    ----------
    explainer : AleExplainer
        If the explainer has previously been initialized, it can be passed in to avoid
        creating a new explainer object. If None, a new AleExplainer will be created.
    est : ADSModel
        Model to explain.
    X_train : pandas.DataFrame
        Training dataset.
    y_train : pandas.DataFrame/Series
        Training labels.
    mode : str
        "classification" or "regression".
    class_names : list, optional
        List of target names. Default value is `None`.
    client : Dask Client, optional
        Specifies that Dask Client object to use in MLX. If `None`, no parallelization.

    Return
    ------
    :class:`mlx.FDExplainer`
    """
    if explainer is None:
        exp = AleExplainer()
    else:
        if not isinstance(explainer, AleExplainer):
            raise TypeError(
                "Invalid explainer provided to "
                "init_partial_dependence_explainer: {}".format(type(explainer))
            )
        exp = explainer
    if check_tabular_or_text(est, X_train) == "text":
        raise TypeError(
            "Global partial dependence explainers are currently not "
            "supported for text datasets."
        )
    exp.set_config(mode=mode, client=client)
    exp.fit(est, X_train, y=y_train, target_names=class_names)
    return exp


@deprecated(
    details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
    raise_error=True,
)
def init_whatif_explainer(
    explainer,
    est,
    X_test,
    y_test,
    mode,
    class_names=None,
    train=None,
    target_title="target",
    random_state=42,
    **kwargs,
):
    if explainer is None:
        width = kwargs.get("width", 1100)
        exp = WhatIf(mode=mode, random_state=random_state, width=width)
    else:
        if not isinstance(explainer, WhatIf):
            raise TypeError(
                "Invalid explorer provided to "
                "init_explorer: {}".format(type(explainer))
            )
        exp = explainer
    exp_type = check_tabular_or_text(est, X_test)
    if exp_type == "text":
        raise TypeError(
            "WhatIf explainer are currently not "
            "supported for text datasets.".format(type(explainer))
        )
    exp.fit(
        model=est,
        X=X_test,
        y=y_test,
        target_names=class_names,
        train=train,
        target_title=target_title,
    )
    return exp

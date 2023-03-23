#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import pandas as pd
from ads.common import logger
from ads.explanations.base_explainer import LocalExplainer
from ads.explanations.mlx_interface import init_lime_explainer
from ads.common.decorator.deprecate import deprecated


class MLXLocalExplainer(LocalExplainer):
    """
    Local Explainer class.

    Generates explanations for single predictions from machine learning models.
    For tabular and text datasets, supports

        - Binary classification
        - Multi-class classification
        - Regression
    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self):
        super(LocalExplainer, self).__init__()
        self.explainer = None

    def explain(
        self,
        row,
        y=None,
        num_features=None,
        report_fidelity=True,
        validation_percentiles=None,
        selected_features=None,
    ):
        """
        Explains the sample, row. Returns a local explanation object.

        Parameters
        ----------
        row : pandas.DataFrame
            Pandas DataFrame of one row (instance) to explain.
        y : pandas.DataFrame/Series, optional
            True target label/value for row. Default value is `None`.
        num_features : int, optional
            Number of features to show in the local explanation. By default `None`,
            which includes all features.
        report_fidelity : bool, optional
            If `True`, the explanation quality (fidelity) is computed and shown in
            the local explanation visualization. Default value is `True`.
        validation_percentiles : list of int, optional
            List of values specifying the different regions to evaluate the local explanation model
            quality (fidelity). This is specified in percentiles of the distances from the
            instance to explain to all of the samples in the dataset used to fit the explainer.
            For example, [1, 5, 10], generates three evaluation datasets with a maximum distance from
            the instance to explain (row) of 1, 5, and 10 percentiles of the distances from the instance
            to explain (row) and all other samples in the train set. The evaluation dataset at percentile
            1 is very local to the sample to explain, the evaluation dataset at percentile 10 evaluates
            the explanation quality further away from the instance to explain (row). This can be helpful
            to see how the explanation generalizes to nearby samples. By default None ([1, 5, 15]).
        selected_features: list[int], optional
            List of the selected features (list of numbers of features). It can be any subset of
            the original features that are in the dataset provided to the model.
            Default value is None.

        Return
        ------
        LocalExplanation
            Local explanation object.
        """
        self.configure_local_explainer(selected_features=selected_features)
        assert row is not None and isinstance(row, pd.DataFrame) and row.shape[0] == 1

        labels = list(range(len(self.class_names)))

        if num_features is None:
            num_features = len(row.columns.values)

        try:
            explanation = self.explainer.compute(
                row.copy(),
                y=None if y is None else y.copy(),
                labels=labels,
                verbose=False,
                num_features=num_features,
                report_fidelity=report_fidelity,
                validation_percentiles=validation_percentiles,
            )[0]
        except IndexError as e:
            if selected_features is not None:
                raise IndexError(
                    f"Unable to generate local explanations due to: {e}. "
                    f"selected_features must be a list of features within the bounds of the existing features "
                    f"(that were provided to model). Provided selected_features: {selected_features}."
                )
        except Exception as e:
            logger.error(f"Unable to generate local explanations due to: {e}.")
            raise e

        return LocalExplanation(explanation, self.class_names)

    def configure_local_explainer(self, **kwargs):
        """
        Validates the local explanation configuration parameters and initializes the local
        explainer with the provided configuration parameters.

        Supported configuration options in kwargs:
            - `surrogate_model` (str): Surrogate model to use. Can be 'linear' or 'decision_tree'.
            - `num_samples` (int): Number of generated samples to fit the surrogate model.
            - `exp_sorting` (str): Feature importance sorting. Can be 'absolute' or 'ordered'.
            - `scale_weight` (bool): Normalizes the feature importance coefficients from LIME to sum to one.
            - `client`: Only allowed to be None to disable parallelization.
            - `batch_size` (int): Number of local explanations per Dask worker.
            - `random_state` (`None` or `int` or instance of `RandomState`): the random state.
            - `selected_features` (`None`, or `list`): list of the selected features numbers.

        Parameters
        ----------
        **kwargs : dict
            Keyword parameter dictionary.

        Return
        ------
        MLXLocalExplainer
            Modified instance (self)
        """
        avail_args = [
            "client",
            "surrogate_model",
            "num_samples",
            "exp_sorting",
            "scale_weight",
            "batch_size",
            "selected_features",
        ]
        for k, v in kwargs.items():
            if k not in avail_args:
                raise ValueError(
                    "Unexpected argument for the local explainer: {}".format(k)
                )
        if kwargs.get("client", None) is not None:
            raise ValueError(
                "Invalid client provided. Currently only supports disabling parallelization "
                "by setting client=None"
            )
        if kwargs.get("surrogate_model", None) not in ["linear", "decision_tree", None]:
            raise ValueError(
                "Invalid surrogate_model provided. Currently only supports linear or decision_tree"
            )
        selected_features = kwargs.get("selected_features", None)
        if selected_features is not None and not isinstance(selected_features, list):
            raise ValueError(
                f"selected_features ({selected_features}) value must be a list of features, "
                f"but it is of type: {type(selected_features)}."
            )

        self._init_explainer(**kwargs)
        return self

    def summary(self):
        """
        Displays detailed information about the local LIME explainer.

        Return
        ------
        str
            HTML object representing the explainer summary.
        """
        if self.explainer is None:
            self.configure_local_explainer()
        return self.explainer.show_in_notebook()

    def _init_explainer(self, **kwargs):
        """
        Internal function to initialize the local explainer.

        Parameters
        ----------
        **kwargs : dict
            Keyword parameter dictionary.
        """
        if self.mode == "regression":
            self.class_names_ = ["Target"]
        self.explainer = init_lime_explainer(
            self.explainer,
            self.est,
            self.X_train,
            self.y_train,
            self.mode,
            class_names=self.class_names,
            **kwargs,
        )


class LocalExplanation:
    """
    Local explanation object constructed by the :class:`MLXLocalExplainer`.

    Contains functions to visualize the explanation and extract the
    raw explanation data.
    """

    def __init__(self, explanation, class_names):
        self.explanation = explanation
        self.class_names = class_names
        if isinstance(self.class_names, np.ndarray):
            self.class_names = self.class_names.tolist()

    def show_in_notebook(
        self, mode="lime", labels=None, colormap=None, return_wordcloud=False, **kwargs
    ):
        """
        Generate a local explanation visualization for this explanation object.

        Contains:
            - Information about the model's prediction and the true target label/value.
            - The instance being explained.
            - Information about the local explainer configuration parameters.
            - Legend describing how to interpret the explanation.
            - The actual explanation (ordered list of +/- feature importances).
            - Quality (fidelity) evaluation of the explanation.

        Parameters
        ----------
        mode : str
            Type of visualization to generate for the explanation. `mode` can be one of the following:
                - **lime**: Horizontal bar chart where each feature contributes either to an increase
                        or decrease in the target value.
                - **stacked**: Stacked horizontal bar chart where each feature contributes either to an
                        increase or decrease in the target value.
                - **dual**: Horizontal bar chart where each feature can contribute both to an increase
                        or decrease in the target value (only supported when the :class:`MLXLocalExplainer`
                        surrogate model is set to "decision_tree").
                - **wordcloud**: Word cloud highlighting the important features/words for a given target
                             label/value. Features/ or words with higher importance are larger than those
                             of a lower importance.
        labels : tuple, list, int, bool, str
            Label indices or name of label to visualize. If `None`, all of the labels that the
            explanation was generated for will be visualized. Default value is `None`.
        colormap : list of str, optional
            List of standard colormaps to use for the wordclouds. One per label. Defaults to `None`.
        return_wordcloud : bool, optional
            If `True`, the wordcloud objects are returned instead of visualized. Defaults to `False`.
        **kwargs : dict
            Keyword dictionary for configuring the wordclouds.

        Return
        ------
        HTML or pyplot figure
            Returns an HTML object for this local explanation. If `mode=wordcloud` and `return_wordcloud=True`,
            returns the wordcloud
        """
        if labels is None:
            labels = (0,)
        if np.isscalar(labels):
            labels = [labels]
        labels = [
            self.class_names.index(label) if isinstance(label, str) else label
            for label in labels
        ]
        return self.explanation.show_in_notebook(
            mode=mode,
            labels=labels,
            colormap=colormap,
            return_wordcloud=return_wordcloud,
            **kwargs,
        )

    def get_diagnostics(self):
        """
        Extracts the raw explanation and evaluation data from the explanation object to generate
        the visualizations.

        Return
        ------
        dict
            Dictionary containing the raw explanation/evaluation data.
        """
        return self.explanation.get_diagnostic()

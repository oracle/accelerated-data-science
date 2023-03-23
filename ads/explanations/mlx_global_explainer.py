#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from abc import ABC, abstractmethod

from ads.common import logger, utils
from ads.explanations.base_explainer import GlobalExplainer
from ads.explanations.mlx_interface import check_tabular_or_text
from ads.explanations.mlx_interface import init_lime_explainer
from ads.explanations.mlx_interface import init_permutation_importance_explainer
from ads.explanations.mlx_interface import (
    init_partial_dependence_explainer,
    init_ale_explainer,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.decorator.deprecate import deprecated


class MLXGlobalExplainer(GlobalExplainer):
    """
    Global Explainer class.

    Generates global explanations to help understand the general model
    behavior. Supported explanations:

        - (Tabular) Feature Permutation Importance.
        - (Tabular) Partial Dependence Plots (PDP) & Individual Conditional
          Expectation (ICE).
        - (Text) Aggregate local explanations (global explanation approximation
          constructed from multiple local explanations).

    Supports:

        - Binary classification.
        - Multi-class classification.
        - Regression.

    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self):
        super(GlobalExplainer, self).__init__()
        self.explainer = None
        self.selected_features = None
        self.pdp_explainer = None
        self.ale_explainer = None

    def compute_feature_importance(
        self,
        n_iter=20,
        sampling=None,
        balance=False,
        scoring_metric=None,
        selected_features=None,
    ):
        """
        Generates a global explanation to help understand the general behavior
        of the model. This explainer identifies which features are most important
        to the model.

        If the dataset is tabular, computes a global feature permutation importance
        explanation. If the dataset is text, approximates a global explanation by
        generating and aggregating multiple local explanations.

        Parameters
        ----------
        n_iter : int, optional
            Number of iterations of the permutation importance algorithm to
            perform. Increasing this value increases the quality/stability of
            the explanation, but increases the explanation time. Default value is 20.
        sampling : dict, optional
            If not `None`, the dataset is clustered or sampled according to the
            provided technique. `sampling` is a dictionary containing the technique
            to use and the corresponding parameters. Format is described below:

            - `technique`: Either `cluster` or `random`.
            - If `cluster`, also requires:

                - `eps`: Maximum distance between two samples to be considered
                  in the same cluster.
                - `min_samples`: Minimum number of samples to include in each
                  cluster.

            - If `random`, also requires:

                - `n_samples`: Number of samples to return.

            By default None. Note that text datasets are always sampled. If not provided
            with a sampling option, defaults to 40 random samples.
        balance : bool, optional
            If True, the dataset will be balanced via sampling. If 'sampling' is not
            set, the sampling technique defaults to 'random'.
        scoring_metric : string, optional
            If specified, propegates a string indicating the supported scoring metric.
            The scoring metrics available out of the box are the ones made available
            by ScyPy. Supported Metrics:

            - Multi-class Classification

                `f1_weighted`, `f1_micro`, `f1_macro`, `recall_weighted`, `recall_micro`, `recall_macro`,
                `accuracy`, `balanced_accuracy`, `roc_auc`, `precision_weighted`, `precision_macro`,
                `precision_micro`

            - Binary Classification

                Same as multi-class classification

            - Regression

                `r2`, `neg_mean_squared_error`, `neg_root_mean_squared_error`, `neg_mean_absolute_error`,
                `neg_median_absolute_error`, `neg_mean_absolute_percentage_error`,
                `neg_symmetric_mean_absolute_percentage_error`
        selected_features: list[str], list[int], optional
            List of the selected features. It can be any subset of
            the original features that are in the dataset provided to the model.
            Default value is None.

        Returns
        -------
        :class:FeatureImportance
            `FeaturePermutationImportance` explanation object.

        """
        self.selected_features = selected_features
        self.configure_feature_importance(selected_features=self.selected_features)
        if self.explainer.config.type == "text":
            labels = list(range(len(self.class_names)))
            # The requirement to downsample the text datasets should be fixed at somepoint
            if sampling is None:
                sampling = {"technique": "random", "n_samples": 40}
            explanation = self.explainer.explain_aggregate_local(
                self.X_test, sampling=sampling, labels=labels
            )
        else:
            if self.mode_ == "regression":
                allowed_metrics = [
                    "r2",
                    "neg_mean_squared_error",
                    "neg_root_mean_squared_error",
                    "neg_mean_absolute_error",
                    "neg_median_absolute_error",
                    "neg_mean_absolute_percentage_error",
                    "neg_symmetric_mean_absolute_percentage_error",
                ]
            elif self.mode_ == "classification" and len(self.class_names) == 2:
                # Binary classification
                allowed_metrics = [
                    "f1_weighted",
                    "f1_micro",
                    "f1_macro",
                    "recall_weighted",
                    "recall_micro",
                    "recall_macro",
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "precision_weighted",
                    "precision_macro",
                    "precision_micro",
                ]
            else:
                # Multiclass classification
                allowed_metrics = [
                    "f1_weighted",
                    "f1_micro",
                    "f1_macro",
                    "recall_weighted",
                    "recall_micro",
                    "recall_macro",
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "precision_weighted",
                    "precision_macro",
                    "precision_micro",
                ]
            if scoring_metric not in allowed_metrics and scoring_metric is not None:
                raise Exception(
                    "Scoring Metric not supported for this type of problem: {}, for problem type {}, the availble supported metrics are {}".format(
                        scoring_metric, self.mode_, allowed_metrics
                    )
                )
            if balance and sampling is None:
                sampling = {"technique": "random"}
            try:
                explanation = self.explainer.compute(
                    self.X_test,
                    self.y_test,
                    n_iter=n_iter,
                    sampling=sampling,
                    balance=balance,
                    scoring_metric=scoring_metric,
                )
            except IndexError as e:
                if selected_features is not None:
                    raise IndexError(
                        f"Unable to calculate permutation importance due to: {e}. "
                        f"selected_features must be a list of features within the bounds of the existing features "
                        f"(that were provided to model). Provided selected_features: {selected_features}."
                    )
            except Exception as e:
                logger.error(
                    f"Unable to calculate permutation importance scores due to: {e}."
                )
                raise e
        return FeatureImportance(explanation, self.class_names, self.explainer.config)

    def compute_partial_dependence(
        self, features, partial_range=(0.00, 1.0), num_samples=30, sampling=None
    ):
        """
        Generates a global partial dependence plot (PDP) and individual conditional
        expectation (ICE) plots to help understand the relationship between feature
        values and the model target.

        Only supported for tabular datasets.

        Parameters
        ----------

        features : list of int, list of str
            List of feature names or feature indices to explain.
        partial_range : tuple, optional
            2-tuple with the minimum and maximum percentile values to consider for the PDP from
            the feature's train distribution. Must be between 0.0 and 1.0.
            Defaults to `partial_range = (0.05, 0.95)`.
        num_samples : int, optional
            Maximum number of samples to generate for each feature within the
            `partial_range` of its value distribution. Increasing this value
            generates more points to evaluate, but increases the explanation
            time. If there are fewer unique values for a feature within the
            `partial_range`, the number of unique values is selected. For two-feature
            PDP, the total number of evaluated samples is the multiplication
            of `num_samples`. Default value is 30.
        sampling : dict, optional
            If not None, the dataset will be clustered or sampled according to the
            provided technique. 'sampling' is a dictionary containing the technique
            to use and the corresponding parameters. Format is described below:

            - `technique`: Either "cluster" or "random".
            - `cluster` also requires:

                - `eps`: Maximum distance between two samples to be considered
                  in the same cluster.
                - `min_samples`: Minimum number of samples to include in each
                  cluster.

            - `random` also requires:

                - 'n_samples': Number of samples to return.

            Default value is `None` (no sampling).

        Returns
        -------
        :class:MLXPartialDependencies
            `MLXPartialDependencies` object.

        """
        if self.pdp_explainer is None:
            self._init_partial_dependence()

        # Wrap in a list if a list is not provided
        if not isinstance(features, list):
            features = [features]

        # Convert to uppercase to be case-insensitive
        features = [str(f).upper() for f in features]
        feature_names = np.char.upper(self.X_train.columns.tolist())

        # Fail if we were not provided valid feature names
        if not all(np.isin(features, feature_names)):
            print("One or more features (%s) does not exist in data." % str(features))
            print("Existing features: %s" % str(feature_names))
            return

        # Extract the feature ids
        feature_ids = np.where(np.isin(feature_names, features))[0].tolist()

        if check_tabular_or_text(self.est, self.X_train) == "tabular":
            if len(feature_ids) > 2:
                raise ValueError("Maximum number of partial dependency features is 2.")

            return MLXPartialDependencies(
                pdp=self.pdp_explainer.compute(
                    data=self.X_train,
                    partial_ids=feature_ids,
                    partial_range=partial_range,
                    num_samples=num_samples,
                    sampling=sampling,
                ),
                pdp_exp=self.pdp_explainer,
            )
        else:
            raise ValueError(
                "Partial Dependence Plot is not supported for text classification dataset."
            )

    def compute_accumulated_local_effects(
        self,
        feature,
        partial_range=(0.00, 1.0),
        num_samples=30,
        sampling=None,
        corr_threshold=0.7,
    ):
        """
        Generates the accumulated local effects plots to help understand the relationship between feature
        values and the model target.

        Only supported for tabular datasets.

        Parameters
        ----------
        feature : str
            Feature name to explain.
        partial_range : tuple, optional
            Min/max percentile values to consider for the ALE from the
            feature's train distribution. Must be between 0.0 and 1.0.
            By default `partial = (0.05, 0.95)`.
        num_samples : int, optional
            Maximum number of samples to generate for each feature within the
            `partial_range` of its value distribution. Increasing this value
            generates more points to evaluate, but increases the explanation
            time. If there are fewer unique values for a feature within the
            `partial_range`, the number of unique values is selected.
        sampling : dict, optional
            If not `None`, the dataset is clustered or sampled according to the
            provided technique. `sampling` is a dictionary containing the technique
            to use and the corresponding parameters. The format is:

            - `technique`: Can be either "cluster" or "random".
            - `cluster` also requires:

                - `eps`: Maximum distance between two samples to be considered
                  in the same cluster.
                - `min_samples`: Minimum number of samples to include in each
                  cluster.

            - `random` also requires:

                - `n_samples`: Number of samples to return.

            Defaults to `None` (no sampling).

        corr_threshold : float, optional
             Value between 0.0 and 1.0 for which a feature is considered highly correlated with
             another feature (Default = 0.7).

        Returns
        -------
        :class:MLXAccumulatedLocalEffects
            `AccumulatedLocalEffects` explanation object.

        """
        if self.ale_explainer is None:
            self._init_accumulated_local_effects()

        # Wrap in a list if a list is not provide, to be able support list of two features in the near future.
        if not isinstance(feature, list):
            feature = [feature]

        # Convert to uppercase to be case-insensitive
        feature = [str(f).upper() for f in feature]
        feature_names = np.char.upper(self.X_train.columns.tolist())

        # Fail if we were not provided valid feature names
        if not all(np.isin(feature, feature_names)):
            print("One or more features (%s) does not exist in data." % str(feature))
            print("Existing features: %s" % str(feature_names))
            return

        # Extract the feature ids
        feature_ids = np.where(np.isin(feature_names, feature))[0].tolist()

        if check_tabular_or_text(self.est, self.X_train) == "tabular":
            if len(feature_ids) > 1:
                raise ValueError(
                    "Maximum number of Accumulated Local Effects features is 1."
                )

            return MLXAccumulatedLocalEffects(
                ale=self.ale_explainer.compute(
                    data=self.X_train,
                    partial_ids=feature_ids,
                    partial_range=partial_range,
                    num_samples=num_samples,
                    sampling=sampling,
                    corr_threshold=corr_threshold,
                ),
                ale_exp=self.ale_explainer,
            )
        else:
            raise ValueError(
                "Accumulated Local Effects Plot is not supported for text classification dataset."
            )

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def show_in_notebook(self):  # pragma: no cover
        """
        Generates and visualizes the global feature importance explanation.
        """
        with utils.get_progress_bar(3, description="Model Explanation") as bar:
            bar.update("begin computing")
            bar.update("calculating feature importance")
            explainer_holder = self.compute_feature_importance(
                selected_features=self.selected_features
            )
            plot1 = explainer_holder.show_in_notebook()
            bar.update("calculating partial dependence plot")
            pdp_plot_feature_name = explainer_holder.explanation
            # pdp_plot_feature_name = explainer_holder.explanation.get_global_explanation().index[0]
            pdp_plot = self.compute_partial_dependence([pdp_plot_feature_name])
            # plot2 = pdp_plot.show_in_notebook()

        from IPython.core.display import display, HTML

        display(HTML(plot1.data))
        # display(HTML(plot1.data + plot2.data))

    def configure_feature_importance(self, **kwargs):
        """
        Validates and initializes the feature importance explainer based on the provided
        configuration parameters in kwargs. Tabular datasets use the feature permutation
        importance explainer, text datasets use the aggregate local explainer.

        Supported configuration options:

        - For tabular datasets:

            - `client`: Currently only allowed to be None to disable parallelization.
            - `random_state`: None, int, or instance of Randomstate.
            - `selected_features`: None, or list of the selected features.

        - For text datasets:

            - `surrogate_model`: Surrogate model to use. Can be 'linear' or 'decision_tree'.
            - `num_samples`: Number of generated samples to fit the surrogate model. Int.
            - `exp_sorting`: Feature importance sorting. Can be 'absolute' or 'ordered'.
            - `scale_weight`: Normalizes the feature importance coefficients from LIME to sum to one.
            - `client`: Currently only allowed to be None to disable parallelization.
            - `batch_size`: Number of local explanations per Dask worker.
            - `random_state`: None, int, or instance of Randomstate.
            - `selected_features`: None, or list of the selected features.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.

        Returns
        -------
        MLXGlobalExplainer
            the modified instance (self)

        """

        if check_tabular_or_text(self.est, self.X_train) == "tabular":
            avail_args = ["client", "random_state", "selected_features"]
        else:
            avail_args = [
                "client",
                "random_state",
                "surrogate_model",
                "num_samples",
                "exp_sorting",
                "scale_weight",
                "batch_size",
                "selected_features",
            ]

        for k, _ in kwargs.items():
            if k not in avail_args:
                raise ValueError(
                    "Unexpected argument for the feature importance explainer: {}".format(
                        k
                    )
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
        selected_features = kwargs.get("selected_features")
        if selected_features is not None and not isinstance(selected_features, list):
            raise ValueError(
                f"selected_features ({selected_features}) value must be a list of features, "
                f"but it is of type: {type(selected_features)}."
            )

        self._init_feature_importance(**kwargs)
        return self

    def configure_partial_dependence(self, **kwargs):
        """
        Validates and initializes the partial dependence explainer based on the provided
        configuration parameters in kwargs. Only supports tabular datasets.

        Supported configuration options:
            client: Currently only supports 'None' to disable parallelization.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.

        Returns
        -------
        MLXGlobalExplainer
            the modified instance (self)
        """

        for k, _ in kwargs.items():
            if k not in ["client"]:
                raise ValueError(
                    "Unexpected argument for the partial dependence explainer: {}".format(
                        k
                    )
                )
        if kwargs.get("client", None) is not None:
            raise ValueError(
                "Invalid client provided. Currently only supports disabling parallelization "
                "by setting client=None"
            )
        self._init_partial_dependence(**kwargs)
        return self

    def configure_accumulated_local_effects(self, **kwargs):
        """
        Validates and initializes the accumulated local effects explainer based on the provided
        configuration parameters in kwargs. Only supports tabular datasets.

        Supported configuration options:

        - client: Currently only supports 'None' to disable parallelization.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.

        Returns
        -------
        MLXGlobalExplainer
            the modified instance (self)
        """

        for k, _ in kwargs.items():
            if k not in ["client"]:
                raise ValueError(
                    "Unexpected argument for the accumulated local effects explainer: {}".format(
                        k
                    )
                )
        if kwargs.get("client", None) is not None:
            raise ValueError(
                "Invalid client provided. Currently only supports disabling parallelization "
                "by setting client=None"
            )
        self._init_accumulated_local_effects(**kwargs)
        return self

    def feature_importance_summary(self):
        """
        Displays detailed information about the feature importance explainer.

        Returns
        -------
        str
            HTML object representing the explainer summary.

        """

        if self.explainer is None:
            self.compute_feature_importance(selected_features=self.selected_features)
        return self.explainer.show_in_notebook()

    def partial_dependence_summary(self):
        """
        Displays detailed information about the partial dependence explainer.

        Returns
        -------
        str
            HTML object representing the explainer summary.
        """

        if self.pdp_explainer is None:
            self._init_partial_dependence()
        return self.pdp_explainer.show_in_notebook()

    def accumulated_local_effects_summary(self):
        """
        Displays detailed information about the accumulated local effects explainer.

        Returns
        -------
        str
            HTML object representing the explainer summary.
        """

        if self.ale_explainer is None:
            self._init_accumulated_local_effects()
        return self.ale_explainer.show_in_notebook()

    def _init_feature_importance(self, **kwargs):
        """
        Internal function to initialize the feature importance explainer. Tabular datasets
        use the feature permutation importance explainer, text datasets use the aggregate local
        explainer.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.
        """
        if self.mode == "regression":
            self.class_names_ = ["Target"]
        if check_tabular_or_text(self.est, self.X_train) == "tabular":
            self.explainer = init_permutation_importance_explainer(
                self.explainer,
                self.est,
                self.X_train,
                self.y_train,
                self.mode,
                class_names=self.class_names,
                **kwargs,
            )
        else:
            self.explainer = init_lime_explainer(
                self.explainer,
                self.est,
                self.X_train,
                self.y_train,
                self.mode,
                class_names=self.class_names,
                **kwargs,
            )

    def _init_partial_dependence(self, **kwargs):
        """
        Internal function to initialize the partial dependence explainer.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.
        """
        if self.mode == "regression":
            self.class_names_ = ["Target"]
        self.pdp_explainer = init_partial_dependence_explainer(
            self.pdp_explainer,
            self.est,
            self.X_train,
            self.y_train,
            self.mode,
            class_names=self.class_names,
            **kwargs,
        )

    def _init_accumulated_local_effects(self, **kwargs):
        """
        Internal function to initialize the accumulated local effects explainer.

        Parameters
        ----------
        kwargs : dict
            Keyword parameter dictionary.
        """
        if self.mode == "regression":
            self.class_names_ = ["Target"]
        self.ale_explainer = init_ale_explainer(
            self.ale_explainer,
            self.est,
            self.X_train,
            self.y_train,
            self.mode,
            class_names=self.class_names,
            **kwargs,
        )


class MLXFeatureDependenceExplanation(ABC):

    __name__ = "MLXFeatureDependenceExplanation"

    def __init__(self, fd, fd_exp):
        self.fd = fd
        self.fd_exp = fd_exp

    @abstractmethod
    def show_in_notebook(
        self,
        labels=None,
        cscale="YIGnBu",
        show_distribution=True,
        discrete_threshold=0.15,
        # line_gap=0,  # will add it back after ALE starts handling two features, remember to add the doc string too
        show_correlation_warning=True,
        centered=False,
        show_median=True,
    ):  # pragma: no cover
        """
        Visualize PDP/ICE plots in the Notebook.

        Parameters
        ----------
        labels : tuple, list, int, bool, str, optional
            labels to visualize.
        cscale : str, optional
            Plotly color scale to use for the heatmap. See the standard Plotly color scales for available options
            Default value is "YIGnBu".
        show_distribution : bool, optional
            If `True`, the feature’s value distribution (from the train set) will be shown along the
            corresponding axis in the 1-feature or 2-feature plot. Default is `True`.
        discrete_threshold : float, optional
            Value between 0.0 and 1.0 indicating the fraction of unique values required for a numerical feature
            to be considered discrete or continuous. Default is 0.15.
        show_correlation_warning : bool, optional
            If `True`, the correlated feature warning is shown. Default is `True`.
        centered : bool, optional
            If `True`, ICE plots is centered based on the first value of each sample (i.e., all values are
            subtracted from the first value). Default is False.
        show_median : bool, optional
            If True, a median line is included in the ICE explanation plot. Default is True.

        Returns
        -------
        str
            Plotly HTML object containing a line chart, heat map, or violin plot for this feature dependence explanation
        """
        pass

    def as_dataframe(self):
        """
        Returns the raw explanation data as a pandas.DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the raw PDP explanation data.
        """
        return self.fd.as_dataframe()

    def get_diagnostics(self):
        """
        Extracts the raw explanation and evaluation data from the explanation object
        (Used to generate the visualizations).

        Returns
        -------
        dict
            Dictionary containing the raw explanation/evaluation data.
        """
        return self.fd.get_diagnostic()


class MLXPartialDependencies(MLXFeatureDependenceExplanation):
    """
    Represents the object constructed by the :class:`MLXGlobalExplainer`.

    Contains functions to visualize the explanation and extract raw explanation data.
    """

    __name__ = "MLXPartialDependencies"

    def __init__(self, pdp, pdp_exp):
        super(MLXPartialDependencies, self).__init__(pdp, pdp_exp)

    def show_in_notebook(
        self,
        mode="pdp",
        labels=None,
        cscale="YIGnBu",
        show_distribution=True,
        discrete_threshold=0.15,
        line_gap=0,
        show_correlation_warning=True,
        centered=False,
        show_median=True,
    ):
        """
        Visualize PDP/ICE plots in the Notebook.

        Parameters
        ----------
        mode : str, optional
            Type to visualize. Either "pdp" or "ice". Default is "pdp".
        labels : tuple, list, int, bool, str, optional
            labels to visualize.
        cscale : str, optional
            Plotly color scale to use for the heatmap. See the standard Plotly color scales for available options
            Default value is "YIGnBu".
        show_distribution : bool, optional
            If `True`, the feature’s value distribution (from the train set) will be shown along the
            corresponding axis in the 1-feature or 2-feature plot. Default is `True`.
        discrete_threshold : float, optional
            Value between 0.0 and 1.0 indicating the fraction of unique values required for a numerical feature
            to be considered discrete or continuous. Default is 0.15.
        line_gap : int, optional
            Width of the gap between values in the two-feature PDP heat map. Default is 0.
        show_correlation_warning : bool, optional
            If `True`, the correlated feature warning are shown. Default is `True`.
        centered : bool, optional
            If `True`, ICE plots are centered based on the first value of each sample (i.e., all values are
            subtracted from the first value). Default is `False`.
        show_median : bool, optional
            If `True`, a median line is included in the ICE explanation plot. Default is `True`.

        Returns
        -------
        str
            Plotly HTML object containing a line chart, heat map, or violin plot for this feature dependence explanation
        """
        return self.fd.show_in_notebook(
            mode=mode,
            labels=labels,
            cscale=cscale,
            show_distribution=show_distribution,
            discrete_threshold=discrete_threshold,
            line_gap=line_gap,
            show_correlation_warning=show_correlation_warning,
            centered=centered,
            show_median=show_median,
        )


class MLXAccumulatedLocalEffects(MLXFeatureDependenceExplanation):
    """
    Accumulated Local Effects explanation object constructed by the `:class:MLXGlobalExplainer`.

    Contains functions to visualize the explanation in a Notebook and extract the
    raw explanation data.
    """

    __name__ = "MLXAccumulatedLocalEffects"

    def __init__(self, ale, ale_exp):
        super(MLXAccumulatedLocalEffects, self).__init__(ale, ale_exp)

    def show_in_notebook(
        self,
        labels=None,
        cscale="YIGnBu",
        show_distribution=True,
        discrete_threshold=0.15,
        show_correlation_warning=True,
        centered=False,
        show_median=True,
    ):
        """
        Visualize ALE plots in the Notebook.

        Parameters
        ----------
        labels : tuple, list, int, bool, str, optional
            labels to visualize.
        cscale : str, optional
            Plotly color scale to use for the heatmap. See the standard Plotly color scales for available options
            Default value is "YIGnBu".
        show_distribution : bool, optional
            If `True`, the feature’s value distribution (from the train set) will be shown along the
            corresponding axis in the 1-feature. Default is `True`.
        discrete_threshold : float, optional
            Value between 0.0 and 1.0 indicating the fraction of unique values required for a numerical feature
            to be considered discrete or continuous. Default is 0.15.
        show_correlation_warning : bool, optional
            If `True`, the correlated feature warning will be shown. Default is `True`.
        centered : bool, optional
            If `True`, ALE plots will be centered based on the first value of each sample (i.e., all values are
            subtracted from the first value). Default is `False`.
        show_median : bool, optional
            If `True`, a median line is included in the ALE explanation plot. Default is `True`.

        Returns
        -------
        str
            Plotly HTML object containing a line chart, heat map, or violin plot for this feature dependence explanation
        """
        return self.fd.show_in_notebook(
            mode="pdp",
            labels=labels,
            cscale=cscale,
            show_distribution=show_distribution,
            discrete_threshold=discrete_threshold,
            line_gap=0,
            show_correlation_warning=show_correlation_warning,
            centered=centered,
            show_median=show_median,
        )


class FeatureImportance:
    """
    Feature Permutation Importance explanation object constructed by the
    :class:`MLXGlobalExplainer` class.

    Contains functions to visualize the explanation in a Notebook and extract the
    raw explanation data.
    """

    def __init__(self, explanation, class_names, type):
        self.explanation = explanation
        self.class_names = class_names
        self.type = type
        if isinstance(self.class_names, np.ndarray):
            self.class_names = self.class_names.tolist()

    def show_in_notebook(
        self,
        mode=None,
        show=None,
        labels=None,
        cscale="YIGnBu",
        colormap=None,
        return_wordcloud=False,
        n_features=None,
        **kwargs,
    ):
        """
        Generates a visualization for the local explanation. Depending on the type of explanation, different
        visualizations are supported. See the "mode" and "show" parameters below.

        Parameters
        ----------
        mode : str
            Type of visualization to generate. Certain visualization modes are only supported for either text or
            tabular datasets. Supported options:

            - `bar`: Generates a horizontal bar chart for the most important features (text and tabular).
            - `stacked`: Generates a stacked horizontal bar chart for the most important features (text and tabular).
            - `box_plot`: Generates a box plot for the most important features, which provides more information
              about the explanation over the different iterations of the feature permutation importance
              algorithm (tabular only).
            - `detailed`: Generates a scatter plot for the most important features, providing even more
              information about the explanation over the different iterations of the Feature Permutation
              Importance algorithm (tabular only).
            - `heatmap`: Generates a heatmap representing the average feature/word importance over multiple
              local explanations (aggregates local explanations). Average feature importance is measured by
              the fraction of local explanations where a given feature was assigned a given importance
              (text only).
            - `wordcloud`: Generates a wordcloud from the average feature importance. Features/words with
              higher importance are larger than features/words with lower importance (text only).

            Default value is "bar" for tabular and "wordcloud" for text.
        show : str
            (text only) Secondary visualization mode for configuring the visualization.
            Can be one of:

            - `absolute`: The absolute value of feature importances are shown (i.e., a feature that is highly
              important towards or against the target label is considered important).
            - `posneg`: Shows both the positive and negative global feature attributions. For bar, the features
              are ordered based on their absolute feature importance (sum of pos/neg) and a dual bar chart shows
              the fraction of local explanations where the feature contributed both towards and against the
              corresponding label. For wordcloud, two wordclouds are generated for the positive and negative feature
              importances. Only valid for mode=bar and mode=wordcloud. `mode=heatmap` defaults to `show=absolute`.

        labels : tuple, list, int, bool, str
            (text only) Label indices to visualize. If `None`, all of the labels that the explanation was generated for
            will be visualized. By default None.
        cscale : str, optional
            Plotly color scale to use for the heatmap. See the standard Plotly color scales for available
            options. Default value is "YIGnBu".
        colormap : list of str, optional
            List of colormaps to use for the wordclouds. One per label. Defaults to `None`.
        return_wordcloud : bool, optional
            If `True`, the generated wordcloud objects are returned instead of visualized. Defaults
            to `False`.
        n_features : int, optional
            (tabular only). Allows the user to visualize a subset of the top-N most important features from the explainer.
            If `n_features` is `None` or greater than the total number of features, all features are shown. If
            `n_features` is not an `int` or <= 0, an exception is thrown.
        kwargs : dict
            Keyword arguments for configuring the wordclouds.

        Returns
        -------
        str, list of wordcloud
            HTML string for the visualization or list of generated wordcloud objects if `return_wordcloud=True`,
            two per label (+/-).
        """
        if self.type.type == "text":
            if labels:
                labels = [
                    self.class_names.index(label) if isinstance(label, str) else label
                    for label in labels
                ]
            else:
                labels = list(range(len(self.class_names)))
            if len(self.class_names) == 2:
                return self.explanation.show_in_notebook()
            else:
                return self.explanation.show_in_notebook(
                    mode=mode if mode else "wordcloud",
                    show=show,
                    labels=labels,
                    cscale=cscale,
                    colormap=colormap,
                    return_wordcloud=return_wordcloud,
                    **kwargs,
                )
        else:
            if labels:
                raise ValueError("label is supported only for text explanation.")
            return self.explanation.show_in_notebook(
                n_features=n_features, mode=mode if mode else "bar"
            )

    def get_global_explanation(self):
        """
        Returns the raw global explanation data only.

        Returns
        -------
        dict
            Dictionary containing raw explanation data.
        """
        return self.explanation.get_global_explanation()

    def get_diagnostics(self):
        """
        Extracts the raw explanation and evaluation data from the explanation object
        (Used to generate the visualizations).

        Returns
        -------
        dict
            Dictionary containing the raw explanation/evaluation data.
        """
        return self.explanation.get_diagnostic()


class GlobalExplanationsException(TypeError):
    def __init__(self, msg):
        super(GlobalExplanationsException, self).__init__(msg)

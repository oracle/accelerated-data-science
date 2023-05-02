#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import time
import sys
import warnings
from abc import ABC, abstractmethod, abstractproperty
import math
import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.dummy import DummyClassifier, DummyRegressor

import matplotlib.pyplot as plt

import ads
from ads.common.utils import (
    ml_task_types,
    wrap_lines,
    is_documentation_mode,
    is_notebook,
)
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.decorator.deprecate import deprecated
from ads.dataset.label_encoder import DataFrameLabelEncoder
from ads.dataset.helper import is_text_data

from ads.common import logger, utils


class AutoMLProvider(ABC):
    """
    Abstract Base Class defining the structure of an AutoML solution. The solution needs to
    implement train() and get_transformer_pipeline().
    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.client = None
        self.ml_task_type = None
        self.class_names = None
        self.transformer_pipeline = None
        self.est = None

    def setup(
        self,
        X_train,
        y_train,
        ml_task_type,
        X_valid=None,
        y_valid=None,
        class_names=None,
        client=None,
    ):
        """
        Setup arguments to the AutoML instance.

        Parameters
        ----------
        X_train : DataFrame
            Training features
        y_train : DataFrame
            Training labels
        ml_task_type : One of ml_task_type.{REGRESSION,BINARY_CLASSIFICATION,
            MULTI_CLASS_CLASSIFICATION,BINARY_TEXT_CLASSIFICATION,MULTI_CLASS_TEXT_CLASSIFICATION}
        X_valid : DataFrame
            Validation features
        y_valid : DataFrame
            Validation labels
        class_names : list
            Unique values in y_train
        client : object
            Dask client instance for distributed execution
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.ml_task_type = ml_task_type
        self.client = client
        self.class_names = class_names

    @property
    def est(self):
        """
        Returns the estimator.

        The estimator can be a standard sklearn estimator or any object that implement methods from
        (BaseEstimator, RegressorMixin) for regression or (BaseEstimator, ClassifierMixin) for classification.

        Returns
        -------
        est : An instance of estimator
        """
        return self.__est

    @est.setter
    def est(self, est):
        self.__est = est

    @abstractmethod
    def train(self, **kwargs):
        """
        Calls fit on estimator.

        This method is expected to set the 'est' property.

        Parameters
        ----------
        kwargs: dict, optional
        kwargs to decide the estimator and arguments for the fit method
        """
        pass

    @abstractmethod
    def get_transformer_pipeline(self):
        """
        Returns a list of transformers representing the transformations done on data before model prediction.

        This method is optional to implement, and is used only for visualizing transformations on data using
        ADSModel#visualize_transforms().

        Returns
        -------
        transformers_list : list of transformers implementing fit and transform
        """
        pass


class BaselineModel(object):
    """
    A BaselineModel object that supports fit/predict/predict_proba/transform
    interface. Labels (y) are encoded using DataFrameLabelEncoder.
    """

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, est):
        self.est = est
        self.df_label_encoder = DataFrameLabelEncoder()

    def predict(self, X):

        """
        Runs the Baselines predict function and returns the result.

        Parameters
        ----------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be predicted on

        Returns
        -------
        List: A list of predictions performed on the input data.
        """

        X = self.transform(X)
        return self.est.predict(X)

    def predict_proba(self, X):

        """
        Runs the Baselines predict_proba function and returns the result.

        Parameters
        ----------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be predicted on

        Returns
        -------
        List: A list of probabilities of being part of a class
        """

        X = self.transform(X)
        return self.est.predict_proba(X)

    def fit(self, X, y):

        """
        Fits the baseline estimator.

        Parameters
        ----------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be predicted on
        Y: Dataframe, Series, or list-like
          A Dataframe, series, or list-like object holding the labels


        Returns
        -------
        estimator: The fitted estimator
        """

        self.est.fit(X, y)
        return self

    def transform(self, X):

        """
        Runs the Baselines transform function and returns the result.

        Parameters
        ---------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be transformed

        Returns
        -------
        Dataframe or list-like: The transformed Dataframe. Currently, no transformation is performed by the default Baseline Estimator.
        """

        return X

    def __getattr__(self, item):
        return getattr(self.est, item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __repr__(self):
        set_config()
        return str(self.est)[:-2]


class BaselineAutoMLProvider(AutoMLProvider):
    def get_transformer_pipeline(self):
        """
        Returns a list of transformers representing the transformations done on data before model prediction.

        This method is used only for visualizing transformations on data using
        ADSModel#visualize_transforms().

        Returns
        -------
        transformers_list : list of transformers implementing fit and transform
        """
        msg = "Baseline"
        return [("automl_preprocessing", AutoMLPreprocessingTransformer(msg))]

    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, est):
        """
        Generates a baseline model using the Zero Rule algorithm by default. For a classification
        predictive modeling problem where a categorical value is predicted, the Zero
        Rule algorithm predicts the class value that has the most observations in the training dataset.

        Parameters
        ----------
        est : BaselineModel
            An estimator that supports the fit/predict/predict_proba interface.
            By default, DummyClassifier/DummyRegressor are used as estimators
        """
        super(BaselineAutoMLProvider, self).__init__()
        self.est = est

    def __repr__(self):
        set_config()
        return str(self.est)[:-2]

    def train(self, **kwargs):
        self.est = self.decide_estimator(**kwargs)
        if self.est is None:
            raise ValueError(
                "Baseline model for (%s) is not supported" % self.ml_task_type
            )
        try:
            self.est.fit(self.X_train, self.y_train)
        except Exception as e:
            warning_message = f"The baseline estimator failed to fit the data. It could not evaluate {self.est} and gave the exception {e}."
            logger.warning(warning_message)

    def decide_estimator(self, **kwargs):
        """
        Decides which type of BaselineModel to generate.

        Returns
        -------
        Modell: BaselineModel
            A baseline model generated for the particular ML task being performed
        """
        if self.est is not None:
            return self.est
        else:
            if self.ml_task_type == ml_task_types.REGRESSION:
                return BaselineModel(DummyRegressor())
            elif self.ml_task_type in [
                ml_task_types.BINARY_CLASSIFICATION,
                ml_task_types.MULTI_CLASS_CLASSIFICATION,
                ml_task_types.BINARY_TEXT_CLASSIFICATION,
                ml_task_types.MULTI_CLASS_TEXT_CLASSIFICATION,
            ]:
                return BaselineModel(DummyClassifier())


# An installation of oracle labs automl is required only for this class
class OracleAutoMLProvider(AutoMLProvider, ABC):
    @deprecated(
        "2.6.7",
        details="Oracle AutoML is recommended to be directly instantiated by importing automlx package",
        raise_error=True,
    )
    def __init__(
        self, n_jobs=-1, loglevel=None, logger_override=None, model_n_jobs: int = 1
    ):
        """
        The Oracle AutoML Provider automatically provides a tuned ML pipeline that best models the given a training
        dataset and a prediction task at hand.

        Parameters
        ----------
        n_jobs : int
            Specifies the degree of parallelism for Oracle AutoML. -1 (default) means that AutoML will use all
            available cores.
        loglevel : int
            The verbosity of output for Oracle AutoML. Can be specified using the Python logging module
            (https://docs.python.org/3/library/logging.html#logging-levels).
        model_n_jobs: (optional, int). Defaults to 1.
            Specifies the model parallelism used by AutoML.
            This will be passed to the underlying model it is training.
        """
        try:
            self.automl = __import__("automl")
            self.cpuinfo = __import__("cpuinfo")
        except ModuleNotFoundError as e:
            utils._log_missing_module("automl", "ads[labs]")
            raise e
        super(OracleAutoMLProvider, self).__init__()
        if loglevel is None:
            loglevel = logging.DEBUG if ads.debug_mode else logging.ERROR

        self.automl.init(
            engine="local",
            engine_opts={"n_jobs": n_jobs, "model_n_jobs": model_n_jobs},
            logger=logger_override,
            loglevel=loglevel,
        )

    def __repr__(self):
        super(OracleAutoMLProvider, self).__repr__()

    def get_transformer_pipeline(self):
        """
        Returns a list of transformers representing the transformations done on data before model prediction.

        This method is used only for visualizing transformations on data using
        ADSModel#visualize_transforms().

        Returns
        -------
        transformers_list : list of transformers implementing fit and transform
        """
        if hasattr(self.est, "text") and not self.est.text:
            msg1 = wrap_lines(
                self.est.selected_features_names_, heading="Select features:"
            )
            return [("automl_feature_selection", AutoMLFeatureSelection(msg1))]
        else:
            msg = "Apply Tfidf Vectorization\n"
            msg += "Normalize features\n"
            msg += "Label encode target"
            return [("automl_preprocessing", AutoMLPreprocessingTransformer(msg))]

    def selected_model_name(self):
        """
        Return the name of the selected model by AutoML.
        """
        return self.est.selected_model_

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    def print_summary(
        self,
        max_rows=None,
        sort_column="Mean Validation Score",
        ranking_table_only=False,
    ):
        """
        Prints a summary of the Oracle AutoML Pipeline in the last train() call.

        Parameters
        ----------
        max_rows : int
            Number of trials to print. Pass in None to print all trials
        sort_column: string
            Column to sort results by. Must be one of ['Algorithm', '#Samples', '#Features', 'Mean Validation Score',
            'Hyperparameters', 'All Validation Scores', 'CPU Time']
        ranking_table_only: bool
            Table to be displayed. Pass in False to display the complete table.
            Pass in True to display the ranking table only.

        """
        if is_notebook():  # pragma: no cover
            logger.info(
                f"Training time was ({(time.time() - self.train_start_time):.2f} seconds.)"
            )

            if len(self.est.tuning_trials_) == 0 or len(self.est.train_shape_) == 0:
                logger.error(
                    "Unfortunately, there were no trials found, so we cannot visualize it."
                )
                return

            info = [
                ["Training Dataset size", self.X_train.shape],
                [
                    "Validation Dataset size",
                    self.X_valid.shape if self.X_valid is not None else None,
                ],
                ["CV", self.est.num_cv_folds_],
                ["Target variable", self.y_train.name],
                ["Optimization Metric", self.est.inferred_score_metric],
                ["Initial number of Features", self.est.train_shape_[1]],
                ["Selected number of Features", len(self.est.selected_features_names_)],
                ["Selected Features", self.est.selected_features_names_],
                ["Selected Algorithm", self.est.selected_model_],
                [
                    "End-to-end Elapsed Time (seconds)",
                    self.train_end_time - self.train_start_time,
                ],
                ["Selected Hyperparameters", self.est.selected_model_params_],
                ["Mean Validation Score", self.est.tuning_trials_[0][3]],
                ["AutoML n_jobs", self.est.n_jobs_],
                ["AutoML version", self.automl.__version__],
                ["Python version", sys.version],
            ]
            info_df = pd.DataFrame(info)

            # Remove the selected model and its params from the trials since it already shows up in the summary table
            all_trials_ = (
                self.est.model_selection_trials_
                + self.est.adaptive_sampling_trials_
                + self.est.feature_selection_trials_
                + self.est.tuning_trials_[1:]
            )
            col_names = [
                "Algorithm",
                "#Samples",
                "#Features",
                "Mean Validation Score",
                "Hyperparameters",
                "All Validation Scores",
                "CPU Time",
                "Memory Usage",
            ]
            if ranking_table_only:
                dropped_cols = [
                    "#Samples",
                    "#Features",
                    "All Validation Scores",
                    "CPU Time",
                ]
            else:
                dropped_cols = "All Validation Scores"
            summary_df = pd.DataFrame(all_trials_, columns=col_names).drop(
                dropped_cols, axis=1
            )
            sorted_summary_df = summary_df.sort_values(sort_column, ascending=False)
            # Add a rank column at the front
            sorted_summary_df.insert(
                0, "Rank based on Performance", np.arange(2, len(sorted_summary_df) + 2)
            )

            from IPython.core.display import display, HTML

            with pd.option_context(
                "display.max_colwidth",
                1000,
                "display.width",
                None,
                "display.precision",
                4,
            ):
                display(HTML(info_df.to_html(index=False, header=False)))
                if max_rows is None:
                    display(HTML(sorted_summary_df.to_html(index=False)))
                else:
                    display(
                        HTML(sorted_summary_df.to_html(index=False, max_rows=max_rows))
                    )

    def train(self, **kwargs):
        """
        Train the Oracle AutoML Pipeline. This looks at the training data, and
        identifies the best set of features, the best algorithm and the best
        set of hyperparameters for this data. A model is then generated, trained
        on this data and returned.

        Parameters
        ----------
        score_metric : str, callable
            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)`` or string specified as
            https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
        random_state : int
            Random seed used by AutoML
        model_list : list of str
            Models that will be evaluated by the Pipeline. Supported models:
            - Classification: AdaBoostClassifier, DecisionTreeClassifier,
            ExtraTreesClassifier, KNeighborsClassifier,
            LGBMClassifier, LinearSVC, LogisticRegression,
            RandomForestClassifier, SVC, XGBClassifier
            - Regression: AdaBoostRegressor, DecisionTreeRegressor,
            ExtraTreesRegressor, KNeighborsRegressor,
            LGBMRegressor, LinearSVR, LinearRegression, RandomForestRegressor,
            SVR, XGBRegressor
        time_budget : float, optional
            Time budget in seconds where 0 means no time budget constraint (best effort)
        min_features : int, float, list, optional (default: 1)
            Minimum number of features to keep. Acceptable values:
            - If int, 0 < min_features <= n_features
            - If float, 0 < min_features <= 1.0
            - If list, names of features to keep, for example ['a', 'b'] means keep features 'a' and 'b'

        Returns
        -------
        self : object
        """

        """Adding this part to give the correct error for situations when dataset > 10000 rows and user tries SVC or KNN"""
        if len(self.X_train) > 10000:
            if "model_list" in kwargs:
                bad_model_list = ["SVC", "KNeighborsClassifier"]
                for model in kwargs["model_list"]:
                    for item in bad_model_list:
                        if item in model:
                            raise ValueError(
                                "SVC, KNeighborsClassifier are disabled for datasets with > 10K samples"
                            )

        self.train_start_time = time.time()

        self.time_budget = kwargs.pop("time_budget", 0)  # 0 means unlimited

        self.col_types = kwargs.pop("col_types", None)

        self.est = self._decide_estimator(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.est.fit(
                self.X_train,
                self.y_train,
                X_valid=self.X_valid,
                y_valid=self.y_valid,
                time_budget=self.time_budget,
                col_types=self.col_types,
            )
        self.train_end_time = time.time()
        self.print_summary(max_rows=10)

    def print_trials(self, max_rows=None, sort_column="Mean Validation Score"):
        """
        Prints all trials executed by the Oracle AutoML Pipeline in the last train() call.

        Parameters
        ----------
        max_rows : int
            Number of trials to print. Pass in None to print all trials
        sort_column: string
            Column to sort results by. Must be one of ['Algorithm', '#Samples', '#Features', 'Mean Validation Score',
            'Hyperparameters', 'All Validation Scores', 'CPU Time']

        """
        self.est.print_trials(max_rows=max_rows, sort_column=sort_column)

    def _decide_estimator(self, **kwargs):
        """
        Decide arguments to the Oracle AutoML pipeline based on user provided
        arguments
        """
        est = None
        score_metric = None
        # Explicity define the default AutoML metrics
        if (
            self.ml_task_type == ml_task_types.BINARY_CLASSIFICATION
            or self.ml_task_type == ml_task_types.BINARY_TEXT_CLASSIFICATION
        ):
            test_model_list = ["LogisticRegression"]
        elif (
            self.ml_task_type == ml_task_types.MULTI_CLASS_CLASSIFICATION
            or self.ml_task_type == ml_task_types.MULTI_CLASS_TEXT_CLASSIFICATION
        ):
            test_model_list = ["LogisticRegression"]
        elif self.ml_task_type == ml_task_types.REGRESSION:
            test_model_list = ["LinearRegression"]
        else:
            raise ValueError("AutoML for (%s) is not supported" % self.ml_task_type)

        # Respect the user provided scoring metric if given
        if "score_metric" in kwargs:
            score_metric = kwargs.pop("score_metric")

        #
        # ***FOR TESTING PURPOSE ONLY***
        #
        # Ignore model_list for test mode
        if ads.test_mode:  # pragma: no cover
            if "model_list" in kwargs:
                _ = kwargs.pop("model_list")
            kwargs["model_list"] = test_model_list

        if (
            self.ml_task_type == ml_task_types.BINARY_CLASSIFICATION
            or self.ml_task_type == ml_task_types.MULTI_CLASS_CLASSIFICATION
        ):
            est = self.automl.Pipeline(
                task="classification", score_metric=score_metric, **kwargs
            )
        elif (
            self.ml_task_type == ml_task_types.BINARY_TEXT_CLASSIFICATION
            or self.ml_task_type == ml_task_types.MULTI_CLASS_TEXT_CLASSIFICATION
        ):
            est = self.automl.Pipeline(
                task="classification", score_metric=score_metric, **kwargs
            )
            if not self.col_types:
                if len(self.X_train.columns) == 1:
                    self.col_types = ["text"]
                elif len(self.X_train.columns) == 2:
                    self.col_types = ["text", "text"]
                else:
                    raise ValueError(
                        "We detected a text classification problem. Pass "
                        "in `col_types = [<type of column1>, <type of column2>, ...]`."
                        " Valid types are: ['categorical', 'numerical', 'text', 'datetime',"
                        " 'timedelta']."
                    )

        elif self.ml_task_type == ml_task_types.REGRESSION:
            est = self.automl.Pipeline(
                task="regression", score_metric=score_metric, **kwargs
            )
        else:
            raise ValueError("AutoML for (%s) is not supported" % self.ml_task_type)
        return est

    def selected_score_label(self):
        """
        Return the name of score_metric used in train.

        """
        score_label = self.est.score_metric
        if score_label is None:
            score_label = self.est.inferred_score_metric
        return score_label

    @runtime_dependency(module="scipy", install_from=OptionalDependency.VIZ)
    def visualize_algorithm_selection_trials(self, ylabel=None):
        """
        Plot the scores predicted by Algorithm Selection for each algorithm. The
        horizontal line shows the average score across all algorithms. Algorithms
        below the line are colored turquoise, whereas those with a score higher
        than the mean are colored teal. The orange bar shows the algorithm with
        the highest predicted score. The error bar is +/- one standard error.

        Parameters
        ----------
        ylabel : str,
            Label for the y-axis. Defaults to the scoring metric.
        """
        if ylabel is None:
            ylabel = self.selected_score_label().capitalize()
        trials = self.est.model_selection_trials_
        if not len(trials):
            _log_visualize_no_trials("algorithm selection")
            return
        fig, ax = plt.subplots(1, figsize=(6, 3))
        colors = []
        y_error = []
        mean_scores, models, cvscores = [], [], []
        for (
            algorithm,
            samples,
            features,
            mean_score,
            hyperparameters,
            all_scores,
            runtime,
            x,
        ) in trials:
            mean_scores.append(mean_score)
            models.append(algorithm)
            cvscores.append(all_scores)
        mean_scores_ser = pd.Series(mean_scores, index=models).sort_values(
            ascending=False
        )
        scores_ser = pd.Series(cvscores, index=models)
        ax.set_title("Algorithm Selection Trials")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Algorithm")
        for f in mean_scores_ser.keys():
            se = scipy.stats.sem(scores_ser[f], ddof=1)
            y_error.append(se)
            if f == "{}_AS".format(self.est.selected_model_):
                colors.append("orange")
            elif mean_scores_ser[f] >= mean_scores_ser.mean():
                colors.append("teal")
            else:
                colors.append("turquoise")
        mean_scores_ser.plot.bar(ax=ax, color=colors, edgecolor="black", zorder=1)
        ax.errorbar(
            x=mean_scores_ser.index.values,
            y=mean_scores_ser.values,
            yerr=y_error,
            fmt="none",
            capsize=4,
            color="black",
            zorder=0,
        )
        ax.axhline(y=mean_scores_ser.mean(), color="black", linewidth=0.5)
        ax.autoscale_view()
        plt.show()

    def visualize_adaptive_sampling_trials(self):
        """
        Visualize the trials for Adaptive Sampling.
        """
        trials = self.est.adaptive_sampling_trials_
        if len(trials) == 0:
            _log_visualize_no_trials("adaptive sampling")
            return
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.set_title("Adaptive Sampling ({})".format(trials[0][0]))
        ax.set_xlabel("Dataset sample size")
        ax.set_ylabel(r"Predicted model score")
        scores = [
            mean_score
            for (
                algorithm,
                samples,
                features,
                mean_score,
                hyperparameters,
                all_scores,
                runtime,
                x,
            ) in trials
        ]
        n_samples = [
            samples
            for (
                algorithm,
                samples,
                features,
                mean_score,
                hyperparameters,
                all_scores,
                runtime,
                x,
            ) in trials
        ]
        y_margin = 0.10 * (max(scores) - min(scores))
        ax.grid(color="g", linestyle="-", linewidth=0.1)
        ax.set_ylim(min(scores) - y_margin, max(scores) + y_margin)
        ax.plot(n_samples, scores, "k:", marker="s", color="teal", markersize=3)
        plt.show()

    def visualize_feature_selection_trials(self, ylabel=None):
        """
        Visualize the feature selection trials taken to arrive at optimal set of
        features. The orange line shows the optimal number of features chosen
        by Feature Selection.

        Parameters
        ----------
        ylabel : str,
            Label for the y-axis. Defaults to the scoring metric.
        """
        if ylabel is None:
            ylabel = self.selected_score_label().capitalize()
        trials = self.est.feature_selection_trials_
        if len(trials) == 0:
            _log_visualize_no_trials("feature selection")
            return
        fig, ax = plt.subplots(1, figsize=(6, 3))
        ax.set_title("Feature Selection Trials")
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(ylabel)
        scores = [
            mean_score
            for (
                algorithm,
                samples,
                features,
                mean_score,
                hyperparameters,
                all_scores,
                runtime,
                x,
            ) in trials
        ]
        n_features = [
            features
            for (
                algorithm,
                samples,
                features,
                mean_score,
                hyperparameters,
                all_scores,
                runtime,
                x,
            ) in trials
        ]
        y_margin = 0.10 * (max(scores) - min(scores))
        ax.grid(color="g", linestyle="-", linewidth=0.1)
        ax.set_ylim(min(scores) - y_margin, max(scores) + y_margin)
        ax.plot(n_features, scores, "k:", marker="s", color="teal", markersize=3)
        ax.axvline(
            x=len(self.est.selected_features_names_), color="orange", linewidth=2.0
        )
        plt.show()

    def visualize_tuning_trials(self, ylabel=None):
        """
        Visualize (plot) the hyperparamter tuning trials taken to arrive at the optimal
        hyper parameters. Each trial in the plot represents a particular
        hyperparamter combination.

        Parameters
        ----------
        ylabel : str,
            Label for the y-axis. Defaults to the scoring metric.
        """
        if ylabel is None:
            ylabel = self.selected_score_label().capitalize()
        # scores in trials are sorted decreasingly.
        # reversed(trails) : let the scores sort in increasing order from left to right.
        scores = [
            mean_score
            for (
                algorithm,
                samples,
                features,
                mean_score,
                hyperparameters,
                all_scores,
                runtime,
                x,
            ) in reversed(self.est.tuning_trials_)
            if mean_score and not np.isnan(mean_score)
        ]
        if not len(scores) > 1:
            raise RuntimeError("Insufficient tuning trials.")
        else:
            fig, ax = plt.subplots(1, figsize=(6, 3))
            ax.set_title("Hyperparameter Tuning Trials")
            ax.set_xlabel("Iteration $n$")
            ax.set_ylabel(ylabel)
            y_margin = 0.10 * (max(scores) - min(scores))
            ax.grid(color="g", linestyle="-", linewidth=0.1)
            ax.set_ylim(min(scores) - y_margin, max(scores) + y_margin)
            ax.plot(
                range(1, len(scores) + 1),
                scores,
                "k:",
                marker="s",
                color="teal",
                markersize=3,
            )
            plt.show()


class AutoMLPreprocessingTransformer(object):  # pragma: no cover
    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X):
        """
        Fits the preprocessing Transformer

        Parameters
        ----------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be predicted on

        Returns
        -------
        Self: Estimator
            The fitted estimator
        """
        return self

    def transform(self, X):
        """
        Runs the preprocessing transform function and returns the result

        Parameters
        ---------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be transformed

        Returns
        -------
        X: Dataframe or list-like
            The transformed Dataframe.
        """
        return X

    def _log_visualize_no_trials(target):
        logger.error(
            f"There are no trials. Therefore, the {target} cannot be visualized."
        )

    def __repr__(self):
        return self.msg


class AutoMLFeatureSelection(object):  # pragma: no cover
    @deprecated(
        details="Working with AutoML has moved from within ADS to working directly with the AutoMLx library. AutoMLx are preinstalled in conda pack automlx_p38_cpu_v2 and later, and can now be updated independently of ADS. AutoMLx documentation may be found at https://docs.oracle.com/en-us/iaas/tools/automlx/latest/html/multiversion/v23.1.1/index.html. Notebook examples are in Oracle's samples repository: https://github.com/oracle-samples/oci-data-science-ai-samples/tree/master/notebook_examples and a migration tutorial can be found at https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_training/automl/quick_start.html .",
        raise_error=True,
    )
    def __init__(self, msg):
        self.msg = msg

    def fit(self, X):
        """
        Fits the baseline estimator

        Parameters
        ----------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be predicted on

        Returns
        -------
        Self: Estimator
            The fitted estimator
        """
        return self

    def transform(self, X):
        """
        Runs the Baselines transform function and returns the result

        Parameters
        ---------
        X: Dataframe or list-like
          A Dataframe or list-like object holding data to be transformed

        Returns
        -------
        X: Dataframe or list-like
            The transformed Dataframe.
        """
        return X

    def __repr__(self):
        return self.msg

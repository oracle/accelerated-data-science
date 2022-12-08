#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycler import cycler
import matplotlib as mpl
import re

from ads.common import logger
import logging

mlogger = logging.getLogger("matplotlib")
mlogger.setLevel(logging.WARNING)

mpl.rcParams["image.cmap"] = "BuGn"
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=["teal", "blueviolet", "forestgreen", "peru", "y", "dodgerblue", "r"]
)

from ads.evaluations.evaluation_plot import EvaluationPlot
from ads.evaluations.statistical_metrics import ModelEvaluator
from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from ads.common.model import ADSModel
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)


class ADSEvaluator(object):
    """ADS Evaluator class. This class holds field and methods for creating and using
    ADS evaluator objects.

    Attributes
    ----------
    evaluations : list[DataFrame]
        list of evaluations.
    is_classifier : bool
        Whether the model has a non-empty `classes_` attribute indicating the presence of class labels.
    legend_labels : dict
        List of legend labels. Defaults to `None`.
    metrics_to_show : list[str]
        Names of metrics to show.
    models : list[ads.common.model.ADSModel]
        The object built using `ADSModel.from_estimator()`.
    positive_class : str or int
        The class to report metrics for binary dataset, assumed to be true.
    show_full_name :bool
        Whether to show the name of the evaluator in relevant contexts.
    test_data : ads.common.data.ADSData
        Test data to evaluate model on.
    training_data : ads.common.data.ADSData
        Training data to evaluate model.

    Positive_Class_names : list
        Class attribute listing the ways to represent positive classes

    Methods
    -------
    add_metrics(func, names)
        Adds the listed metics to the evaluator it is called on
    del_metrics(names)
        Removes listed metrics from the evaluator object it is called on
    add_models(models, show_full_name)
        Adds the listed models to the evaluator object
    del_models(names)
        Removes the listed models from the evaluator object
    show_in_notebook(plots, use_training_data, perfect, baseline, legend_labels)
        Visualize evalutation plots in the notebook
    calculate_cost(tn_weight, fp_weight, fn_weight, tp_weight, use_training_data)
        Returns a cost associated with the input weights
    """

    Positive_Class_Names = ["yes", "y", "t", "true", "1"]

    def __init__(
        self,
        test_data,
        models,
        training_data=None,
        positive_class=None,
        legend_labels=None,
        show_full_name=False,
    ):
        """Creates an ads evaluator object.

        Parameters
        ----------
        test_data : ads.common.data.ADSData instance
            Test data to evaluate model on.
            The object can be built using `ADSData.build()`.
        models : list[ads.common.model.ADSModel]
            The object can be built using `ADSModel.from_estimator()`.
            Maximum length of the list is 3
        training_data : ads.common.data.ADSData instance, optional
            Training data to evaluate model on and compare metrics against test data.
            The object can be built using `ADSData.build()`
        positive_class : str or int, optional
            The class to report metrics for binary dataset. If the target classes is True or False,
            positive_class will be set to True by default. If the dataset is multiclass or multilabel,
            this will be ignored.
        legend_labels : dict, optional
            List of legend labels. Defaults to `None`.
            If legend_labels not specified class names will be used for plots.
        show_full_name : bool, optional
            Show the name of the evaluator object. Defaults to `False`.

        Examples
        --------

        >>> train, test = ds.train_test_split()
        >>> model1 = MyModelClass1.train(train)
        >>> model2 = MyModelClass2.train(train)
        >>> evaluator = ADSEvaluator(test, [model1, model2])

        >>> legend_labels={'class_0': 'one', 'class_1': 'two', 'class_2': 'three'}
        >>> multi_evaluator = ADSEvaluator(test, models=[model1, model2],
        ...             legend_labels=legend_labels)

        """

        self.evaluations = []
        if isinstance(training_data, ADSDatasetWithTarget):
            training_data, _ = training_data.train_test_split(test_size=0.0)
        if isinstance(test_data, ADSDatasetWithTarget):
            test_data, _ = test_data.train_test_split(test_size=0.0)

        self.test_data = test_data
        self.training_data = training_data
        self.classes = []
        self.is_classifier = (
            hasattr(models[0], "classes_") and models[0].classes_ is not None
        )
        pclass = positive_class
        if self.is_classifier:
            self.classes = list(models[0].classes_)
            if len(self.classes) == 2:
                self.metrics_to_show = [
                    "accuracy",
                    "hamming_loss",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                ]
                if positive_class is None or positive_class not in self.classes:
                    pclass = next(
                        (
                            x
                            for x in list(self.classes)
                            if str(x).lower() in ADSEvaluator.Positive_Class_Names
                        ),
                        self.classes[0],
                    )
                    logger.info(
                        f"Using {pclass} as the positive class. Use `positive_class` to set this value."
                    )
            else:
                # Multi-class
                self.metrics_to_show = [
                    "accuracy",
                    "hamming_loss",
                    "precision_weighted",
                    "precision_micro",
                    "recall_weighted",
                    "recall_micro",
                    "f1_weighted",
                    "f1_micro",
                ]
        else:
            # Regression
            self.metrics_to_show = ["r2_score", "mse", "mae"]
        self.positive_class = pclass
        self.legend_labels = legend_labels

        for m in models:
            if not (isinstance(m, ADSModel)):
                try:
                    m = ADSModel.from_estimator(m.est)
                except:
                    logger.info("This model cannot be converted to an ADS Model.")
        self.evaluations = [pd.DataFrame(), pd.DataFrame()]
        self.model_names = []
        self.add_models(models, show_full_name=show_full_name)

    def add_metrics(self, funcs, names):
        """Adds the listed metrics to the evaluator object it is called on.

        Parameters
        ----------
        funcs : list
            The list of metrics to be added. This function will be provided `y_true`
            and `y_pred`, the true and predicted values for each model.
        names : list[str])
            The list of metric names corresponding to the functions.

        Returns
        -------
        Nothing

        Examples
        --------
        >>> def f1(y_true, y_pred):
        ...    return np.max(y_true - y_pred)
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> evaluator.add_metrics([f1], ['Max Residual'])
        >>> evaluator.metrics
        Output table will include the desired metric
        """

        if len(funcs) != len(names):
            raise ValueError("Could not find 1 unique name for each function")
        for name, f in zip(names, funcs):
            f_res = []
            for m in self.evaluations[1].columns:
                res = f(
                    self.evaluations[1][m]["y_true"], self.evaluations[1][m]["y_pred"]
                )
                f_res.append(res)
            pd_res = pd.DataFrame(
                [f_res], columns=self.evaluations[1].columns, index=[name]
            )
            self.evaluations[1] = pd.concat([self.evaluations[1], pd_res])
            if self.evaluations[0].shape != (0, 0):
                f_res = []
                for m in self.evaluations[0].columns:
                    res = f(
                        self.evaluations[0][m]["y_true"],
                        self.evaluations[0][m]["y_pred"],
                    )
                    f_res.append(res)
                pd_res = pd.DataFrame(
                    [f_res], columns=self.evaluations[0].columns, index=[name]
                )
                self.evaluations[0] = pd.concat([self.evaluations[0], pd_res])
            if name not in self.metrics_to_show:
                self.metrics_to_show.append(name)
        setattr(self, "train_evaluations", self.evaluations[0])
        setattr(self, "test_evaluations", self.evaluations[1])

    def del_metrics(self, names):
        """Removes the listed metrics from the evaluator object it is called on.

        Parameters
        ----------
        names : list[str]
            The list of names of metrics to be deleted. Names can be found by calling
            `evaluator.test_evaluations.index`.

        Returns
        -------
        None
            `None`

        Examples
        --------
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> evaluator.del_metrics(['mse])
        >>> evaluator.metrics
        Output table will exclude the desired metric
        """
        self.evaluations[1].drop(index=names, inplace=True)
        if self.evaluations[0].shape != (0, 0):
            self.evaluations[0].drop(index=names, inplace=True)
        self.metrics_to_show = [met for met in self.metrics_to_show if met not in names]

    def add_models(self, models, show_full_name=False):
        """Adds the listed models to the evaluator object it is called on.

        Parameters
        ----------
        models : list[ADSModel]
            The list of models to be added
        show_full_name : bool, optional
            Whether to show the full model name. Defaults to False.
            ** NOT USED **

        Returns
        -------
        Nothing

        Examples
        --------
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> evaluator.add_models("model3])
        """

        if type(models) is list:
            total_train_metrics = self.evaluations[0]
            total_test_metrics = self.evaluations[1]
            for i, m in enumerate(models):
                # if hasattr(m, 'classes_') != self.is_classifier:
                #     raise ValueError("All models should belong to same problem type.")
                # calculate evaluations on testing and training data (if X_train is not None)
                m_name = self._get_model_name(m.name)

                if self.training_data is not None:
                    y_pred, y_score = self._score_data(m, self.training_data.X)
                    train_metrics = ModelEvaluator(
                        y_true=self.training_data.y,
                        y_pred=y_pred,
                        model_name=m_name,
                        classes=m.classes_ if self.is_classifier else None,
                        y_score=y_score,
                        positive_class=self.positive_class,
                    ).get_metrics()
                    total_train_metrics = pd.concat(
                        [total_train_metrics, train_metrics], axis=1
                    )

                y_pred, y_score = self._score_data(m, self.test_data.X)
                test_metrics = ModelEvaluator(
                    y_true=self.test_data.y,
                    y_pred=y_pred,
                    model_name=m_name,
                    classes=m.classes_ if self.is_classifier else None,
                    y_score=y_score,
                    positive_class=self.positive_class,
                ).get_metrics()
                total_test_metrics = pd.concat(
                    [total_test_metrics, test_metrics], axis=1, sort=False
                )

            self.evaluations = [total_train_metrics, total_test_metrics]
            setattr(self, "train_evaluations", self.evaluations[0])
            setattr(self, "test_evaluations", self.evaluations[1])

    def del_models(self, names):
        """Removes the listed models from the evaluator object it is called on.

        Parameters
        ----------
        names : list[str]
            the list of models to be delete. Names are the model names by default, and
            assigned internally when conflicts exist. Actual names can be found using
            `evaluator.test_evaluations.columns`

        Returns
        -------
        Nothing

        Examples
        --------
        >>> model3.rename("model3")
        >>> evaluator = ADSEvaluator(test, [model1, model2, model3])
        >>> evaluator.del_models([model3])
        """

        if type(names) is list:
            self.model_names = [n for n in self.model_names if n not in names]
            self.evaluations[1].drop(columns=names, inplace=True)
            if self.evaluations[0].shape != (0, 0):
                self.evaluations[0].drop(columns=names, inplace=True)

    def show_in_notebook(
        self,
        plots=None,
        use_training_data=False,
        perfect=False,
        baseline=True,
        legend_labels=None,
    ):
        """Visualize evaluation plots.

        Parameters
        ----------
        plots : list, optional
            Filter the plots that are displayed. Defaults to None. The name of the plots are as below:

                - regression - residuals_qq, residuals_vs_fitted
                - binary classification - normalized_confusion_matrix, roc_curve, pr_curve
                - multi class classification - normalized_confusion_matrix, precision_by_label, recall_by_label, f1_by_label

        use_training_data : bool, optional
            Use training data to generate plots. Defaults to `False`.
            By default, this method uses test data to generate plots
        legend_labels : dict, optional
            Rename legend labels, that used for multi class classification plots. Defaults to None.
            legend_labels dict keys are the same as class names. legend_labels dict values are strings.
            If legend_labels not specified class names will be used for plots.

        Returns
        -------
        None
            Nothing. Outputs several evaluation plots as specified by `plots`.

        Examples
        --------

        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> evaluator.show_in_notebook()

        >>> legend_labels={'class_0': 'green', 'class_1': 'yellow', 'class_2': 'red'}
        >>> multi_evaluator = ADSEvaluator(test, [model1, model2],
        ...             legend_labels=legend_labels)
        >>> multi_evaluator.show_in_notebook(plots=["normalized_confusion_matrix",
        ...             "precision_by_label", "recall_by_label", "f1_by_label"])
        """

        # get evaluations
        if use_training_data:
            if self.training_data is None:
                raise ValueError(
                    "Training data is not provided. Re-build ADSData with training and test data"
                )
            model_evaluation = self.evaluations[0]
        else:
            model_evaluation = self.evaluations[1]
        legend_labels = (
            legend_labels if legend_labels is not None else self.legend_labels
        )
        # pass to plotting class
        EvaluationPlot.plot(
            model_evaluation, plots, len(self.classes), perfect, baseline, legend_labels
        )

    def calculate_cost(
        self, tn_weight, fp_weight, fn_weight, tp_weight, use_training_data=False
    ):
        """Returns a cost associated with the input weights.

        Parameters
        ----------
        tn_weight : int, float
            The weight to assign true negatives in calculating the cost
        fp_weight : int, float
            The weight to assign false positives in calculating the cost
        fn_weight : int, float
            The weight to assign false negatives in calculating the cost
        tp_weight : int, float
            The weight to assign true positives in calculating the cost
        use_training_data : bool, optional
            Use training data to pull the metrics. Defaults to False

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame with the cost calculated for each model

        Examples
        --------
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> costs_table = evaluator.calculate_cost(0, 10, 1000, 0)
        """

        if len(self.classes) != 2:
            raise ValueError(
                "The calculate_cost api is not supported for non-binary classification datasets."
            )
        cost_per_model = []
        if use_training_data:
            if self.training_data is None:
                raise ValueError(
                    "Training data is not provided. Re-build ADSData with training and test data."
                )
            ev = self.evaluations[0]
        else:
            ev = self.evaluations[1]
        list_of_model = ev.columns
        for m in list_of_model:
            tn, fp, fn, tp = ev[m]["raw_confusion_matrix"].ravel()
            cost_per_model.append(
                tn * tn_weight + fp * fp_weight + fn * fn_weight + tp * tp_weight
            )
        cost_df = pd.DataFrame({"model": list_of_model, "cost": cost_per_model})
        return cost_df

    class EvaluationMetrics(object):
        """Class holding evaluation metrics.

        Attributes
        ----------
        ev_test : list
            evaluation test metrics
        ev_train : list
            evaluation training metrics
        use_training : bool
            use training data
        less_is_more : list
            metrics list

        Methods
        -------
        show_in_notebook()
            Shows visualization metrics as a color coded table

        """

        DEFAULT_LABELS_MAP = {
            "accuracy": "Accuracy",
            "hamming_loss": "Hamming distance",
            "kappa_score_": "Cohen's kappa coefficient",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "auc": "ROC AUC",
        }

        def __init__(
            self, ev_test, ev_train, use_training=False, less_is_more=None, precision=4
        ):
            self.ev_test = ev_test
            self.ev_train = ev_train
            self.use_training = use_training
            self.precision = precision
            if isinstance(less_is_more, list):
                self.less_is_more = [
                    "hamming_loss",
                    "hinge_loss",
                    "mse",
                    "mae",
                ] + less_is_more
            else:
                self.less_is_more = ["hamming_loss", "hinge_loss", "mse", "mae"]

        def __repr__(self):
            self.show_in_notebook()
            return ""

        @property
        def precision(self):
            return self._precision

        @precision.setter
        def precision(self, value):
            """
            Set precision to @property of the class.
            """
            if not isinstance(value, int):
                if not (isinstance(value, float) and value.is_integer()):
                    raise TypeError("'value' must be integer")
                value = int(value)
            if value < 0:
                raise ValueError("'value' must be non-negative")
            self._precision = value

        def show_in_notebook(self, labels=DEFAULT_LABELS_MAP):
            """
            Visualizes evaluation metrics as a color coded table.

            Parameters
            ----------
            labels : dictionary
                map printing specific labels for metrics display

            Returns
            -------
            Nothing
            """

            def highlight_max(s):
                """Highlight the maximum in a Series yellow.

                Parameters
                ----------
                s : series object
                    the series being evaluated

                Returns
                -------
                list
                    containing background color data or empty if not max
                """
                if s.name not in self.less_is_more:
                    is_max = s == s.max()
                else:
                    is_max = s == s.min()
                return ["background-color: lightgreen" if v else "" for v in is_max]

            table_styles = [
                dict(props=[("text-align", "right")]),
                dict(selector="caption", props=[("caption-side", "top")]),
            ]

            def _pretty_label(df, labels, copy=False):
                """
                Output specified labels in proper format.
                If the labels are provided in then used them. Otherwise, use default.

                Parameters
                ----------
                labels : dictionary
                    map printing specific labels for metrics display

                Returns
                -------
                dataframe
                    dataframe with index names modified according to input labels
                """
                if copy:
                    df = df.copy()
                for k, v in labels.items():
                    df.rename(index={k: v}, inplace=True)
                return df

            @runtime_dependency(
                module="IPython", install_from=OptionalDependency.NOTEBOOK
            )
            @runtime_dependency(
                module="ipywidgets",
                object="HTML",
                install_from=OptionalDependency.NOTEBOOK,
            )
            def _display_metrics(df, data_name, labels, precision):
                """
                display metrics on web page

                Parameters
                ----------
                df : dataframe
                    metrics in dataframe format
                data_name : string
                    name of data given metrics df describe
                labels : dictionary
                    map printing specific labels for metrics display
                precision : int
                    precision for metrics display

                Returns
                -------
                Nothing
                """
                from IPython.core.display import display, HTML

                display(
                    HTML(
                        _pretty_label(df, labels)
                        .style.apply(highlight_max, axis=1)
                        .format(precision=precision)
                        .set_properties(**{"text-align": "center"})
                        .set_table_attributes("class=table")
                        .set_caption(
                            '<div align="left"><b style="font-size:20px;">'
                            + "Evaluation Metrics ("
                            + data_name
                            + "):</b></div>"
                        )
                        .to_html()
                    )
                )

            _display_metrics(self.ev_test, "testing data", labels, self.precision)
            if self.use_training:
                _display_metrics(self.ev_train, "training data", labels, self.precision)

    @property
    def raw_metrics(self, metrics=None, use_training_data=False):
        """Returns the raw metric numbers

        Parameters
        ----------
        metrics : list, optional
            Request metrics to pull. Defaults to all.
        use_training_data : bool, optional
            Use training data to pull metrics. Defaults to False

        Returns
        -------
        dict
            The requested raw metrics for each model. If `metrics` is `None` return all.

        Examples
        --------
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> raw_metrics_dictionary = evaluator.raw_metrics()
        """

        [train_met, test_met] = self.evaluations
        test_d = test_met.to_dict()
        if use_training_data and train_met is not None:
            train_d = train_met.add_suffix("_train").to_dict()
            test_d.update(train_d)
        for m, data in test_d.items():
            ret = dict()
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    ret[k] = v.tolist()
                else:
                    ret[k] = v
            test_d[m] = ret
        return test_d

    @property
    def metrics(self):
        """Returns evaluation metrics

        Returns
        -------
        metrics
            HTML representation of a table comparing relevant metrics.

        Examples
        --------
        >>> evaluator = ADSEvaluator(test, [model1, model2])
        >>> evaluator.metrics
        Outputs table displaying metrics.
        """

        ev_test = self.evaluations[1].loc[self.metrics_to_show]
        use_training = self.evaluations[0].shape != (0, 0)
        ev_train = (
            self.evaluations[0].loc[self.metrics_to_show] if use_training else None
        )
        return ADSEvaluator.EvaluationMetrics(ev_test, ev_train, use_training)

    """
    Internal methods
    """

    def _get_model_name(self, name, show_full_name=False):
        name_edit = re.sub(r" ?\([^)]+\)", "", name)
        ## if name only has '(' without ')', the code above wouldnt remove the argument followed by '('.
        if "(" in name_edit and not show_full_name:
            name_edit = name.split("(")[0]
            logger.info("Use `show_full_name=True` to show the full model name.")
        if name_edit in self.model_names:
            name_edit += "_1"
            num_tries = 1
            while name_edit in self.model_names:
                num_tries += 1
                name_edit = name_edit[:-1] + str(num_tries)
            if num_tries == 1:
                logger.info(
                    f"The name '{name_edit[:-2]}' is used by multiple models. "
                    f"Use the `rename()` method to change the name."
                )
        self.model_names.append(name_edit)
        return name_edit

    def _score_data(self, est, X):
        y_pred = est.predict(X)
        y_score = None

        # we will compute y_score only for binary classification cases because only for binary classification can
        # we use it for ROC Curves and AUC etc
        if self.is_classifier and hasattr(est.est, "predict_proba"):
            if len(est.classes_) == 2:
                #  positive label index is assumed to be 0 if the ADSModel does not have a positive class defined
                positive_class_index = 0
                # For prediction probability, we only consider the positive class.
                if self.positive_class is not None:
                    if self.positive_class not in list(est.classes_):
                        raise ValueError(
                            "Invalid positive class '%s' for model %s. Positive class should be one of %s."
                            % (
                                self.positive_class,
                                est.est.__class__.__name__,
                                list(est.classes_),
                            )
                        )
                    positive_class_index = list(est.classes_).index(self.positive_class)
                y_score = est.predict_proba(X)[:, positive_class_index]
            else:
                y_score = est.predict_proba(X)
        return y_pred, y_score

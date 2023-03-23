#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from cycler import cycler
import logging
import matplotlib as mpl
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
import tempfile
from typing import List, Any

logging.getLogger("matplotlib").setLevel(logging.WARNING)
mpl.rcParams["image.cmap"] = "BuGn"
mpl.rcParams["axes.prop_cycle"] = cycler(
    color=["teal", "blueviolet", "forestgreen", "peru", "y", "dodgerblue", "r"]
)

from ads.common.data import ADSData
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.decorator.deprecate import deprecated
from ads.common import logger
from ads.common.model import ADSModel
from ads.common.model_metadata import UseCaseType
from ads.dataset.dataset_with_target import ADSDatasetWithTarget
from ads.evaluations.evaluation_plot import EvaluationPlot
from ads.evaluations.statistical_metrics import (
    ModelEvaluator,
    DEFAULT_BIN_CLASS_METRICS,
    DEFAULT_MULTI_CLASS_METRICS,
    DEFAULT_REG_METRICS,
    DEFAULT_BIN_CLASS_LABELS_MAP,
    DEFAULT_MULTI_CLASS_LABELS_MAP,
    DEFAULT_REG_LABELS_MAP,
)
from ads.model.generic_model import GenericModel, VERIFY_STATUS_NAME

METRICS_TO_MINIMIZE = ["hamming_loss", "hinge_loss", "mse", "mae"]
POSITIVE_CLASS_NAMES = ["yes", "y", "t", "true", "1"]


class Evaluator(object):
    """
    BETA FEATURE
    Evaluator is the new and preferred way to evaluate a model of list of models.
    It contains a superset of the features of the soon-to-be-deprecated ADSEvaluator.

    Methods
    -------
    display()
        Shows all plots and metrics within the jupyter notebook.
    html()
        Returns the raw string of the html report
    save(filename)
        Saves the html report to the provided file location.
    add_model(model)
        Adds a model to the existsing report. See documentation for more details.
    add_metric(metric_fn)
        Adds a metric to the existsing report. See documentation for more details.
    add_plot(plotting_fn)
        Adds a plot to the existing report. See documentation for more details.

    """

    def __init__(
        self,
        models: List[GenericModel],
        X: ArrayLike,
        y: ArrayLike,
        y_preds: List[ArrayLike] = None,
        y_scores: List[ArrayLike] = None,
        X_train: ArrayLike = None,
        y_train: ArrayLike = None,
        classes: List = None,
        positive_class: str = None,
        legend_labels: dict = None,
        use_case_type: UseCaseType = None,
    ):
        """Creates an ads evaluator object.

        Parameters
        ----------
        models : ads.model.GenericModel instance
            Test data to evaluate model on.
            The object can be built using from one of the framworks supported in `ads.model.framework`
        X : DataFrame-like
            The data used to make a prediction.
            Can be set to None if `y_preds` is given. (And `y_scores` for more thorough analysis).
        y : array-like
            The true values corresponding to the input data
        y_preds : list of array-like, optional
            The predictions from each model in the same order as the models
        y_scores : list of array-like, optional
            The predict_probas from each model in the same order as the models
        X_train : DataFrame-like, optional
            The data used to train the model
        y_train : array-like, optional
            The true values corresponding to the input training data
        positive_class : str or int, optional
            The class to report metrics for binary dataset. If the target classes is True or False,
            positive_class will be set to True by default. If the dataset is multiclass or multilabel,
            this will be ignored.
        legend_labels : dict, optional
            List of legend labels. Defaults to `None`.
            If legend_labels not specified class names will be used for plots.
        classes : List or None, optional
            A List of the possible labels for y, when evaluating a classification use case
        use_case_type : str, optional
            The type of problem this model is solving. This can be set during `prepare()`.
            Examples: "binary_classification", "regression", "multinomial_classification"
            Full list of supported types can be found here: `ads.common.model_metadata.UseCaseType`

        Examples
        --------
        >>> import tempfile
        >>> from ads.evaluations.evaluator import Evaluator
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from ads.model.framework.sklearn_model import SklearnModel
        >>> from ads.common.model_metadata import UseCaseType
        >>>
        >>> X, y = make_classification(n_samples=1000)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        >>> est = DecisionTreeClassifier().fit(X_train, y_train)
        >>> model = SklearnModel(estimator=est, artifact_dir=tempfile.mkdtemp())
        >>> model.prepare(
                inference_conda_env="generalml_p38_cpu_v1",
                training_conda_env="generalml_p38_cpu_v1",
                X_sample=X_test,
                y_sample=y_test,
                use_case_type=UseCaseType.BINARY_CLASSIFICATION,
            )
        >>> report = Evaluator([my_model], X=X_test, y=y_test)
        >>> report.display()

        """
        self._verify_models(models)
        self.X, self.y, self.X_train, self.y_train = X, y, X_train, y_train
        self.legend_labels = legend_labels
        self.positive_class = positive_class

        self._determine_problem_type(models, use_case_type)
        self._determine_classes(classes)

        self.model_names = []
        self.evaluation = pd.DataFrame()
        self.add_models(models, y_preds=y_preds, y_scores=y_scores)

    def _verify_models(self, models):
        assert isinstance(
            models, list
        ), f"The `models` argument must be a list of models, instead got: {models}"
        for m in models:
            if not isinstance(m, GenericModel):
                raise ValueError(
                    f"Please register and prepare model {m} with ads. More information here: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/introduction.html#register"
                )
            sum_stat = m.summary_status().reset_index()
            if (
                sum_stat.loc[sum_stat["Step"] == VERIFY_STATUS_NAME, "Status"]
                == "Not Available"
            ).any():
                raise ValueError(
                    f"Model {m} has not been prepared, and `verify` cannot be run (including the pre and post processing from the score.py). This may cause issues. Prepare the model in accordance with the documentation: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/model_artifact.html#prepare-the-model-artifact"
                )

    def _determine_problem_type(self, models, use_case_type):
        if use_case_type is not None:
            self.problem_type = use_case_type
        problem_type = models[0].metadata_taxonomy["UseCaseType"].value
        if problem_type is not None:
            for m in models:
                assert (
                    problem_type == m.metadata_taxonomy["UseCaseType"].value
                ), f"Cannot compare models of different Use Case types. The first model is of type {problem_type}, while model: {m} is of Use Case type: {m.metadata_taxonomy['UseCaseType'].value}"
            self.problem_type = problem_type
        else:
            if not models[0].schema_output.keys:
                raise ValueError(
                    f"The Use Case Type of this model, {models[0]}, is ambigious. Please re-run Evaluator with `use_case_type` set to a valid type (full list found here: ads.common.model_metadata.UseCaseType). To avoid setting this in the future, set the `use_case_type` when preparing the model. Or update your model's use_case_type attribute here: `model.metadata_taxonomy['UseCaseType'].value` More information here: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/model_metadata.html#taxonomy-metadata"
                )
            logger.warn(
                f"The Use Case Type of this model, {models[0]}, is ambigious. Please set the `model.metadata_taxonomy['UseCaseType'].value` attribute to one of the options in ads.common.model_metadata.UseCaseType"
            )

            output_col = models[0].schema_output.keys[0]
            if models[0].schema_output[output_col].feature_type != "Continuous":
                if len(np.unique(self.y)) == 2:
                    self.problem_type = UseCaseType.BINARY_CLASSIFICATION
                else:
                    self.problem_type = UseCaseType.MULTINOMIAL_CLASSIFICATION
            else:
                self.problem_type = UseCaseType.REGRESSION
            logger.info(f"Set Use Case Type to: {self.problem_type}")

    def _determine_classes(self, classes):
        if self.problem_type in [UseCaseType.REGRESSION]:
            self.classes = []
            self.metrics_to_show = DEFAULT_REG_METRICS
            self.is_classifier = False
        else:
            self.is_classifier = True
            self.classes = (
                classes or np.unique(self.y_train)
                if self.y_train is not None
                else np.unique(self.y)
            )
            self.num_classes = len(self.classes)
            if len(self.classes) == 2:
                self.metrics_to_show = DEFAULT_BIN_CLASS_METRICS
                if (
                    self.positive_class is None
                    or self.positive_class not in self.classes
                ):
                    self.positive_class = next(
                        (
                            x
                            for x in list(self.classes)
                            if str(x).lower() in POSITIVE_CLASS_NAMES
                        ),
                        self.classes[0],
                    )
                    logger.info(
                        f"Using {self.positive_class} as the positive class. Use `positive_class` to set this value."
                    )
            else:
                self.metrics_to_show = DEFAULT_MULTI_CLASS_METRICS

    def _get_model_name(self, model):
        name = str(model.algorithm) + "_" + str(model.framework)
        name_edit = re.sub(r" ?\([^)]+\)", "", name)
        if name_edit in self.model_names:
            name_edit += "_1"
            num_tries = 1
            while name_edit in self.model_names:
                num_tries += 1
                name_edit = name_edit[:-1] + str(num_tries)
        self.model_names.append(name_edit)
        return name_edit

    def _score_data(self, model, X):
        y_pred = model.verify(X)["prediction"]

        y_score = None
        # we will compute y_score only for binary classification cases because only for binary classification can
        # we use it for ROC Curves and AUC
        if self.is_classifier and hasattr(model.estimator, "predict_proba"):
            if len(self.classes) == 2:
                #  positive label index is assumed to be 0 if the ADSModel does not have a positive class defined
                positive_class_index = 0
                # For prediction probability, we only consider the positive class.
                if self.positive_class is not None:
                    if self.positive_class not in list(self.classes):
                        raise ValueError(
                            "Invalid positive class '%s' for model %s. Positive class should be one of %s."
                            % (
                                self.positive_class,
                                model.estimator.__class__.__name__,
                                list(self.classes),
                            )
                        )
                    positive_class_index = list(self.classes).index(self.positive_class)
                y_score = model.estimator.predict_proba(X)[:, positive_class_index]
            else:
                y_score = model.estimator.predict_proba(X)
        return y_pred, y_score

    def add_models(
        self,
        models: List[GenericModel],
        y_preds: List[Any] = None,
        y_scores: List[Any] = None,
    ):
        """Add a model to an existing Evaluator to avoid re-calculating the values.

        Parameters
        ----------
        models : List[ads.model.GenericModel]
            Test data to evaluate model on.
            The object can be built using from one of the framworks supported in `ads.model.framework`
        y_preds : list of array-like, optional
            The predictions from each model in the same order as the models
        y_scores : list of array-like, optional
            The predict_probas from each model in the same order as the models

        Returns
        -------
        self

        Examples
        --------
        >>> evaluator = Evaluator(models = [model1, model2], X=X, y=y)
        >>> evaluator.add_models(models = [model3])
        """

        assert isinstance(models, List), "The `models` parameter must be of type list."
        if self.is_classifier:
            self._le = LabelEncoder().fit(self.y)
        for i, m in enumerate(models):
            m_name = self._get_model_name(m)

            if y_preds is None:
                y_pred, y_score = self._score_data(m, self.X)
            else:
                y_pred = y_preds[i]
                y_score = y_scores[i] if isinstance(y_scores, list) else None
            if self.is_classifier:
                y_true, y_pred = self._le.transform(self.y), self._le.transform(y_pred)
                classes = self._le.transform(self.classes)
                pos_class = None
                if len(self.classes) == 2:
                    pos_class = self._le.transform([self.positive_class])[0]
            else:
                y_true, y_pred, classes, pos_class = self.y, y_pred, None, None
            new_model_metrics = ModelEvaluator(
                y_true=y_true,
                y_pred=y_pred,
                model_name=m_name,
                classes=classes,
                y_score=y_score,
                positive_class=pos_class,
            ).get_metrics()
            self.evaluation = pd.concat(
                [self.evaluation, new_model_metrics], axis=1, sort=False
            )
        return self

    @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
    @runtime_dependency(
        module="ipywidgets",
        object="HTML",
        install_from=OptionalDependency.NOTEBOOK,
    )
    def display(
        self,
        plots=None,
        perfect=False,
        baseline=True,
        legend_labels=None,
        precision=4,
        metrics_labels=None,
    ):
        """Visualize evaluation report.

        Parameters
        ----------
        plots : list, optional
            Filter the plots that are displayed. Defaults to None. The name of the plots are as below:

                - regression - residuals_qq, residuals_vs_fitted
                - binary classification - normalized_confusion_matrix, roc_curve, pr_curve
                - multi class classification - normalized_confusion_matrix, precision_by_label, recall_by_label, f1_by_label
        perfect: bool, optional (default False)
            If True, will show how a perfect classifier would perform.
        baseline: bool, optional (default True)
            If True, will show how a random classifier would perform.
        legend_labels : dict, optional
            Rename legend labels, that used for multi class classification plots. Defaults to None.
            legend_labels dict keys are the same as class names. legend_labels dict values are strings.
            If legend_labels not specified class names will be used for plots.
        precision: int, optional (default 4)
            The number of decimal points to show for each score/loss value
        metrics_labels: List, optional
            The metrics that should be included in the html table.

        Returns
        -------
        None
            Nothing. Outputs several evaluation plots as specified by `plots`.

        Examples
        --------

        >>> evaluator = Evaluator(models=[model1, model2], X=X, y=y)
        >>> evaluator.display()

        >>> legend_labels={'class_0': 'green', 'class_1': 'yellow', 'class_2': 'red'}
        >>> multi_evaluator = Evaluator(models=[model1, model2], X=X, y=y, legend_labels=legend_labels)
        >>> multi_evaluator.display(plots=["normalized_confusion_matrix",
        ...             "precision_by_label", "recall_by_label", "f1_by_label"])
        """
        from IPython.core.display import display, HTML

        legend_labels = (
            legend_labels if legend_labels is not None else self.legend_labels
        )
        if legend_labels is None and self.is_classifier:
            legend_labels = dict(
                zip([str(x) for x in self._le.transform(self.classes)], self.classes)
            )
        # pass to plotting class
        self._get_plots_html(
            plots=plots, perfect=perfect, baseline=baseline, legend_labels=legend_labels
        )
        display(
            HTML(self._get_metrics_html(precision=precision, labels=metrics_labels))
        )

    def html(
        self,
        plots=None,
        perfect=False,
        baseline=True,
        legend_labels=None,
        precision=4,
        metrics_labels=None,
    ):
        """Get raw HTML report.

        Parameters
        ----------
        plots : list, optional
            Filter the plots that are displayed. Defaults to None. The name of the plots are as below:

                - regression - residuals_qq, residuals_vs_fitted
                - binary classification - normalized_confusion_matrix, roc_curve, pr_curve
                - multi class classification - normalized_confusion_matrix, precision_by_label, recall_by_label, f1_by_label
        perfect: bool, optional (default False)
            If True, will show how a perfect classifier would perform.
        baseline: bool, optional (default True)
            If True, will show how a random classifier would perform.
        legend_labels : dict, optional
            Rename legend labels, that used for multi class classification plots. Defaults to None.
            legend_labels dict keys are the same as class names. legend_labels dict values are strings.
            If legend_labels not specified class names will be used for plots.
        precision: int, optional (default 4)
            The number of decimal points to show for each score/loss value
        metrics_labels: List, optional
            The metrics that should be included in the html table.

        Returns
        -------
        None
            Nothing. Outputs several evaluation plots as specified by `plots`.

        Examples
        --------

        >>> evaluator = Evaluator(models=[model1, model2], X=X, y=y)
        >>> raw_html = evaluator.html()
        """
        html_plots = self._get_plots_html(
            plots=plots, perfect=perfect, baseline=baseline, legend_labels=legend_labels
        )
        html_metrics = self._get_metrics_html(
            precision=precision, labels=metrics_labels
        )
        html_raw = (
            "<h1>Evaluation Report</h1> \
                    <h2>Evaluation Plots</h2> "
            + " \
                    ".join(
                html_plots
            )
            + f"<h2>Evaluation Metrics</h2>  \
                    <p> {html_metrics} </p>"
        )
        return html_raw

    def save(self, filename: str, **kwargs):
        """Save HTML report.

        Parameters
        ----------
        filename: str
            The name and path of where to save the html report.
        plots : list, optional
            Filter the plots that are displayed. Defaults to None. The name of the plots are as below:

                - regression - residuals_qq, residuals_vs_fitted
                - binary classification - normalized_confusion_matrix, roc_curve, pr_curve
                - multi class classification - normalized_confusion_matrix, precision_by_label, recall_by_label, f1_by_label
        perfect: bool, optional (default False)
            If True, will show how a perfect classifier would perform.
        baseline: bool, optional (default True)
            If True, will show how a random classifier would perform.
        legend_labels : dict, optional
            Rename legend labels, that used for multi class classification plots. Defaults to None.
            legend_labels dict keys are the same as class names. legend_labels dict values are strings.
            If legend_labels not specified class names will be used for plots.
        precision: int, optional (default 4)
            The number of decimal points to show for each score/loss value
        metrics_labels: List, optional
            The metrics that should be included in the html table.

        Returns
        -------
        None
            Nothing. Outputs several evaluation plots as specified by `plots`.

        Examples
        --------

        >>> evaluator = Evaluator(models=[model1, model2], X=X, y=y)
        >>> evaluator.save("report.html")
        """
        raw_html = self.html(**kwargs)
        with open(filename, "w") as f:
            f.write(raw_html)

    def _get_plots_html(
        self,
        plots=None,
        perfect=False,
        baseline=True,
        legend_labels=None,
    ):
        return EvaluationPlot.plot(
            self.evaluation, plots, len(self.classes), perfect, baseline, legend_labels
        )

    def _get_metrics_html(self, precision=4, labels=None):
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
            if s.name not in METRICS_TO_MINIMIZE:
                is_max = s == s.max()
            else:
                is_max = s == s.min()
            return ["background-color: lightgreen" if v else "" for v in is_max]

        def _pretty_label(df, labels, copy=True):
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
            df_display = df.loc[list(labels.keys())]

            if copy:
                df_display = df_display.copy()
            for k, v in labels.items():
                df_display.rename(index={k: v}, inplace=True)
            return df_display

        if labels is None:
            if self.is_classifier:
                if len(self.classes) == 2:
                    labels = DEFAULT_BIN_CLASS_LABELS_MAP
                else:
                    labels = DEFAULT_MULTI_CLASS_LABELS_MAP
            else:
                labels = DEFAULT_REG_LABELS_MAP
        html_raw = (
            _pretty_label(self.evaluation, labels)
            .style.apply(highlight_max, axis=1)
            .format(precision=precision)
            .set_properties(**{"text-align": "center"})
            .set_table_attributes("class=table")
            .set_caption(
                '<div align="left"><b style="font-size:20px;">'
                + "Evaluation Metrics:</b></div>"
            )
            .to_html()
        )
        return html_raw


class ADSEvaluator(object):
    """ADS Evaluator class. This class holds field and methods for creating and using
    ADS evaluator objects.

    Attributes
    ----------
    evaluations : list[DataFrame]
        list of evaluations.
    is_classifier : bool
        Whether the dataset looks like a classification problem (versus regression).
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
        classes=None,
        classification_threshold=50,
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
        classes : List or None, optional
            A List of the possible labels for y, when evaluating a classification use case
        classification_threshold : int, defaults to 50
            The maximum number of unique values that y must have to qualify as classification.
            If this threshold is exceeded, Evaluator assumes the model is regression.

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
        if any(isinstance(m, ADSModel) for m in models):
            logger.warn(
                f"ADSModel is being deprecated. Users should instead use GenericModel or one of its subclasses. More information here: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/model_registration/introduction.html#register"
            )
        self.evaluations = []
        if isinstance(training_data, ADSDatasetWithTarget):
            training_data, _ = training_data.train_test_split(test_size=0.0)
        if isinstance(test_data, ADSDatasetWithTarget):
            test_data, _ = test_data.train_test_split(test_size=0.0)

        if not isinstance(test_data, ADSData):
            raise ValueError(
                "Expected test_data to be of type ADSData. More information here: https://accelerated-data-science.readthedocs.io/en/latest/ads.common.html#ads.common.data.ADSData"
            )
        if training_data and not isinstance(training_data, ADSData):
            raise ValueError(
                "Expected training_data to be of type ADSData. More information here: https://accelerated-data-science.readthedocs.io/en/latest/ads.common.html#ads.common.data.ADSData"
            )
        assert isinstance(
            models, list
        ), "The `models` argument should be a list of GenericModels. More information here: https://accelerated-data-science.readthedocs.io/en/latest/ads.common.html#ads.common.data.ADSData"

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

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import print_function, absolute_import, division

import base64
from io import BytesIO
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
from ads.common import logger
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
import itertools
import pandas as pd

MAX_TITLE_LEN = 20
MAX_LEGEND_LEN = 10
MAX_PLOTS_PER_ROW = 2
# Maximum class number evaluation plotting supporting for multiclass problems
MAX_PLOTTING_CLASSES = 10
# Maximum characters in class label able to be shown without being truncated
MAX_CHARACTERS_LEN = 13


def _fig_to_html(fig):
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format="png")
    encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")

    html = "<img src='data:image/png;base64,{}'>".format(encoded)
    return html


class EvaluationPlot:
    """EvaluationPlot holds data and methods for plots and it used to output them

    Attributes
    ----------
    baseline (bool):
        whether to plot the null model or zero information model
    baseline_kwargs (dict):
        keyword arguments for the baseline plot
    color_wheel (dict):
        color information used by the plot
    font_sz (dict):
        dictionary of plot methods
    perfect (bool):
        determines whether a "perfect" classifier curve is displayed
    perfect_kwargs (dict):
        parameters for the perfect classifier for precision/recall curves
    prob_type (str):
        model type, i.e. classification or regression

    Methods
    -------
    get_legend_labels(legend_labels)
        Renders the legend labels on the plot
    plot(evaluation, plots, num_classes, perfect, baseline, legend_labels)
        Generates the evalation plot
    """

    # dict of plot methods
    font_sz = {
        "xl": 16,  # Plot type title
        "l": 14,  # Individual plot title (name of model)
        "m": 12,  # Axis titles
        "s": 10,  # Axis labels
        "xs": 8,
    }  # test within the plot

    baseline_kwargs = {"ls": "--", "c": ".2"}  # lw = ??
    perfect_kwargs = {"label": "Perfect Classifier", "ls": "--", "color": "gold"}
    perfect = None
    baseline = None
    prob_type = None
    color_wheel = ["teal", "blueviolet", "forestgreen", "peru", "y", "dodgerblue", "r"]

    _pretty_titles_map = {
        "normalized_confusion_matrix": "Normalized Confusion Matrix",
        "lift_chart": "Lift Chart",
        "gain_chart": "Gain Chart",
        "ks_statistics": "KS Statistics",
        "residuals_qq": "Residuals Q-Q Plot",
        "residuals_vs_predicted": "Residuals vs Predicted",
        "residuals_vs_observed": "Residuals vs Observed",
        "observed_vs_predicted": "Observed vs Predicted",
        "precision_by_label": "Precision by Label",
        "recall_by_label": "Recall by Label",
        "f1_by_label": "F1 by Label",
        "jaccard_by_label": "Jaccard by Label",
        "pr_curve": "PR Curve",
        "roc_curve": "ROC Curve",
        "pr_and_roc_curve": "PR Curve, ROC Curve",
        "lift_and_gain_chart": "Lift Chart, Gain Chart",
    }

    _ugly_titles_map = {v: k for k, v in _pretty_titles_map.items()}

    double_overlay_plots = ["pr_and_roc_curve", "lift_and_gain_chart"]
    single_overlay_plots = ["lift_chart", "gain_chart", "roc_curve", "pr_curve"]

    _bin_plots = [
        "pr_curve",
        "roc_curve",
        "lift_chart",
        "gain_chart",
        "normalized_confusion_matrix",
    ]
    _multi_plots = [
        "normalized_confusion_matrix",
        "roc_curve",
        "pr_curve",
        "precision_by_label",
        "recall_by_label",
        "f1_by_label",
        "jaccard_by_label",
    ]
    _reg_plots = [
        "observed_vs_predicted",
        "residuals_qq",
        "residuals_vs_predicted",
        "residuals_vs_observed",
    ]

    # list of detailed descriptions of each plot type for every classification type, can be extended when adding more metrics
    _bin_plots_details = """
    In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value)
    is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the
    fraction of relevant instances that have been retrieved over the total amount of relevant instances. \n
    A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a
    binary classifier system as its discrimination threshold is varied. \n
    In data mining and association rule learning, lift is a measure of the performance of a targeting model (association rule)
    at predicting or classifying cases as having an enhanced response (with respect to the population as a whole), measured
    against a random choice targeting model. \n
    A gain graph is a graph whose edges are labelled 'invertibly', or 'orientably', by elements of a group G. \n
    In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known
    as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a
    supervised learning one (in unsupervised learning it is usually called a matching matrix).
    """
    # Removed for now:
    # Kuiper 's test (ks_statistics) is used in statistics to test that whether a given distribution, or family of
    # distributions, is contradicted by evidence from a sample of data. \n

    _multi_plots_details = """
    In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known
    as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a
    supervised learning one (in unsupervised learning it is usually called a matching matrix). \n
    A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the diagnostic ability of a
    binary classifier system as its discrimination threshold is varied. \n
    In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value)
    is the fraction of relevant instances among the retrieved instances, while recall (also known as sensitivity) is the
    fraction of relevant instances that have been retrieved over the total amount of relevant instances. \n
    In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy.
    It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results
    divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided
    by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic mean
    of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. \n
    The Jaccard index, also known as Intersection over Union and the Jaccard similarity coefficient, is a statistic used for gauging
    the similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets, and is defined
    as the size of the intersection divided by the size of the union of the sample sets
    """
    _reg_plots_details = """
    In statistics, a Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions
    by plotting their quantiles against each other. \n
    In statistics and optimization, errors and residuals are two closely related and easily confused measures of the deviation of an observed value
    of an element of a statistical sample from its "theoretical value". The error (or disturbance) of an observed value is the deviation of the
    observed value from the (unobservable) true value of a quantity of interest (for example, a population mean), and the residual of an observed
    value is the difference between the observed value and the estimated value of the quantity of interest (for example, a sample mean).
    """

    @classmethod
    def _get_formatted_title(cls, title, max_len=MAX_TITLE_LEN):
        return title if len(title) < max_len + 3 else title[:max_len] + "..."

    @classmethod
    def get_legend_labels(cls, legend_labels):
        """Gets the legend labels, resolves any conflicts such as length, and renders
        the labels for the plot

        Parameters
        ----------
        legend_labels (dict):
            key/value dictionary containing legend label data

        Returns
        -------
        Nothing

        Examples
        --------

        EvaluationPlot.get_legend_labels({'class_0': 'green', 'class_1': 'yellow', 'class_2': 'red'})
        """

        @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
        @runtime_dependency(
            module="ipywidgets", object="HTML", install_from=OptionalDependency.NOTEBOOK
        )
        def render_legend_labels(label_dict):
            encodings = pd.DataFrame(
                pd.Series(label_dict, index=label_dict.keys()),
                columns=["Shortened labels"],
            )
            from IPython.core.display import display, HTML

            display(
                HTML(
                    encodings.style.format(precision=4)
                    .set_properties(**{"text-align": "center"})
                    .set_table_styles(
                        [dict(selector="", props=[("text-align", "center")])]
                    )
                    .set_table_attributes("class=table")
                    .set_caption(
                        '<div align="left"><b style="font-size:20px;">'
                        + "Legend for labels of the target feature:</b></div>"
                    )
                    .to_html()
                )
            )

        if legend_labels is not None:
            # CAUTION: cls.classes is a list of strings. Make sure users know that labels are all converted to strings.
            if isinstance(legend_labels, dict) and set(cls.classes).issubset(
                set(legend_labels.keys())
            ):
                render_legend_labels(legend_labels)
                cls.legend_labels = legend_labels
            else:
                logger.error(
                    "The provided `legend_labels` is either not a Python dict or does not possess all possible class labels."
                )
            return

        # try to remove leading words
        def _check_for_redundant_words(label_vec, prefix, max_len=MAX_LEGEND_LEN):
            words_found = False
            classes = [lab for lab in label_vec]
            while len(set([lab.split()[0] for lab in classes])) <= 1:
                # remove that word from vec, and add it to prefix
                additional_prefix = classes[0].split()[0]
                prefix = (
                    prefix[:-3] + additional_prefix + " ..."
                    if words_found
                    else additional_prefix + " ..."
                )
                classes = [lab[len(additional_prefix) + 1 :] for lab in classes]
                words_found = True

            classes = (
                classes if words_found else [label[max_len:] for label in label_vec]
            )
            return classes, prefix

        # returns mapping from real labels to psuedo-labels, when psuedo-labels are not the first X letter
        def _resolve_conflict(label_vec, prefix, max_len=MAX_LEGEND_LEN):
            classes, prefix = _check_for_redundant_words(label_vec, prefix)
            label_dict = _get_labels(classes, max_len=max_len)
            resolved = {}
            for orig, new in label_dict.items():
                resolved[prefix[:-3] + orig] = "..." + new if new[:3] != "..." else new
            return resolved

        # returns mapping from provided list of strings, to short, unique substrings
        def _get_labels(classes, max_len=MAX_LEGEND_LEN):
            conflict_dict = {}
            for label in classes:
                prefix = label if len(label) < max_len + 3 else label[:max_len] + "..."
                if conflict_dict.get(prefix, None) is None:
                    conflict_dict[prefix] = [label]
                else:
                    conflict_dict[prefix].append(label)
            out = {}
            for k, v in conflict_dict.items():
                if len(v) == 1:
                    out[v[0]] = k
                else:
                    resolved = _resolve_conflict(v, k, max_len=MAX_LEGEND_LEN)
                    out.update(resolved)
            return out

        cls.legend_labels = _get_labels(cls.classes)
        if set(cls.legend_labels.keys()) != set(cls.legend_labels.values()):
            logger.info(
                f"Class labels greater than {MAX_CHARACTERS_LEN} characters have been truncated. "
                "Use the `legend_labels` parameter to define labels."
            )
            render_legend_labels(cls.legend_labels)

    # evaluation is a DataFrame with models as columns and metrics as rows
    @classmethod
    def plot(
        cls,
        evaluation,
        plots,
        num_classes,
        perfect=False,
        baseline=True,
        legend_labels=None,
    ):
        """Generates the evaluation plot

        Parameters
        ----------
        evaluation (DataFrame):
            DataFrame with models as columns and metrics as rows.
        plots (str):
            The plot type based on class attribute `prob_type`.
        num_classes (int):
            The number of classes for the model.
        perfect (bool, optional):
            Whether to display the curve of a perfect classifier. Default value is `False`.
        baseline (bool, optional):
            Whether to display the curve of the baseline, featureless model. Default value is `True`.
        legend_labels (dict, optional):
            Legend labels dictionary. Default value is `None`. If legend_labels not specified class names will be used for plots.

        Returns
        -------
        Nothing
        """

        cls.perfect = perfect
        cls.baseline = baseline
        # get plots to show
        if num_classes == 2:
            cls.prob_type = "_bin"
        elif num_classes > 2:
            cls.prob_type = "_multi"
        else:
            cls.prob_type = "_reg"
        plot_details = getattr(cls, cls.prob_type + "_plots_details")
        if plots is None:
            plots = getattr(cls, cls.prob_type + "_plots")
            logger.info(
                "Showing plot types: {}.".format(
                    ", ".join(
                        [
                            "{}".format(EvaluationPlot._pretty_titles_map[str(p)])
                            for p in plots
                        ]
                    ),
                    ", ".join(["{}".format(x) for x in map(str, plots)]),
                )
            )
            logger.info(plot_details)

        if cls.prob_type == "_bin":
            if "lift_chart" in plots and "gain_chart" in plots:
                plots.remove("lift_chart")
                plots.remove("gain_chart")
                plots.insert(0, "lift_and_gain_chart")

            if "roc_curve" in plots and "pr_curve" in plots:
                plots.remove("roc_curve")
                plots.remove("pr_curve")
                plots.insert(0, "pr_and_roc_curve")
        elif cls.prob_type == "_multi":
            if (
                "normalized_confusion_matrix" in plots
                and len(evaluation[evaluation.columns[0]]["classes"])
                >= MAX_PLOTTING_CLASSES
            ):
                logger.error(
                    f"Evaluation plotting is not yet supported for multiclass problems with {MAX_PLOTTING_CLASSES} or more classes."
                )
                plots = []
        classes = evaluation[evaluation.columns[0]]["classes"]
        if classes is not None:
            # CAUTION: class labels are converted to strings here.
            # If users are passing in legend_labels, they have to use strings as well.
            # Otherwise get_legend_labels() will complaint.
            # If users are not passing in legend_labels, get_legend_labels() generates them from cls.classes.
            # cls.legend_labels are assigned/created in cls.get_lengend_labels() and contain only strings as keys.
            cls.classes = [str(c) for c in classes]
            cls.get_legend_labels(legend_labels)

        mpl.style.use("default")
        html_raw = []
        for i, plot_type in enumerate(plots):
            fig_title, fig = None, None
            try:
                fig_title, ax_title = plt.subplots(1, 1, figsize=(18, 0.5), dpi=144)
                ax_title.text(
                    0.5,
                    0.5,
                    cls._pretty_titles_map[plot_type],
                    fontsize=16,
                    fontweight="semibold",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax_title.transAxes,
                )
                ax_title.axis("off")
                html_raw.append(_fig_to_html(fig_title))
                if cls.prob_type == "_bin" and plot_type in cls.double_overlay_plots:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), dpi=144)
                elif cls.prob_type == "_bin" and plot_type in ["roc_curve", "pr_curve"]:
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), dpi=144)
                    axs[1].axis("off")
                    ax = [axs[0]]
                elif cls.prob_type == "_bin" and plot_type in [
                    "lift_chart",
                    "gain_chart",
                ]:
                    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), dpi=144)
                    axs[1].axis("off")
                    ax = axs[0]
                else:
                    nrows = math.ceil(len(evaluation.columns) / MAX_PLOTS_PER_ROW)
                    fig, ax = plt.subplots(
                        nrows, MAX_PLOTS_PER_ROW, figsize=(10, 4 * nrows), dpi=144
                    )  # 10, 3.5
                    ax = ax.flatten()
                getattr(cls, "_" + plot_type)(ax, evaluation)
                fig.tight_layout()
                html_raw.append(_fig_to_html(fig))
            except KeyError as e:
                try:
                    if fig_title:
                        plt.close(fig=fig_title)
                    if fig:
                        plt.close(fig=fig)
                except:
                    pass
                logger.warning(
                    f"Evaluator was not able to plot "
                    f"{cls._pretty_titles_map.get(plot_type, plot_type)}, because the relevant "
                    f"metrics had complications. Ensure that `predict` and `predict_proba` "
                    f"are valid."
                )
        return html_raw

    @classmethod
    def _lift_and_gain_chart(cls, ax, evaluation):
        cls._lift_chart(ax[0], evaluation)
        cls._gain_chart(ax[1], evaluation)

    @classmethod
    def _lift_chart(cls, ax, evaluation):
        for mod_name, col in evaluation.iteritems():
            if col["y_score"] is not None:
                ax.plot(
                    col["percentages"][1:],
                    [1] + list(col["lift"]),
                    label=cls._get_formatted_title(mod_name),
                )
        if cls.baseline:
            ax.plot([-10, 110], [1, 1], **cls.baseline_kwargs)
        if cls.perfect:
            perf_idx = next(
                idx
                for idx, scores in enumerate(evaluation.loc["y_score"])
                if scores is not None
            )
            ax.plot(
                evaluation.loc["percentages"][perf_idx][1:],
                [1] + list(evaluation.loc["perfect_lift"][perf_idx]),
                **cls.perfect_kwargs,
            )
        ax.legend(loc="upper right", frameon=False)
        ax.set_xlabel("Percentage of Population", fontsize=12)
        ax.set_ylabel("Lift", fontsize=12)
        ax.set_title("Lift Chart", y=1.08, fontsize=14)
        ax.grid(linewidth=0.2, which="both")
        ax.set_xlim([-10, 110])

    @classmethod
    def _gain_chart(cls, ax, evaluation):
        for mod_name, col in evaluation.iteritems():
            if col["y_score"] is not None:
                ax.plot(
                    col["percentages"],
                    list(col["cumulative_gain"]),
                    label=cls._get_formatted_title(mod_name),
                )
        if cls.baseline:
            ax.plot([-10, 110], [-10, 110], **cls.baseline_kwargs)
        if cls.perfect:
            perf_idx = next(
                idx
                for idx, scores in enumerate(evaluation.loc["y_score"])
                if scores is not None
            )
            ax.plot(
                evaluation.loc["percentages"][perf_idx],
                evaluation.loc["perfect_gain"][perf_idx],
                **cls.perfect_kwargs,
            )
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlabel("Percentage of Population", fontsize=12)
        ax.set_ylabel("Percentage of Positive Class", fontsize=12)
        ax.set_title("Gain Chart", y=1.08, fontsize=14)
        ax.grid(linewidth=0.2, which="both")
        ax.set_xlim([-10, 110])
        ax.set_ylim([-10, 110])

    @classmethod
    def _pr_and_roc_curve(cls, ax, evaluation):
        cls._pr_curve([ax[0]], evaluation)
        cls._roc_curve([ax[1]], evaluation)

    @classmethod
    def _pr_curve(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            if cls.prob_type == "_bin":
                for mod_name, col in evaluation.iteritems():
                    if col["y_score"] is not None:
                        ax.plot(
                            col["recall_values"],
                            col["precision_values"],
                            label="%s (Precision: %s)"
                            % (
                                cls._get_formatted_title(mod_name),
                                "{:.3f}".format(col["precision"]),
                            ),
                        )
                        ax.plot(
                            *col["pr_best_model_score"],
                            color=ax.get_lines()[-1].get_color(),
                            marker="*",
                        )
            else:
                model_name = evaluation.columns[i]
                mod = evaluation[model_name]
                if mod["y_score"] is not None:
                    for j, lab in enumerate(mod.classes):
                        # cls.legend_labels contains only strings as keys.
                        lab = str(lab)
                        ax.plot(
                            mod["recall_values"][j],
                            mod["precision_values"][j],
                            label="%s (Precision: %s)"
                            % (
                                cls.legend_labels[lab],
                                "{:.3f}".format(mod["precision_by_label"][j]),
                            ),
                        )
                        ax.plot(
                            *mod["pr_best_model_score"][j],
                            color=ax.get_lines()[-1].get_color(),
                            marker="*",
                        )

            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("Precision Recall Curve", y=1.08, fontsize=14)
            ax.grid(linewidth=0.2, which="both")
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
            handles, labels = ax.get_legend_handles_labels()
            star = mlines.Line2D(
                [],
                [],
                color="black",
                marker="*",
                linestyle="None",
                markersize=5,
                label="Minimum Error Rate",
            )
            handles.append(star)
            labels.append("Minimum Error Rate")
            ax.legend(
                loc="upper right",
                labels=labels,
                handles=handles,
                frameon=False,
                fontsize="x-small",
            )

    @classmethod
    def _roc_curve(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            if cls.prob_type == "_bin":
                for mod_name, col in evaluation.iteritems():
                    if col["y_score"] is not None:
                        ax.plot(
                            col["false_positive_rate"],
                            col["true_positive_rate"],
                            label="%s (AUC: %s)"
                            % (
                                cls._get_formatted_title(mod_name),
                                "{:.3f}".format(col["auc"]),
                            ),
                        )
                        ax.plot(
                            *col["roc_best_model_score"],
                            color=ax.get_lines()[-1].get_color(),
                            marker="*",
                        )
            else:
                model_name = evaluation.columns[i]
                mod = evaluation[model_name]
                if mod["y_score"] is not None:
                    for j, lab in enumerate(mod.classes):
                        # cls.legend_labels contains only strings as keys.
                        lab = str(lab)
                        ax.plot(
                            mod["fpr_by_label"][j],
                            mod["tpr_by_label"][j],
                            label="%s (AUC: %s)"
                            % (cls.legend_labels[lab], "{:.3f}".format(mod["auc"][j])),
                        )
                        ax.plot(
                            *mod["roc_best_model_score"][j],
                            color=ax.get_lines()[-1].get_color(),
                            marker="*",
                        )
            if cls.baseline:
                ax.plot([-0.1, 1.1], [-0.1, 1.1], **cls.baseline_kwargs)
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("ROC Curve", y=1.08, fontsize=14)
            ax.grid(linewidth=0.2, which="both")
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
            handles, labels = ax.get_legend_handles_labels()
            star = mlines.Line2D(
                [],
                [],
                color="black",
                marker="*",
                linestyle="None",
                markersize=5,
                label="Youden's J Statistic",
            )
            handles.append(star)
            labels.append("Youden's J Statistic")
            ax.legend(
                loc="lower right",
                labels=labels,
                handles=handles,
                frameon=False,
                fontsize="x-small",
            )

    @classmethod
    def _ks_statistics(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            model_name = evaluation.columns[i]
            mod = evaluation[model_name]

            ax.set_title(model_name, fontsize=14)
            if mod["y_score"] is not None:
                ax.plot(
                    mod["ks_thresholds"],
                    mod["ks_pct1"],
                    lw=3,
                    label=mod["ks_labels"][0],
                )
                ax.plot(
                    mod["ks_thresholds"],
                    mod["ks_pct2"],
                    lw=3,
                    label=mod["ks_labels"][1],
                )
                if cls.baseline:
                    idx = np.where(mod["ks_thresholds"] == mod["max_distance_at"])[0][0]
                    ax.axvline(
                        mod["max_distance_at"],
                        *sorted([mod["ks_pct1"][idx], mod["ks_pct2"][idx]]),
                        label="KS Statistic: {:.3f} at {:.3f}".format(
                            mod["ks_statistic"], mod["max_distance_at"]
                        ),
                        linestyle="--",
                        color=".2",
                    )

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])

            ax.set_xlabel("Threshold", fontsize=12)
            ax.set_ylabel("Percentage below threshold", fontsize=12)
            ax.tick_params(labelsize=10)
            ax.legend(loc="lower right", fontsize=8)

    @classmethod
    def _pretty_barh(
        cls, ax, x, y, axis_labels=None, title=None, axis_lim=None, plot_kwargs=None
    ):
        # cls.legend_labels contains only strings as keys.
        new_lab = [cls.legend_labels[str(item)] for item in x]
        ax.barh(
            new_lab,
            y,
            color=["teal", "blueviolet", "forestgreen", "peru", "y", "dodgerblue", "r"],
        )
        for j, v in enumerate(y):
            ax.annotate("{:.3f}".format(v), xy=(v / 2, j), va="center", ha="left")
        if axis_labels:
            if axis_labels[0]:
                ax.set_xlabel(axis_labels[0], fontsize=12)
            if axis_labels[1]:
                ax.set_ylabel(axis_labels[1], fontsize=12)
        if title:
            title = cls._get_formatted_title(title)
            ax.set_title(title, y=1.08, fontsize=14)
        if axis_lim:
            ax.set_xlim(axis_lim)

    @classmethod
    def _precision_by_label(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i < n_models:
                col = evaluation.columns[i]
                cls._pretty_barh(
                    ax,
                    evaluation[col]["classes"],
                    evaluation[col]["precision_by_label"],
                    axis_lim=[0, 1],
                    axis_labels=["Precision", None],
                    title=col,
                )
            else:
                ax.axis("off")

    @classmethod
    def _recall_by_label(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i < n_models:
                col = evaluation.columns[i]
                cls._pretty_barh(
                    ax,
                    evaluation[col]["classes"],
                    evaluation[col]["recall_by_label"],
                    axis_lim=[0, 1],
                    axis_labels=["Recall", None],
                    title=col,
                )
            else:
                ax.axis("off")

    @classmethod
    def _f1_by_label(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i < n_models:
                col = evaluation.columns[i]
                cls._pretty_barh(
                    ax,
                    evaluation[col]["classes"],
                    evaluation[col]["f1_by_label"],
                    axis_lim=[0, 1],
                    axis_labels=["F1 Score", None],
                    title=col,
                )
            else:
                ax.axis("off")

    @classmethod
    def _jaccard_by_label(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i < n_models:
                col = evaluation.columns[i]
                cls._pretty_barh(
                    ax,
                    evaluation[col]["classes"],
                    evaluation[col]["jaccard_by_label"],
                    axis_lim=[0, 1],
                    axis_labels=["Jaccard Score", None],
                    title=col,
                )
            else:
                ax.axis("off")

    @classmethod
    def _pretty_scatter(
        cls,
        ax,
        x,
        y,
        s=5,
        alpha=1.0,
        title=None,
        legend=False,
        axis_labels=None,
        axis_lim=None,
        grid=True,
        label=None,
        plot_kwargs=None,
    ):

        if plot_kwargs is None:
            plot_kwargs = {}
        ax.scatter(x, y, s=s, label=label, marker="o", alpha=alpha, **plot_kwargs)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if legend:
            ax.legend(frameon=False)
        if axis_labels:
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
        if title:
            ax.set_title(title, y=1.08, fontsize=14)
        if grid:
            ax.grid(linewidth=0.2)
        if axis_lim:
            if axis_lim[0]:
                ax.set_xlim(axis_lim[0])
            if axis_lim[1]:
                ax.set_ylim(axis_lim[1])

    @classmethod
    def _top_2_features(cls, axs, evaluation):
        pass

    @classmethod
    def _residuals_qq(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            model_name = evaluation.columns[i]
            mod = evaluation[model_name]
            # getattr(ax, self.plot_method)(self.x, y, **self.plot_kwargs, color='#4a91c2', label=label)
            cls._pretty_scatter(
                ax,
                mod["norm_quantiles"],
                mod["residual_quantiles"],
                title=model_name,
                axis_lim=[(-2.7, 2.7), (-3.1, 3.1)],
                axis_labels=["Theoretical Quantiles", "Sample Quantiles"],
            )
            if cls.baseline:
                ax.plot((-100, 100), (-100, 100), **cls.baseline_kwargs)

    @classmethod
    def _residuals_vs_predicted(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            model_name = evaluation.columns[i]
            mod = evaluation[model_name]
            y_pred = np.asarray(mod["y_pred"])
            resid = np.asarray(mod["residuals"])
            cls._pretty_scatter(
                ax,
                y_pred,
                resid,
                s=4,
                alpha=0.5,
                title=model_name,
                axis_labels=["Predicted Values", "Residuals"],
            )
            x_lim = (
                y_pred.min() - y_pred.min() * 0.05,
                y_pred.max() + y_pred.max() * 0.05,
            )
            if cls.baseline:
                ax.plot(x_lim, (0, 0), **cls.baseline_kwargs)
            ax.set_xlim(x_lim)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    @classmethod
    def _residuals_vs_observed(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return
            model_name = evaluation.columns[i]
            mod = evaluation[model_name]
            y_true = np.asarray(mod["y_true"])
            cls._pretty_scatter(
                ax,
                y_true,
                mod["residuals"],
                s=4,
                alpha=0.5,
                title=model_name,
                axis_labels=["Observed Values", "Residuals"],
            )
            x_lim = (
                y_true.min() - y_true.min() * 0.05,
                y_true.max() + y_true.max() * 0.05,
            )
            if cls.baseline:
                ax.plot(x_lim, (0, 0), **cls.baseline_kwargs)
            ax.set_xlim(x_lim)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    @classmethod
    def _observed_vs_predicted(cls, axs, evaluation):
        n_models = len(evaluation.columns)
        for i, ax in enumerate(axs):
            if i >= n_models:
                ax.axis("off")
                return

            model_name = evaluation.columns[i]
            mod = evaluation[model_name]
            ax.scatter(mod["y_true"], mod["y_pred"], s=4, marker="o", alpha=0.5)

            y_true = np.asarray(mod["y_true"])

            yt_min = y_true.min()
            yt_max = y_true.max()

            x_lim = (yt_min - yt_min * 0.05, yt_max + yt_max * 0.05)
            if cls.baseline:
                ax.plot(x_lim, x_lim, **cls.baseline_kwargs)
            ax.set_xlabel("Observed Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(model_name, y=1.08, fontsize=14)
            ax.grid(linewidth=0.2)
            ax.set_xlim(x_lim)

            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    @classmethod
    def _normalized_confusion_matrix(cls, axs, evaluation):
        for model_num, ax in enumerate(axs):
            if model_num >= len(evaluation.columns):
                ax.axis("off")
                return
            model_name = evaluation.columns[model_num]
            mod = evaluation[model_name]
            if cls.prob_type == "_bin":
                labels = [str(lab == mod["positive_class"]) for lab in mod["classes"]]
            else:
                labels = cls.legend_labels.values()

            raw_cm = mod["raw_confusion_matrix"]
            cm = np.asarray(mod["confusion_matrix"])

            ax.set_title(
                "%s\n" % cls._get_formatted_title(model_name), y=1.08, fontsize=14
            )
            ax.imshow(cm, interpolation="nearest", cmap="BuGn")
            x_tick_marks = np.arange(len(labels))
            y_tick_marks = np.arange(len(labels))
            ax.set_xticks(x_tick_marks)
            ax.set_yticks(y_tick_marks)

            ax.set_xticklabels(labels, rotation=90, fontsize=10)
            ax.set_yticklabels(labels, fontsize=10)

            for i, j in itertools.product(
                range(raw_cm.shape[0]), range(raw_cm.shape[1])
            ):
                ax.text(
                    j,
                    i,
                    "%s [%s]" % (round(cm[i][j], 3), raw_cm[i, j]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=45,
                    fontsize=max(3, 10 - max(raw_cm.shape[0], raw_cm.shape[1])),
                    color="white" if cm[i, j] > 0.5 else "black",
                )

            ax.set_ylabel("True label", fontsize=10)
            ax.set_xlabel("Predicted label", fontsize=10)
            ax.grid(False)

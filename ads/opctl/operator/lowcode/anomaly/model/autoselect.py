#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

import report_creator as rc

from ads.opctl.operator.lowcode.anomaly.const import (
    SUBSAMPLE_THRESHOLD,
    OutputColumns,
)

from ..utils import plot_anomaly_threshold_gain
from .base_model import AnomalyOperatorBaseModel

logging.getLogger("report_creator").setLevel(logging.WARNING)


class AutoSelectOperatorModel(AnomalyOperatorBaseModel):
    def _build_model(**kwargs):
        pass

    def generate_report(self):
        from .factory import AnomalyOperatorModelFactory

        anom_outputs = {}
        all_plots = {}
        model_list = self.spec.model_kwargs.pop("model_list", ["lof", "prophet"])
        for m in model_list:
            config_i = self.config
            config_i.spec.model = m
            try:
                anom_outputs[m] = AnomalyOperatorModelFactory.get_model(
                    config_i, self.datasets
                )._build_model()
                all_plots[m] = self._get_plots_from_output(anom_outputs[m], m)
            except:
                logging.debug(f"Model {m} failed. skipping.")
        return self._generate_report(all_plots, anom_outputs, model_list)

    def _get_plots_from_output(self, anomaly_output, model):
        import matplotlib.pyplot as plt

        plt.rcParams.update({"figure.max_open_warning": 0})

        blocks = []
        date_column = self.spec.datetime_column.name
        for target, df in self.datasets.full_data_dict.items():
            if target in anomaly_output.list_categories():
                figure_blocks = []
                time_col = df[date_column].reset_index(drop=True)
                anomaly_col = anomaly_output.get_anomalies_by_cat(category=target)[
                    OutputColumns.ANOMALY_COL
                ]
                anomaly_indices = [
                    i for i, index in enumerate(anomaly_col) if index == 1
                ]
                downsampled_time_col = time_col
                selected_indices = list(range(len(time_col)))
                if self.spec.subsample_report_data:
                    non_anomaly_indices = [
                        i for i in range(len(time_col)) if i not in anomaly_indices
                    ]
                    # Downsample non-anomalous data if it exceeds the threshold (1000)
                    if len(non_anomaly_indices) > SUBSAMPLE_THRESHOLD:
                        downsampled_non_anomaly_indices = non_anomaly_indices[
                            :: len(non_anomaly_indices) // SUBSAMPLE_THRESHOLD
                        ]
                        selected_indices = (
                            anomaly_indices + downsampled_non_anomaly_indices
                        )
                        selected_indices.sort()
                    downsampled_time_col = time_col[selected_indices]

                columns = set(df.columns).difference({date_column})
                for col in columns:
                    y = df[col].reset_index(drop=True)

                    downsampled_y = y[selected_indices]

                    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
                    ax.grid()
                    ax.plot(downsampled_time_col, downsampled_y, color="black")
                    # Plot anomalies
                    for i in anomaly_indices:
                        ax.scatter(time_col[i], y[i], color="red", marker="o")
                    plt.xlabel(date_column)
                    plt.ylabel(col)
                    plt.title(f"`{col}` with reference to anomalies")
                    figure_blocks.append(rc.Widget(ax))
            else:
                figure_blocks = None

            blocks.append(
                rc.Group(*figure_blocks, label=f"{target}_{model}")
            ) if figure_blocks else None
        plots = rc.Select(blocks)
        return plots

    def _generate_report(self, all_plots, anomaly_outputs, model_list):
        """Genreates a report for the model."""
        title_text = rc.Heading("Auto-Select Report", level=2)
        summary = rc.Text(
            "This report presents the results of multiple model tuning experiments, visualized to facilitate comparative analysis and informed model selection. Each modeling framework has been systematically trained, hyperparameter-tuned, and validated to optimize performance based on the characteristics of your dataset."
        )

        model_sections = []
        for m in model_list:
            model_sections.append(all_plots[m])
            sec_text = rc.Heading(f"Train Evaluation Metrics for {m}", level=3)
            sec = rc.DataTable(self._evaluation_metrics(anomaly_outputs[m]), index=True)
            model_sections.append(sec_text)
            model_sections.append(sec)
            cat1 = anomaly_outputs[m].list_categories()[0]
            print(anomaly_outputs[m].get_scores_by_cat(cat1))
            fig = plot_anomaly_threshold_gain(
                anomaly_outputs[m].get_scores_by_cat(cat1)["score"],
                title=f"Threshold Analysis for {m}",
            )
            model_sections.append(rc.Widget(fig))

        report_sections = [title_text, summary] + model_sections

        # save the report and result CSV
        self._save_report(
            report_sections=report_sections,
            anomaly_output=None,
            test_metrics=None,
        )

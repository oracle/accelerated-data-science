#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import fsspec
import pandas as pd
import report_creator as rc

from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import default_signer
from ads.opctl.operator.lowcode.common.utils import (
    human_time_friendly,
    enable_print,
    disable_print,
    write_data,
)
from .factory import SupportedModels
from .recommender_dataset import RecommenderDatasets
from ..operator_config import RecommenderOperatorConfig
from plotly import graph_objects as go
import matplotlib.pyplot as plt


class RecommenderOperatorBaseModel(ABC):
    """The base class for the recommender detection operator models."""

    def __init__(self, config: RecommenderOperatorConfig, datasets: RecommenderDatasets):
        self.config = config
        self.spec = self.config.spec
        self.datasets = datasets

    def generate_report(self):
        item_col = self.spec.item_column
        user_col = self.spec.user_column
        interaction_col = self.spec.interaction_column
        start_time = time.time()
        result_df, metrics = self._build_model()
        elapsed_time = time.time() - start_time
        logger.info("Building the models completed in %s seconds", elapsed_time)

        if self.spec.generate_report:
            # build the report
            (
                model_description,
                other_sections,
            ) = self._generate_report()

            header_section = rc.Block(
                rc.Heading("Recommender Report", level=1),
                rc.Text(
                    f"The recommendations was generated using {SupportedModels.SVD.upper()}. {model_description}"
                ),
                rc.Group(
                    rc.Metric(
                        heading="Recommendations was generated in ",
                        value=human_time_friendly(elapsed_time),
                    ),
                    rc.Metric(
                        heading="Num users",
                        value=len(self.datasets.users),
                    ),
                    rc.Metric(
                        heading="Num items",
                        value=len(self.datasets.items),
                    )
                ),
            )

        summary = rc.Block(
            header_section,
        )
        # user and item distributions in interactions
        user_title = rc.Heading("User Statistics", level=2)
        user_rating_counts = self.datasets.interactions[user_col].value_counts()
        fig_user = go.Figure(data=[go.Histogram(x=user_rating_counts, nbinsx=100)])
        fig_user.update_layout(
            title=f'Distribution of the number of interactions by {user_col}',
            xaxis_title=f'Number of {interaction_col}',
            yaxis_title=f'Number of {user_col}',
            bargap=0.2
        )
        item_title = rc.Heading("Item Statistics", level=2)
        item_rating_counts = self.datasets.interactions[item_col].value_counts()
        fig_item = go.Figure(data=[go.Histogram(x=item_rating_counts, nbinsx=100)])
        fig_item.update_layout(
            title=f'Distribution of the number of interactions by {item_col}',
            xaxis_title=f'Number of {interaction_col}',
            yaxis_title=f'Number of {item_col}',
            bargap=0.2
        )
        result_heatmap_title = rc.Heading("Sample Recommendations", level=2)
        sample_items = result_df[item_col].head(100).index
        filtered_df = result_df[result_df[item_col].isin(sample_items)]
        data = filtered_df.pivot(index=user_col, columns=item_col, values=interaction_col)
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='Viridis'
        ))
        fig.update_layout(
            title='Recommendation heatmap of User-Item Interactions (sample)',
            width=1500,
            height=800,
            xaxis_title=item_col,
            yaxis_title=user_col,
            coloraxis_colorbar=dict(title=interaction_col)
        )
        plots = [user_title, rc.Widget(fig_user),
                 item_title, rc.Widget(fig_item),
                 result_heatmap_title, rc.Widget(fig)]

        test_metrics_sections = [rc.DataTable(pd.DataFrame(metrics, index=[0]))]
        yaml_appendix_title = rc.Heading("Reference: YAML File", level=2)
        yaml_appendix = rc.Yaml(self.config.to_dict())
        report_sections = (
                [summary]
                + plots
                + test_metrics_sections
                + other_sections
                + [yaml_appendix_title, yaml_appendix]
        )

        # save the report and result CSV
        self._save_report(
            report_sections=report_sections,
            result_df=result_df
        )

    def _evaluation_metrics(self):
        pass

    def _test_data_evaluate_metrics(self):
        pass

    def _save_report(self, report_sections: Tuple, result_df: pd.DataFrame):
        """Saves resulting reports to the given folder."""

        unique_output_dir = self.spec.output_directory.url

        if ObjectStorageDetails.is_oci_path(unique_output_dir):
            storage_options = default_signer()
        else:
            storage_options = dict()

            # report-creator html report
            if self.spec.generate_report:
                with tempfile.TemporaryDirectory() as temp_dir:
                    report_local_path = os.path.join(temp_dir, "___report.html")
                    disable_print()
                    with rc.ReportCreator("My Report") as report:
                        report.save(rc.Block(*report_sections), report_local_path)
                    enable_print()

                    report_path = os.path.join(unique_output_dir, self.spec.report_filename)
                    with open(report_local_path) as f1:
                        with fsspec.open(
                                report_path,
                                "w",
                                **storage_options,
                        ) as f2:
                            f2.write(f1.read())

        # recommender csv report
        write_data(
            data=result_df,
            filename=os.path.join(unique_output_dir, self.spec.recommendations_filename),
            format="csv",
            storage_options=storage_options,
        )

        logger.info(
            f"The outputs have been successfully "
            f"generated and placed into the directory: {unique_output_dir}."
        )

    @abstractmethod
    def _generate_report(self):
        """
        Generates the report for the particular model.
        The method that needs to be implemented on the particular model level.
        """

    @abstractmethod
    def _build_model(self) -> [pd.DataFrame, Dict]:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Tuple

import fsspec
import pandas as pd

from ads.common.auth import default_signer
from ads.opctl import logger

from .. import utils
from ..operator_config import AnomalyOperatorConfig, AnomalyOperatorSpec
from .anomaly_dataset import AnomalyDatasets

class AnomalyOperatorBaseModel(ABC):
    """The base class for the anomaly detection operator models."""

    def __init__(self, config: AnomalyOperatorConfig, datasets: AnomalyDatasets):
        """Instantiates the AnomalyOperatorBaseModel instance.

        Properties
        ----------
        config: AnomalyOperatorConfig
            The anomaly detection operator configuration.
        """

        self.config: AnomalyOperatorConfig = config
        self.spec: AnomalyOperatorSpec = config.spec
        self.datasets = datasets

    def generate_report(self):
        """Generates the report."""
        import datapane as dp

        # load data and build models
        start_time = time.time()
        input_data = self._load_data()
        result_df = self._build_model()
        elapsed_time = time.time() - start_time

        report_sections = []
        title_text = dp.Text("# Anomaly Detection Report")

        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=self.config.to_yaml(), language="yaml")
        report_sections = [title_text] + [yaml_appendix_title, yaml_appendix]

        # save the report and result CSV
        self._save_report(
            report_sections=report_sections,
            result_df=result_df,
        )

    def _load_data(self):
        """Loads input data."""

        return utils._load_data(
            filename=self.spec.input_data.url,
            format=self.spec.input_data.format,
            storage_options=default_signer(),
            columns=self.spec.input_data.columns,
        )

    def _save_report(self, report_sections: Tuple, result_df: pd.DataFrame):
        """Saves resulting reports to the given folder."""
        import datapane as dp

        if self.spec.output_directory:
            output_dir = self.spec.output_directory.url
        else:
            output_dir = "tmp_operator_result"
            logger.warn(
                "Since the output directory was not specified, the output will be saved to {} directory.".format(
                    output_dir
                )
            )
        # datapane html report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_local_path = os.path.join(temp_dir, "___report.html")
            dp.save_report(report_sections, report_local_path)
            with open(report_local_path) as f1:
                with fsspec.open(
                    os.path.join(output_dir, self.spec.report_file_name),
                    "w",
                    **default_signer(),
                ) as f2:
                    f2.write(f1.read())

        logger.warn(
            f"The report has been successfully "
            f"generated and placed to the: {output_dir}."
        )

    @abstractmethod
    def _generate_report(self):
        """
        Generates the report for the particular model.
        The method that needs to be implemented on the particular model level.
        """

    @abstractmethod
    def _build_model(self) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

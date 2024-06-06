#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import time
from abc import ABC, abstractmethod

import pandas as pd

from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import default_signer
from ads.opctl.operator.lowcode.common.utils import (
    write_data,
)
from .recommender_dataset import RecommenderDatasets
from ..operator_config import RecommenderOperatorConfig


class RecommenderOperatorBaseModel(ABC):
    """The base class for the recommender detection operator models."""

    def __init__(self, config: RecommenderOperatorConfig, datasets: RecommenderDatasets):
        self.spec = config.spec
        self.datasets = datasets

    def generate_report(self):
        start_time = time.time()
        result_df = self._build_model()
        elapsed_time = time.time() - start_time
        logger.info("Building the models completed in %s seconds", elapsed_time)
        # save the report and result CSV
        self._save_report(
            result_df=result_df
        )

    def _save_report(self, result_df):
        """Saves resulting reports to the given folder."""

        unique_output_dir = self.spec.output_directory.url

        if ObjectStorageDetails.is_oci_path(unique_output_dir):
            storage_options = default_signer()
        else:
            storage_options = dict()

        # forecast csv report
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
    def _build_model(self) -> pd.DataFrame:
        """
        Build the model.
        The method that needs to be implemented on the particular model level.
        """

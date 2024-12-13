#!/usr/bin/env python
import json
# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pickle
import shutil
import sys
import tempfile

import pandas as pd
from joblib import dump

from ads.opctl import logger
from ads.common.model_export_util import prepare_generic_model
from ads.opctl.operator.lowcode.common.utils import write_data, call_pandas_fsspec

from ..model.forecast_datasets import AdditionalData
from ..operator_config import ForecastOperatorSpec


class ModelDeploymentManager:
    def __init__(self, spec: ForecastOperatorSpec, additional_data: AdditionalData, previous_model_version=None):
        self.spec = spec
        self.model_name = spec.model
        self.horizon = spec.horizon
        self.additional_data = additional_data.get_dict_by_series()
        self.model_obj = {}
        self.display_name = spec.what_if_analysis.model_name
        self.project_id = spec.what_if_analysis.project_id
        self.compartment_id = spec.what_if_analysis.compartment_id
        self.path_to_artifact = f"{self.spec.output_directory.url}/artifacts/"
        self.pickle_file_path = f"{self.spec.output_directory.url}/model.pkl"
        self.model_version = previous_model_version + 1 if previous_model_version else 1

    def _satiny_test(self):
        """
        Function perform sanity test for saved artifact
        """
        sys.path.insert(0, f"{self.path_to_artifact}")
        from score import load_model, predict
        _ = load_model()

        # Write additional data to tmp file and perform sanity check
        with tempfile.NamedTemporaryFile(suffix='.csv') as temp_file:
            one_series = next(iter(self.additional_data))
            sample_prediction_data = self.additional_data[one_series].tail(self.horizon)
            sample_prediction_data[self.spec.target_category_columns[0]] = one_series
            date_col_name = self.spec.datetime_column.name
            date_col_format = self.spec.datetime_column.format
            sample_prediction_data[date_col_name] = sample_prediction_data[date_col_name].dt.strftime(date_col_format)
            sample_prediction_data.to_csv(temp_file.name, index=False)
            additional_data_uri = "additional_data"
            input_data = {additional_data_uri: {"url": temp_file.name}}
            prediction_test = predict(input_data, _)
            logger.info(f"prediction test completed with result :{prediction_test}")

    def _copy_score_file(self):
        """
        Copies the score.py to the artifact_path.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            score_file = os.path.join(current_dir, "score.py")
            destination_file = os.path.join(self.path_to_artifact, os.path.basename(score_file))
            shutil.copy2(score_file, destination_file)
            logger.info(f"score.py copied successfully to {self.path_to_artifact}")
        except Exception as e:
            logger.warn(f"Error copying file: {e}")
            raise e

    def save_to_catalog(self):
        """Save the model to a model catalog"""
        with open(self.pickle_file_path, 'rb') as file:
            self.model_obj = pickle.load(file)

        if not os.path.exists(self.path_to_artifact):
            os.mkdir(self.path_to_artifact)

        artifact_dict = {"spec": self.spec.to_dict(), "models": self.model_obj}
        dump(artifact_dict, os.path.join(self.path_to_artifact, "model.joblib"))
        artifact = prepare_generic_model(self.path_to_artifact, function_artifacts=False, force_overwrite=True,
                                         data_science_env=True)

        self._copy_score_file()
        self._satiny_test()

        if isinstance(self.model_obj, dict):
            series = self.model_obj.keys()
        else:
            series = self.additional_data.keys()
        description = f"The object contains {len(series)} {self.model_name} models"

        catalog_id = "None"
        if not os.environ.get("TEST_MODE", False):
            catalog_entry = artifact.save(display_name=self.display_name,
                                          compartment_id=self.compartment_id,
                                          project_id=self.project_id,
                                          description=description)
            catalog_id = catalog_entry.id


        logger.info(f"Saved {self.model_name} version-v{self.model_version} to model catalog"
              f" with catalog id : {catalog_id}")

        catalog_mapping = {"catalog_id": catalog_id, "series": list(series)}

        write_data(
            data=pd.DataFrame([catalog_mapping]),
            filename=os.path.join(
                self.spec.output_directory.url, "model_ids.csv"
            ),
            format="csv"
        )
        return catalog_id

    def create_deployment(self, deployment_config):
        """Create a model deployment serving"""
        pass

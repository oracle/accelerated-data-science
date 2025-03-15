#!/usr/bin/env python
# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pickle
import shutil
import sys
import tempfile

import cloudpickle
import oci
import pandas as pd
from oci.data_science import DataScienceClient, DataScienceClientCompositeOperations
from oci.data_science.models import (
    CategoryLogDetails,
    CreateModelDeploymentDetails,
    FixedSizeScalingPolicy,
    InstanceConfiguration,
    LogDetails,
    ModelConfigurationDetails,
    SingleModelDeploymentConfigurationDetails,
)

from ads.common.model_export_util import prepare_generic_model
from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.common.utils import create_log_in_log_group
from ads.opctl.operator.lowcode.common.utils import (
    default_signer,
    write_data,
    write_simple_json,
)

from ..model.forecast_datasets import AdditionalData
from ..operator_config import ForecastOperatorSpec


class ModelDeploymentManager:
    def __init__(
        self,
        spec: ForecastOperatorSpec,
        additional_data: AdditionalData,
        previous_model_version=None,
    ):
        self.spec = spec
        self.model_name = spec.model
        self.horizon = spec.horizon
        self.additional_data = additional_data.get_dict_by_series()
        self.model_obj = {}
        self.display_name = spec.what_if_analysis.model_display_name
        self.project_id = (
            spec.what_if_analysis.project_id
            if spec.what_if_analysis.project_id
            else os.environ.get("PROJECT_OCID")
        )
        self.compartment_id = (
            spec.what_if_analysis.compartment_id
            if spec.what_if_analysis.compartment_id
            else os.environ.get("NB_SESSION_COMPARTMENT_OCID")
        )
        if self.project_id is None or self.compartment_id is None:
            raise ValueError("Either project_id or compartment_id cannot be None.")
        self.path_to_artifact = f"{self.spec.output_directory.url}/artifacts/"
        self.pickle_file_path = f"{self.spec.output_directory.url}/model.pkl"
        self.model_version = previous_model_version + 1 if previous_model_version else 1
        self.catalog_id = None
        self.test_mode = os.environ.get("TEST_MODE", False)
        self.deployment_info = {}

    def _sanity_test(self):
        """
        Function perform sanity test for saved artifact
        """
        org_sys_path = sys.path[:]
        try:
            sys.path.insert(0, f"{self.path_to_artifact}")
            from score import load_model, predict

            _ = load_model()

            # Write additional data to tmp file and perform sanity check
            with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
                one_series = next(iter(self.additional_data))
                sample_prediction_data = self.additional_data[one_series].tail(
                    self.horizon
                )
                sample_prediction_data[self.spec.target_category_columns[0]] = (
                    one_series
                )
                date_col_name = self.spec.datetime_column.name
                date_col_format = self.spec.datetime_column.format
                sample_prediction_data[date_col_name] = sample_prediction_data[
                    date_col_name
                ].dt.strftime(date_col_format)
                sample_prediction_data.to_csv(temp_file.name, index=False)
                input_data = {"additional_data": {"url": temp_file.name}}
                prediction_test = predict(input_data, _)
                logger.info(f"prediction test completed with result :{prediction_test}")
        except Exception as e:
            logger.error(f"An error occurred during the sanity test: {e}")
            raise
        finally:
            sys.path = org_sys_path

    def _copy_score_file(self):
        """
        Copies the score.py to the artifact_path.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            score_file = os.path.join(current_dir, "score.py")
            destination_file = os.path.join(
                self.path_to_artifact, os.path.basename(score_file)
            )
            shutil.copy2(score_file, destination_file)
            logger.info(f"score.py copied successfully to {self.path_to_artifact}")
        except Exception as e:
            logger.warning(f"Error copying file: {e}")
            raise e

    def save_to_catalog(self):
        """Save the model to a model catalog"""
        with open(self.pickle_file_path, "rb") as file:
            self.model_obj = pickle.load(file)

        if not os.path.exists(self.path_to_artifact):
            os.mkdir(self.path_to_artifact)

        artifact_dict = {"spec": self.spec.to_dict(), "models": self.model_obj}
        with open(f"{self.path_to_artifact}/models.pickle", "wb") as f:
            cloudpickle.dump(artifact_dict, f)
        artifact = prepare_generic_model(
            self.path_to_artifact,
            function_artifacts=False,
            force_overwrite=True,
            data_science_env=True,
        )

        self._copy_score_file()
        self._sanity_test()

        if isinstance(self.model_obj, dict):
            series = self.model_obj.keys()
        else:
            series = self.additional_data.keys()
        description = f"The object contains {len(series)} {self.model_name} models"

        if not self.test_mode:
            catalog_entry = artifact.save(
                display_name=self.display_name,
                compartment_id=self.compartment_id,
                project_id=self.project_id,
                description=description,
            )
            self.catalog_id = catalog_entry.id

        logger.info(
            f"Saved {self.model_name} version-v{self.model_version} to model catalog"
            f" with model ocid : {self.catalog_id}"
        )

        self.deployment_info = {"model_ocid": self.catalog_id, "series": list(series)}

    def create_deployment(self):
        """Create a model deployment serving"""

        # create new model deployment
        initial_shape = self.spec.what_if_analysis.model_deployment.initial_shape
        name = self.spec.what_if_analysis.model_deployment.display_name
        description = self.spec.what_if_analysis.model_deployment.description
        auto_scaling_config = self.spec.what_if_analysis.model_deployment.auto_scaling

        # if auto_scaling_config is defined
        if auto_scaling_config:
            scaling_policy = oci.data_science.models.AutoScalingPolicy(
                policy_type="AUTOSCALING",
                auto_scaling_policies=[
                    oci.data_science.models.ThresholdBasedAutoScalingPolicyDetails(
                        auto_scaling_policy_type="THRESHOLD",
                        rules=[
                            oci.data_science.models.PredefinedMetricExpressionRule(
                                metric_expression_rule_type="PREDEFINED_EXPRESSION",
                                metric_type=auto_scaling_config.scaling_metric,
                                scale_in_configuration=oci.data_science.models.PredefinedExpressionThresholdScalingConfiguration(
                                    scaling_configuration_type="THRESHOLD",
                                    threshold=auto_scaling_config.scale_in_threshold,
                                ),
                                scale_out_configuration=oci.data_science.models.PredefinedExpressionThresholdScalingConfiguration(
                                    scaling_configuration_type="THRESHOLD",
                                    threshold=auto_scaling_config.scale_out_threshold,
                                ),
                            )
                        ],
                        maximum_instance_count=auto_scaling_config.maximum_instance,
                        minimum_instance_count=auto_scaling_config.minimum_instance,
                        initial_instance_count=auto_scaling_config.minimum_instance,
                    )
                ],
                cool_down_in_seconds=auto_scaling_config.cool_down_in_seconds,
                is_enabled=True,
            )
            logger.info(
                f"Using autoscaling {auto_scaling_config.scaling_metric} for creating MD"
            )
        else:
            scaling_policy = FixedSizeScalingPolicy(instance_count=1)
            logger.info("Using fixed size policy for creating MD")

        model_configuration_details_object = ModelConfigurationDetails(
            model_id=self.catalog_id,
            instance_configuration=InstanceConfiguration(
                instance_shape_name=initial_shape
            ),
            scaling_policy=scaling_policy,
            bandwidth_mbps=20,
        )

        single_model_config = SingleModelDeploymentConfigurationDetails(
            deployment_type="SINGLE_MODEL",
            model_configuration_details=model_configuration_details_object,
        )

        log_group = self.spec.what_if_analysis.model_deployment.log_group
        log_id = self.spec.what_if_analysis.model_deployment.log_id
        if not log_id and not self.test_mode:
            signer = oci.auth.signers.get_resource_principals_signer()
            auth = {"signer": signer, "config": {}}
            log_id = create_log_in_log_group(self.compartment_id, log_group, auth)

        logs_configuration_details_object = CategoryLogDetails(
            access=LogDetails(log_group_id=log_group, log_id=log_id),
            predict=LogDetails(log_group_id=log_group, log_id=log_id),
        )

        model_deploy_configuration = CreateModelDeploymentDetails(
            display_name=name,
            description=description,
            project_id=self.project_id,
            compartment_id=self.compartment_id,
            model_deployment_configuration_details=single_model_config,
            category_log_details=logs_configuration_details_object,
        )

        if not self.test_mode:
            auth = oci.auth.signers.get_resource_principals_signer()
            data_science = DataScienceClient({}, signer=auth)
            data_science_composite = DataScienceClientCompositeOperations(data_science)
            model_deployment = (
                data_science_composite.create_model_deployment_and_wait_for_state(
                    model_deploy_configuration, wait_for_states=["SUCCEEDED", "FAILED"]
                )
            )
            self.deployment_info["work_request"] = model_deployment.data.id
            logger.info(f"deployment metadata :{model_deployment.data}")
            md = data_science.get_model_deployment(
                model_deployment_id=model_deployment.data.resources[0].identifier
            )
            self.deployment_info["model_deployment_ocid"] = md.data.id
            self.deployment_info["status"] = md.data.lifecycle_state
            endpoint_url = md.data.model_deployment_url
            self.deployment_info["model_deployment_endpoint"] = (
                f"{endpoint_url}/predict"
            )
            self.deployment_info["log_id"] = log_id

    def save_deployment_info(self):
        output_dir = self.spec.output_directory.url
        if ObjectStorageDetails.is_oci_path(output_dir):
            storage_options = default_signer()
        else:
            storage_options = {}
        write_data(
            data=pd.DataFrame.from_dict(self.deployment_info),
            filename=os.path.join(output_dir, "deployment_info.json"),
            format="json",
            storage_options=storage_options,
            index=False,
            indent=4,
            orient="records",
        )
        write_simple_json(
            self.deployment_info, os.path.join(output_dir, "deployment_info.json")
        )
        logger.info(f"Saved deployment info to {output_dir}")

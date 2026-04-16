#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import pickle
import shutil
import sys
import tempfile

import cloudpickle
import fsspec
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
    load_data,
    write_simple_json,
    write_data,
)
from ads.opctl.operator.lowcode.regression.operator_config import RegressionOperatorSpec


class ModelDeploymentManager:
    """Handles regression model packaging, catalog save, sanity test, and deployment."""

    def __init__(
        self,
        spec: RegressionOperatorSpec,
        model_name: str,
        previous_model_version=None,
    ):
        self.spec = spec
        self.model_name = model_name
        self.display_name = spec.save_and_deploy_to_md.model_catalog_display_name
        self.project_id = (
            spec.save_and_deploy_to_md.project_id
            if spec.save_and_deploy_to_md.project_id
            else os.environ.get("PROJECT_OCID")
        )
        self.compartment_id = (
            spec.save_and_deploy_to_md.compartment_id
            if spec.save_and_deploy_to_md.compartment_id
            else os.environ.get("NB_SESSION_COMPARTMENT_OCID")
        )
        if self.project_id is None or self.compartment_id is None:
            raise ValueError("Either project_id or compartment_id cannot be None.")

        tmp_dir_obj = tempfile.TemporaryDirectory()
        self._artifact_dir_obj = tmp_dir_obj
        self.path_to_artifact = tmp_dir_obj.name
        self.pickle_file_path = f"{self.spec.output_directory.url}/model.pkl"
        self.model_version = previous_model_version + 1 if previous_model_version else 1
        self.catalog_id = None
        self.model_obj = None
        self.test_mode = os.environ.get("TEST_MODE", False)
        self.deployment_info = {
            "model_name": self.model_name,
            "saved_to_model_catalog": False,
            "deployed_to_model_deployment": False,
        }

    def _get_sanity_test_data(self) -> pd.DataFrame:
        """Loads a small subset from training data for packaging sanity checks."""
        try:
            training_data = load_data(self.spec.training_data)
        except Exception as e:
            logger.warning(
                "Skipping regression deployment sanity test because training data "
                f"could not be loaded. Error: {e}"
            )
            return pd.DataFrame()

        if training_data is None or training_data.empty:
            logger.info(
                "Skipping regression deployment sanity test because training data is unavailable."
            )
            return pd.DataFrame()

        return training_data.head(min(len(training_data), 5)).copy()

    def _sanity_test(self):
        """Runs a local sanity check against the generated score.py and packaged model."""
        sanity_data = self._get_sanity_test_data()
        if sanity_data.empty:
            return

        org_sys_path = sys.path[:]
        try:
            sys.path.insert(0, f"{self.path_to_artifact}")
            from score import load_model, predict

            loaded_model = load_model()
            input_data = {"data": sanity_data.to_dict(orient="records")}
            prediction_test = predict(input_data, loaded_model)
            logger.info(f"Regression deployment sanity test completed with result: {prediction_test}")
        except Exception as e:
            logger.error(f"An error occurred during regression deployment sanity test: {e}")
            raise
        finally:
            sys.path = org_sys_path

    def _copy_score_file(self):
        """Copies the deployment score.py to the artifact path."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            score_file = os.path.join(current_dir, "score.py")
            destination_file = os.path.join(
                self.path_to_artifact, os.path.basename(score_file)
            )
            shutil.copy2(score_file, destination_file)
            logger.info(f"score.py copied successfully to {self.path_to_artifact}")
        except Exception as e:
            logger.warning(f"Error copying score.py: {e}")
            raise

    def save_to_catalog(self):
        """Save the regression artifact to OCI Model Catalog."""
        is_oci = ObjectStorageDetails.is_oci_path(self.pickle_file_path)
        storage_signer = default_signer() if is_oci else {}
        with fsspec.open(
            self.pickle_file_path,
            "rb",
            **storage_signer,
        ) as f:
            self.model_obj = pickle.load(f)

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

        description = f"Regression operator deployment artifact for model `{self.model_name}`."
        if not self.test_mode:
            catalog_entry = artifact.save(
                display_name=self.display_name,
                compartment_id=self.compartment_id,
                project_id=self.project_id,
                description=description,
            )
            self.catalog_id = catalog_entry.id

        self.deployment_info.update(
            {
                "model_ocid": self.catalog_id,
                "saved_to_model_catalog": True,
            }
        )
        logger.info(
            f"Saved {self.model_name} version-v{self.model_version} to model catalog"
            f" with model ocid : {self.catalog_id}"
        )

    def create_deployment(self):
        """Create an OCI model deployment for the saved regression model."""
        if not self.catalog_id and not self.test_mode:
            raise ValueError("Model must be saved to catalog before creating deployment.")

        initial_shape = self.spec.save_and_deploy_to_md.model_deployment.initial_shape
        name = self.spec.save_and_deploy_to_md.model_deployment.display_name
        description = self.spec.save_and_deploy_to_md.model_deployment.description
        auto_scaling_config = self.spec.save_and_deploy_to_md.model_deployment.auto_scaling

        if auto_scaling_config and auto_scaling_config.maximum_instance:
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

        log_group = self.spec.save_and_deploy_to_md.model_deployment.log_group
        log_id = self.spec.save_and_deploy_to_md.model_deployment.log_id
        if not log_id and log_group and not self.test_mode:
            signer = oci.auth.signers.get_resource_principals_signer()
            auth = {"signer": signer, "config": {}}
            log_id = create_log_in_log_group(self.compartment_id, log_group, auth)

        logs_configuration_details_object = None
        if log_group and log_id:
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

        self.deployment_info["deployed_to_model_deployment"] = True

    def save_deployment_info(self):
        output_dir = self.spec.output_directory.url
        if ObjectStorageDetails.is_oci_path(output_dir):
            storage_options = default_signer()
        else:
            storage_options = {}
        write_data(
            data=pd.DataFrame.from_dict(self.deployment_info, orient="index").T,
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

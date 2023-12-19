#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/from typing import Dict


import copy
import logging
import traceback
from typing import Any, Dict

import oci.util as oci_util

from ads.common.oci_datascience import DSCNotebookSession
from ads.config import COMPARTMENT_OCID, NB_SESSION_OCID, PROJECT_OCID
from ads.jobs.builders.base import Builder

MODEL_DEPLOYMENT_INFRASTRUCTURE_TYPE = "datascienceModelDeployment"
MODEL_DEPLOYMENT_INFRASTRUCTURE_KIND = "infrastructure"

DEFAULT_BANDWIDTH_MBPS = 10
DEFAULT_REPLICA = 1
DEFAULT_SHAPE_NAME = "VM.Standard.E4.Flex"
DEFAULT_OCPUS = 1
DEFAULT_MEMORY_IN_GBS = 16

logger = logging.getLogger(__name__)


class ModelDeploymentInfrastructure(Builder):
    """A class used to represent a Model Deployment Infrastructure.

    Attributes
    ----------
    compartment_id: str
        The compartment id of model deployment
    project_id: str
        The project id of model deployment
    shape_name: str
        The shape name of model deployment
    shape_config_details: Dict
        The shape config details of model deployment
    replica: int
        The replica of model deployment
    bandwidth_mbps: int
        The bandwidth of model deployment in mbps
    access_log: Dict
        The access log of model deployment
    predict_log: Dict
        The predict log of model deployment
    log_group_id: str
        The access and predict log group id of model deployment
    log_id: str
        The access and predict log id of model deployment
    web_concurrency: int
        The web concurrency of model deployment
    subnet_id: str
        The subnet id of model deployment

    Methods
    -------
    with_compartment_id(compartment_id)
        Sets the compartment id of model deployment
    with_project_id(project_id)
        Sets the project id of model deployment
    with_shape_name(shape_name)
        Sets the shape name of model deployment
    with_shape_config_details(shape_config_details)
        Sets the shape config details of model deployment
    with_replica(replica)
        Sets the replica of model deployment
    with_bandwidth_mbps(bandwidth_mbps)
        Sets the bandwidth of model deployment in mbps
    with_access_log(log_group_id, log_id)
        Sets the access log of model deployment
    with_predict_log(log_group_id, log_id)
        Sets the predict log of model deployment
    with_log_group_id(log_group_id)
        Sets the access and predict log group id of model deployment
    with_log_id(log_id)
        Sets the access and predict log id of model deployment
    with_web_concurrency(web_concurrency)
        Sets the web concurrency of model deployment
    with_subnet_id(subnet_id)
        Sets the subnet id of model deployment

    Example
    -------
    Build infrastructure from builder apis:
    >>> infrastructure = ModelDeploymentInfrastructure()
    ...        .with_project_id(<project_id>)
    ...        .with_compartment_id(<compartment_id>)
    ...        .with_shape_name("VM.Standard.E4.Flex")
    ...        .with_shape_config_details(
    ...            ocpus=1,
    ...            memory_in_gbs=16
    ...        )
    ...        .with_replica(1)
    ...        .with_bandwidth_mbps(10)
    ...        .with_web_concurrency(10)
    ...        .with_subnet_id(<subnet_id>)
    ...        .with_access_log(
    ...            log_group_id=<log_group_id>,
    ...            log_id=<log_id>
    ...        )
    ...        .with_predict_log(
    ...            log_group_id=<log_group_id>,
    ...            log_id=<log_id>
    ...        )
    >>> infrastructure.to_dict()

    Build infrastructure from yaml:
    >>> infrastructure = ModelDeploymentInfrastructure.from_yaml(uri=<path_to_yaml>)
    """

    CONST_PROJECT_ID = "projectId"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_MODEL_DEPLOYMENT_CONFIG_DETAILS = "modelDeploymentConfigurationDetails"
    CONST_DEPLOYMENT_TYPE = "deploymentType"
    CONST_MODEL_CONFIG_DETAILS = "modelConfigurationDetails"
    CONST_INSTANCE_CONFIG = "instanceConfiguration"
    CONST_SHAPE_NAME = "shapeName"
    CONST_INSTANCE_SHAPE_NAME = "instanceShapeName"
    CONST_SHAPE_CONFIG_DETAILS = "shapeConfigDetails"
    CONST_MODEL_DEPLOYMENT_INSTANCE_SHAPE_CONFIG_DETAILS = (
        "modelDeploymentInstanceShapeConfigDetails"
    )
    CONST_OCPUS = "ocpus"
    CONST_MEMORY_IN_GBS = "memoryInGBs"
    CONST_SCALING_POLICY = "scalingPolicy"
    CONST_POLICY_TYPE = "policyType"
    CONST_REPLICA = "replica"
    CONST_INSTANCE_COUNT = "instanceCount"
    CONST_BANDWIDTH_MBPS = "bandwidthMbps"
    CONST_CATEGORY_LOG_DETAILS = "categoryLogDetails"
    CONST_ACCESS_LOG = "accessLog"
    CONST_ACCESS = "access"
    CONST_PREDICT_LOG = "predictLog"
    CONST_PREDICT = "predict"
    CONST_LOG_ID = "logId"
    CONST_LOG_GROUP_ID = "logGroupId"
    CONST_WEB_CONCURRENCY = "webConcurrency"
    CONST_STREAM_CONFIG_DETAILS = "streamConfigurationDetails"
    CONST_SUBNET_ID = "subnetId"

    attribute_map = {
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_SHAPE_NAME: "shape_name",
        CONST_SHAPE_CONFIG_DETAILS: "shape_config_details",
        CONST_OCPUS: "ocpus",
        CONST_MEMORY_IN_GBS: "memory_in_gbs",
        CONST_REPLICA: "replica",
        CONST_BANDWIDTH_MBPS: "bandwidth_mbps",
        CONST_ACCESS_LOG: "access_log",
        CONST_PREDICT_LOG: "predict_log",
        CONST_LOG_ID: "log_id",
        CONST_LOG_GROUP_ID: "log_group_id",
        CONST_WEB_CONCURRENCY: "web_concurrency",
        CONST_SUBNET_ID: "subnet_id",
    }

    shape_config_details_attribute_map = {
        CONST_MEMORY_IN_GBS: "memory_in_gbs",
        CONST_OCPUS: "ocpus",
    }

    access_log_attribute_map = {
        CONST_LOG_ID: "log_id",
        CONST_LOG_GROUP_ID: "log_group_id",
    }

    predict_log_attribute_map = {
        CONST_LOG_ID: "log_id",
        CONST_LOG_GROUP_ID: "log_group_id",
    }

    MODEL_CONFIG_DETAILS_PATH = (
        "model_deployment_configuration_details.model_configuration_details"
    )

    payload_attribute_map = {
        CONST_PROJECT_ID: "project_id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_SHAPE_NAME: f"{MODEL_CONFIG_DETAILS_PATH}.instance_configuration.instance_shape_name",
        CONST_SHAPE_CONFIG_DETAILS: f"{MODEL_CONFIG_DETAILS_PATH}.instance_configuration.model_deployment_instance_shape_config_details",
        CONST_SUBNET_ID: f"{MODEL_CONFIG_DETAILS_PATH}.instance_configuration.subnet_id",
        CONST_REPLICA: f"{MODEL_CONFIG_DETAILS_PATH}.scaling_policy.instance_count",
        CONST_BANDWIDTH_MBPS: f"{MODEL_CONFIG_DETAILS_PATH}.bandwidth_mbps",
        CONST_ACCESS_LOG: "category_log_details.access",
        CONST_PREDICT_LOG: "category_log_details.predict",
        CONST_DEPLOYMENT_TYPE: "model_deployment_configuration_details.deployment_type",
        CONST_POLICY_TYPE: f"{MODEL_CONFIG_DETAILS_PATH}.scaling_policy.policy_type",
    }

    sub_level_attribute_maps = {
        CONST_SHAPE_CONFIG_DETAILS: shape_config_details_attribute_map,
        CONST_ACCESS_LOG: access_log_attribute_map,
        CONST_PREDICT_LOG: predict_log_attribute_map,
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        super().__init__(spec, **kwargs)

    def _load_default_properties(self) -> Dict:
        """Load default properties from environment variables, notebook session, etc.

        Returns
        -------
        Dict
            A dictionary of default properties.
        """
        defaults = super()._load_default_properties()
        if COMPARTMENT_OCID:
            defaults[self.CONST_COMPARTMENT_ID] = COMPARTMENT_OCID
        if PROJECT_OCID:
            defaults[self.CONST_PROJECT_ID] = PROJECT_OCID

        defaults[self.CONST_BANDWIDTH_MBPS] = DEFAULT_BANDWIDTH_MBPS
        defaults[self.CONST_REPLICA] = DEFAULT_REPLICA

        if NB_SESSION_OCID:
            nb_session = None
            try:
                nb_session = DSCNotebookSession.from_ocid(NB_SESSION_OCID)
            except Exception as e:
                logger.warning(
                    f"Error fetching details about Notebook "
                    f"session: {NB_SESSION_OCID}. {e}"
                )
                logger.debug(traceback.format_exc())

            nb_config = getattr(
                nb_session, "notebook_session_config_details", None
            ) or getattr(nb_session, "notebook_session_configuration_details", None)
            if nb_config:
                defaults[self.CONST_SHAPE_NAME] = nb_config.shape

                if nb_config.notebook_session_shape_config_details:
                    notebook_shape_config_details = oci_util.to_dict(
                        nb_config.notebook_session_shape_config_details
                    )
                    defaults[self.CONST_SHAPE_CONFIG_DETAILS] = copy.deepcopy(
                        notebook_shape_config_details
                    )

        return defaults

    @property
    def kind(self) -> str:
        """The kind of the object as showing in YAML.

        Returns
        -------
        str
            infrastructure
        """
        return MODEL_DEPLOYMENT_INFRASTRUCTURE_KIND

    @property
    def type(self) -> str:
        """The type of the object as showing in YAML.

        Returns
        -------
        str
            datascienceModelDeployment
        """
        return MODEL_DEPLOYMENT_INFRASTRUCTURE_TYPE

    @property
    def compartment_id(self) -> str:
        """The model deployment compartment id.

        Returns
        -------
        str
            The model deployment compartment id.
        """
        return self.get_spec(self.CONST_COMPARTMENT_ID, None)

    def with_compartment_id(
        self, compartment_id: str
    ) -> "ModelDeploymentInfrastructure":
        """Sets the compartment id of model deployment.

        Parameters
        ----------
        compartment_id: str
            The compartment id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def project_id(self) -> str:
        """The model deployment project id.

        Returns
        -------
        str
            The model deployment project id.
        """
        return self.get_spec(self.CONST_PROJECT_ID, None)

    def with_project_id(self, project_id: str) -> "ModelDeploymentInfrastructure":
        """Sets the project id of model deployment.

        Parameters
        ----------
        project_id: str
            The project id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_PROJECT_ID, project_id)

    @property
    def shape_config_details(self) -> Dict:
        """The model deployment shape config details.

        Returns
        -------
        Dict
            The model deployment shape config details.
        """
        return self.get_spec(self.CONST_SHAPE_CONFIG_DETAILS, {})

    def with_shape_config_details(
        self, ocpus: float, memory_in_gbs: float, **kwargs: Dict[str, Any]
    ) -> "ModelDeploymentInfrastructure":
        """Sets the shape config details of model deployment.

        Parameters
        ----------
        ocpus: float
            The ocpus number.
        memory_in_gbs: float
            The memory in gbs number.
        kwargs: Dict

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(
            self.CONST_SHAPE_CONFIG_DETAILS,
            {
                self.CONST_OCPUS: ocpus,
                self.CONST_MEMORY_IN_GBS: memory_in_gbs,
                **kwargs,
            },
        )

    @property
    def shape_name(self) -> str:
        """The model deployment shape name.

        Returns
        -------
        str
            The model deployment shape name.
        """
        return self.get_spec(self.CONST_SHAPE_NAME, None)

    def with_shape_name(self, shape_name: str) -> "ModelDeploymentInfrastructure":
        """Sets the shape name of model deployment.

        Parameters
        ----------
        shape_name: str
            The shape name of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_SHAPE_NAME, shape_name)

    @property
    def replica(self) -> int:
        """The model deployment instance count.

        Returns
        -------
        int
            The model deployment instance count.
        """
        return self.get_spec(self.CONST_REPLICA, None)

    def with_replica(self, replica: int) -> "ModelDeploymentInfrastructure":
        """Sets the instance count of model deployment.

        Parameters
        ----------
        replica: int
            The instance count of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_REPLICA, replica)

    @property
    def bandwidth_mbps(self) -> int:
        """The model deployment bandwidth in mbps.

        Returns
        -------
        int
            The model deployment bandwidth in mbps.
        """
        return self.get_spec(self.CONST_BANDWIDTH_MBPS, None)

    def with_bandwidth_mbps(
        self, bandwidth_mbps: int
    ) -> "ModelDeploymentInfrastructure":
        """Sets the bandwidth of model deployment.

        Parameters
        ----------
        bandwidth_mbps: int
            The bandwidth of model deployment in mbps.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_BANDWIDTH_MBPS, bandwidth_mbps)

    @property
    def access_log(self) -> Dict:
        """The model deployment access log.

        Returns
        -------
        Dict
            The model deployment access log.
        """
        return self.get_spec(self.CONST_ACCESS_LOG, {})

    def with_access_log(
        self, log_group_id: str, log_id: str
    ) -> "ModelDeploymentInfrastructure":
        """Sets the access log of model deployment.

        Parameters
        ----------
        log_group_id: str
            The access log group id of model deployment.
        log_id: str
            The access log id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(
            self.CONST_ACCESS_LOG,
            {self.CONST_LOG_GROUP_ID: log_group_id, self.CONST_LOG_ID: log_id},
        )

    @property
    def predict_log(self) -> Dict:
        """The model deployment predict log.

        Returns
        -------
        Dict
            The model deployment predict log.
        """
        return self.get_spec(self.CONST_PREDICT_LOG, {})

    def with_predict_log(
        self, log_group_id: str, log_id: str
    ) -> "ModelDeploymentInfrastructure":
        """Sets the predict log of model deployment.

        Parameters
        ----------
        log_group_id: str
            The predict log group id of model deployment.
        log_id: str
            The predict log id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(
            self.CONST_PREDICT_LOG,
            {self.CONST_LOG_GROUP_ID: log_group_id, self.CONST_LOG_ID: log_id},
        )

    @property
    def log_group_id(self) -> str:
        """The model deployment log group id.

        Returns
        -------
        str
            The model deployment log group id.
        """
        return self.get_spec(self.CONST_LOG_GROUP_ID, None)

    def with_log_group_id(self, log_group_id: str) -> "ModelDeploymentInfrastructure":
        """Sets the log group id of model deployment.

        Parameters
        ----------
        log_group_id: str
            The predict and access log group id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_LOG_GROUP_ID, log_group_id)

    @property
    def log_id(self) -> str:
        """The model deployment log id.

        Returns
        -------
        str
            The model deployment log id.
        """
        return self.get_spec(self.CONST_LOG_ID, None)

    def with_log_id(self, log_id: str) -> "ModelDeploymentInfrastructure":
        """Sets the log id of model deployment.

        Parameters
        ----------
        log_id: str
            The predict and access log id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_LOG_ID, log_id)

    @property
    def web_concurrency(self) -> int:
        """The model deployment web concurrency.

        Returns
        -------
        int
            The model deployment web concurrency.
        """
        return self.get_spec(self.CONST_WEB_CONCURRENCY, None)

    def with_web_concurrency(
        self, web_concurrency: int
    ) -> "ModelDeploymentInfrastructure":
        """Sets the web concurrency of model deployment.

        Parameters
        ----------
        web_concurrency: int
            The web concurrency of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_WEB_CONCURRENCY, web_concurrency)

    def with_subnet_id(self, subnet_id: str) -> "ModelDeploymentInfrastructure":
        """Sets the subnet id of model deployment.

        Parameters
        ----------
        subnet_id : str
            The subnet id of model deployment.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self).
        """
        return self.set_spec(self.CONST_SUBNET_ID, subnet_id)

    @property
    def subnet_id(self) -> str:
        """The model deployment subnet id.

        Returns
        -------
        str
            The model deployment subnet id.
        """
        return self.get_spec(self.CONST_SUBNET_ID, None)

    def init(self, **kwargs) -> "ModelDeploymentInfrastructure":
        """Initializes a starter specification for the ModelDeploymentInfrastructure.

        Returns
        -------
        ModelDeploymentInfrastructure
            The ModelDeploymentInfrastructure instance (self)
        """
        return (
            self.build()
            .with_compartment_id(self.compartment_id or "{Provide a compartment OCID}")
            .with_project_id(self.project_id or "{Provide a project OCID}")
            .with_bandwidth_mbps(self.bandwidth_mbps or DEFAULT_BANDWIDTH_MBPS)
            .with_replica(self.replica or DEFAULT_REPLICA)
            .with_shape_name(self.shape_name or DEFAULT_SHAPE_NAME)
            .with_shape_config_details(
                ocpus=self.shape_config_details.get(self.CONST_OCPUS, DEFAULT_OCPUS),
                memory_in_gbs=self.shape_config_details.get(
                    self.CONST_MEMORY_IN_GBS, DEFAULT_MEMORY_IN_GBS
                ),
            )
        )

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse
from tornado.web import HTTPError
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.deployment import AquaDeploymentApp
from ads.config import PROJECT_OCID


class AquaDeploymentHandler(AquaAPIhandler):
    """
    Handler for Aqua Deployment REST APIs.

    Methods
    -------
    get(self, id="")
        Retrieves a list of AQUA deployments or model info or logs by ID.
    post(self, *args, **kwargs)
        Creates a new AQUA deployment.
    read(self, id: str)
        Reads the AQUA deployment information.
    list(self)
        Lists all the AQUA deployments.
    get_deployment_config(self, model_id)
        Gets the deployment config for Aqua model.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    def get(self, id=""):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/deployments/config"):
            if not id:
                raise HTTPError(
                    400, f"The request {self.request.path} requires model id."
                )
            return self.get_deployment_config(id)
        elif paths.startswith("aqua/deployments"):
            if not id:
                return self.list()
            return self.read(id)
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    def post(self, *args, **kwargs):
        """
        Handles post request for the deployment APIs
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        # required input parameters
        compartment_id = input_data.get("compartment_id")
        if not compartment_id:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("compartment_id")
            )
        display_name = input_data.get("display_name")
        if not display_name:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("display_name")
            )
        instance_count = input_data.get("instance_count")
        if not instance_count:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("instance_count")
            )
        instance_shape = input_data.get("instance_shape")
        if not instance_count:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("instance_shape")
            )
        log_group_id = input_data.get("log_group_id")
        if not log_group_id:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("log_group_id")
            )
        access_log_id = input_data.get("access_log_id")
        if not access_log_id:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("access_log_id")
            )
        predict_log_id = input_data.get("predict_log_id")
        if not predict_log_id:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("predict_log_id")
            )

        model_id = input_data.get("model_id")
        if not instance_count:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model_id"))

        project_id = input_data.get("project_id", PROJECT_OCID)
        description = input_data.get("description")
        bandwidth_mbps = input_data.get("bandwidth_mbps", 10)

        try:
            self.finish(
                AquaDeploymentApp().create(
                    compartment_id=compartment_id,
                    project_id=project_id,
                    model_id=model_id,
                    display_name=display_name,
                    description=description,
                    instance_count=instance_count,
                    instance_shape=instance_shape,
                    log_group_id=log_group_id,
                    access_log_id=access_log_id,
                    predict_log_id=predict_log_id,
                    bandwidth_mbps=bandwidth_mbps,
                )
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def read(self, id):
        """Read the information of an Aqua model deployment."""
        try:
            return self.finish(AquaDeploymentApp().get(model_deployment_id=id))
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id")
        if not compartment_id:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("compartment_id")
            )
        # project_id is optional.
        project_id = self.get_argument("project_id", default=PROJECT_OCID)
        try:
            # todo: update below after list is implemented
            return self.finish(
                AquaDeploymentApp().list(
                    compartment_id=compartment_id, project_id=project_id
                )
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def get_deployment_config(self, model_id):
        """Gets the deployment config for Aqua model."""
        try:
            return self.finish(
                AquaDeploymentApp().get_deployment_config(model_id=model_id)
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

__handlers__ = [
    ("deployments/?([^/]*)", AquaDeploymentHandler)
    ("deployments/config/?([^/]*)", AquaDeploymentHandler)
]

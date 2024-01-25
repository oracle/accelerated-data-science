#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from tornado.web import HTTPError
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.deployment import AquaDeploymentApp


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

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    def get(self, id=""):
        """Handle GET request."""
        # todo: handle list, read and logs for model deployment
        pass

    def post(self, **kwargs):
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

        try:
            # todo: identify optional params
            compartment_id = input_data.get("compartment_id")
            project_id = input_data.get("project_id", None)
            model_id = input_data.get("model_id")
            display_name = input_data.get("display_name")
            description = input_data.get("description")
            instance_count = input_data.get("instance_count")
            instance_shape = input_data.get("instance_shape")
            access_log_group_id = input_data.get("access_log_group_id")
            access_log_id = input_data.get("access_log_id")
            predict_log_group_id = input_data.get("predict_log_group_id")
            predict_log_id = input_data.get("predict_log_id")
            bandwidth = input_data.get("bandwidth")
        except Exception as ex:
            # todo: handle missing parameter error
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format(ex))

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
                    access_log_group_id=access_log_group_id,
                    access_log_id=access_log_id,
                    predict_log_group_id=predict_log_group_id,
                    predict_log_id=predict_log_id,
                    bandwidth=bandwidth,
                    wait_for_completion=False,
                )
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id")
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaDeploymentApp().list(compartment_id, project_id))


__handlers__ = [("deployments/?([^/]*)", AquaDeploymentHandler)]

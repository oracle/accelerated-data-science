#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from tornado.web import HTTPError
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.deployment import AquaDeploymentApp


class AquaDeploymentHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    def get(self, model_id=""):
        """Handle GET request."""
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
            model_id = input_data.get("model_id")
            compartment_id = input_data.get("compartment_id")
            project_id = input_data.get("project_id", None)
            display_name = input_data.get("display_name")
            description = input_data.get("description")
            instance_count = input_data.get("instance_count")
            shape = input_data.get("shape")
            access_log_group_id = input_data.get("access_log_group_id")
            access_log_id = input_data.get("access_log_id")
            predict_log_group_id = input_data.get("predict_log_group_id")
            predict_log_id = input_data.get("predict_log_id")
            bandwidth = input_data.get("bandwidth")
        except Exception as ex:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format(ex))

        try:
            self.finish(
                AquaDeploymentApp().create(
                    model_id=model_id,
                    compartment_id=compartment_id,
                    project_id=project_id,
                    display_name=display_name,
                    description=description,
                    instance_count=instance_count,
                    shape=shape,
                    access_log_group_id=access_log_group_id,
                    access_log_id=access_log_id,
                    predict_log_group_id=predict_log_group_id,
                    predict_log_id=predict_log_id,
                    bandwidth=bandwidth,
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.deployment import AquaDeploymentApp, MDInferenceResponse, ModelParams
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.aqua.decorator import handle_exceptions


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

    @handle_exceptions
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

    @handle_exceptions
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
        display_name = input_data.get("display_name")
        if not display_name:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("display_name")
            )
        instance_shape = input_data.get("instance_shape")
        if not instance_shape:
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format("instance_shape")
            )
        model_id = input_data.get("model_id")
        if not model_id:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model_id"))

        compartment_id = input_data.get("compartment_id", COMPARTMENT_OCID)
        project_id = input_data.get("project_id", PROJECT_OCID)
        log_group_id = input_data.get("log_group_id")
        access_log_id = input_data.get("access_log_id")
        predict_log_id = input_data.get("predict_log_id")
        description = input_data.get("description")
        instance_count = input_data.get("instance_count")
        bandwidth_mbps = input_data.get("bandwidth_mbps")

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

    @handle_exceptions
    def read(self, id):
        """Read the information of an Aqua model deployment."""
        return self.finish(AquaDeploymentApp().get(model_deployment_id=id))

    @handle_exceptions
    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(
            AquaDeploymentApp().list(
                compartment_id=compartment_id, project_id=project_id
            )
        )

    @handle_exceptions
    def get_deployment_config(self, model_id):
        """Gets the deployment config for Aqua model."""
        return self.finish(AquaDeploymentApp().get_deployment_config(model_id=model_id))


class AquaDeploymentInferenceHandler(AquaAPIhandler):
    @staticmethod
    def validate_predict_url(endpoint):
        try:
            url = urlparse(endpoint)
            if url.scheme != "https":
                return False
            if not url.netloc:
                return False
            if not url.path.endswith("/predict"):
                return False
            return True
        except Exception:
            return False

    @handle_exceptions
    def post(self, *args, **kwargs):
        """
        Handles inference request for the Active Model Deployments
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

        endpoint = input_data.get("endpoint")
        if not endpoint:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("endpoint"))

        if not self.validate_predict_url(endpoint):
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT.format("endpoint"))

        prompt = input_data.get("prompt")
        if not prompt:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("prompt"))

        model_params = (
            input_data.get("model_params") if input_data.get("model_params") else {}
        )
        try:
            model_params_obj = ModelParams(**model_params)
        except:
            raise HTTPError(
                400, Errors.INVALID_INPUT_DATA_FORMAT.format("model_params")
            )

        return self.finish(
            MDInferenceResponse(prompt, model_params_obj).get_model_deployment_response(
                endpoint
            )
        )


__handlers__ = [
    ("deployments/?([^/]*)", AquaDeploymentHandler),
    ("deployments/config/?([^/]*)", AquaDeploymentHandler),
    ("inference", AquaDeploymentInferenceHandler),
]

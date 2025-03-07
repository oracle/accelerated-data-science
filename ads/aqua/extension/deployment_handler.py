#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.modeldeployment import AquaDeploymentApp, MDInferenceResponse
from ads.aqua.modeldeployment.entities import ModelParams
from ads.config import COMPARTMENT_OCID, PROJECT_OCID


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
        web_concurrency = input_data.get("web_concurrency")
        server_port = input_data.get("server_port")
        health_check_port = input_data.get("health_check_port")
        env_var = input_data.get("env_var")
        container_family = input_data.get("container_family")

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
                web_concurrency=web_concurrency,
                server_port=server_port,
                health_check_port=health_check_port,
                env_var=env_var,
                container_family=container_family,
            )
        )

    def read(self, id):
        """Read the information of an Aqua model deployment."""
        return self.finish(AquaDeploymentApp().get(model_deployment_id=id))

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


class AquaDeploymentParamsHandler(AquaAPIhandler):
    """Handler for Aqua deployment params REST APIs.

    Methods
    -------
    get(self, model_id)
        Retrieves a list of model deployment parameters.
    post(self, *args, **kwargs)
        Validates parameters for the given model id.
    """

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""
        instance_shape = self.get_argument("instance_shape")
        return self.finish(
            AquaDeploymentApp().get_deployment_default_params(
                model_id=model_id, instance_shape=instance_shape
            )
        )

    @handle_exceptions
    def post(self, *args, **kwargs):
        """Handles post request for the deployment param handler API.

        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid.
        """
        try:
            input_data = self.get_json_body()
        except Exception:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT)

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        model_id = input_data.get("model_id")
        if not model_id:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model_id"))

        params = input_data.get("params")
        container_family = input_data.get("container_family")
        return self.finish(
            AquaDeploymentApp().validate_deployment_params(
                model_id=model_id,
                params=params,
                container_family=container_family,
            )
        )


__handlers__ = [
    ("deployments/?([^/]*)/params", AquaDeploymentParamsHandler),
    ("deployments/config/?([^/]*)", AquaDeploymentHandler),
    ("deployments/?([^/]*)", AquaDeploymentHandler),
    ("inference", AquaDeploymentInferenceHandler),
]

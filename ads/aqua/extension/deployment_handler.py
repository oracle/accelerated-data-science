#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List, Union
from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.modeldeployment import AquaDeploymentApp, MDInferenceResponse
from ads.aqua.modeldeployment.entities import ModelParams
from ads.config import COMPARTMENT_OCID


class AquaDeploymentHandler(AquaAPIhandler):
    """
    Handler for Aqua Deployment REST APIs.

    Methods
    -------
    get(self, id: Union[str, List[str]])
        Retrieves a list of AQUA deployments or model info or logs by ID.
    post(self, *args, **kwargs)
        Creates a new AQUA deployment.
    read(self, id: str)
        Reads the AQUA deployment information.
    list(self)
        Lists all the AQUA deployments.
    get_deployment_config(self, model_id)
        Gets the deployment config for Aqua model.
    list_shapes(self)
        Lists the valid model deployment shapes.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    @handle_exceptions
    def get(self, id: Union[str, List[str]] = None):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/deployments/config"):
            if not id or not isinstance(id, str):
                raise HTTPError(
                    400,
                    f"Invalid request format for {self.request.path}. "
                    "Expected a single model ID or a comma-separated list of model IDs.",
                )
            id = id.replace(" ", "")
            return self.get_deployment_config(
                model_id=id.split(",") if "," in id else id
            )
        elif paths.startswith("aqua/deployments/shapes"):
            return self.list_shapes()
        elif paths.startswith("aqua/deployments"):
            if not id:
                return self.list()
            return self.read(id)
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def delete(self, model_deployment_id):
        return self.finish(AquaDeploymentApp().delete(model_deployment_id))

    @handle_exceptions
    def put(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles put request for the activating and deactivating OCI datascience model deployments
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/").split("/")
        if len(paths) != 4 or paths[0] != "aqua" or paths[1] != "deployments":
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

        model_deployment_id = paths[2]
        action = paths[3]
        if action == "activate":
            return self.finish(AquaDeploymentApp().activate(model_deployment_id))
        elif action == "deactivate":
            return self.finish(AquaDeploymentApp().deactivate(model_deployment_id))
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles post request for the deployment APIs
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

        if not input_data:
            raise HTTPError(400, Errors.NO_INPUT_DATA)

        self.finish(AquaDeploymentApp().create(**input_data))

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

    def get_deployment_config(self, model_id: Union[str, List[str]]):
        """
        Retrieves the deployment configuration for one or more Aqua models.

        Parameters
        ----------
        model_id : Union[str, List[str]]
            A single model ID (str) or a list of model IDs (List[str]).

        Returns
        -------
        None
            The function sends the deployment configuration as a response.
        """
        app = AquaDeploymentApp()

        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)

        if isinstance(model_id, list):
            # Handle multiple model deployment
            primary_model_id = self.get_argument("primary_model_id", default=None)
            deployment_config = app.get_multimodel_deployment_config(
                model_ids=model_id,
                primary_model_id=primary_model_id,
                compartment_id=compartment_id,
            )
        else:
            # Handle single model deployment
            deployment_config = app.get_deployment_config(model_id=model_id)

        return self.finish(deployment_config)

    def list_shapes(self):
        """
        Lists the valid model deployment shapes.

        Returns
        -------
        List[ComputeShapeSummary]:
            The list of the model deployment shapes.
        """
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)

        return self.finish(
            AquaDeploymentApp().list_shapes(compartment_id=compartment_id)
        )


class AquaDeploymentInferenceHandler(AquaAPIhandler):
    @staticmethod
    def validate_predict_url(endpoint):
        try:
            url = urlparse(endpoint)
            if url.scheme != "https":
                return False
            if not url.netloc:
                return False
            return url.path.endswith("/predict")
        except Exception:
            return False

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """
        Handles inference request for the Active Model Deployments
        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

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
        except Exception as ex:
            raise HTTPError(
                400, Errors.INVALID_INPUT_DATA_FORMAT.format("model_params")
            ) from ex

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
        gpu_count = self.get_argument("gpu_count", default=None)
        return self.finish(
            AquaDeploymentApp().get_deployment_default_params(
                model_id=model_id, instance_shape=instance_shape, gpu_count=gpu_count
            )
        )

    @handle_exceptions
    def post(self, *args, **kwargs):  # noqa: ARG002
        """Handles post request for the deployment param handler API.

        Raises
        ------
        HTTPError
            Raises HTTPError if inputs are missing or are invalid.
        """
        try:
            input_data = self.get_json_body()
        except Exception as ex:
            raise HTTPError(400, Errors.INVALID_INPUT_DATA_FORMAT) from ex

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
    ("deployments/shapes/?([^/]*)", AquaDeploymentHandler),
    ("deployments/?([^/]*)", AquaDeploymentHandler),
    ("deployments/?([^/]*)/activate", AquaDeploymentHandler),
    ("deployments/?([^/]*)/deactivate", AquaDeploymentHandler),
    ("inference", AquaDeploymentInferenceHandler),
]

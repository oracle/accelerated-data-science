#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.common.errors import AquaValueError
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.errors import Errors
from ads.aqua.model import AquaModelApp
from ads.aqua.ui import ModelFormat


class AquaModelHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    @handle_exceptions
    def get(
        self,
        model_id="",
    ):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/model/files"):
            os_path = self.get_argument("os_path")
            if not os_path:
                raise HTTPError(
                    400, Errors.MISSING_REQUIRED_PARAMETER.format("os_path")
                )
            model_format = self.get_argument("model_format")
            if not model_format:
                raise HTTPError(
                    400, Errors.MISSING_REQUIRED_PARAMETER.format("model_format")
                )
            try:
                model_format = ModelFormat(model_format.upper())
            except ValueError:
                raise AquaValueError(f"Invalid model format: {model_format}")
            else:
                return self.finish(AquaModelApp.get_model_files(os_path, model_format))
        elif not model_id:
            return self.list()

        return self.read(model_id)

    def read(self, model_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaModelApp().get(model_id))

    @handle_exceptions
    def delete(self, id=""):
        """Handles DELETE request for clearing cache"""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/model/cache"):
            return self.finish(AquaModelApp().clear_model_list_cache())
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=None)
        # project_id is no needed.
        project_id = self.get_argument("project_id", default=None)
        model_type = self.get_argument("model_type", default=None)
        return self.finish(
            AquaModelApp().list(
                compartment_id=compartment_id,
                project_id=project_id,
                model_type=model_type,
            )
        )

    @handle_exceptions
    def post(self, *args, **kwargs):
        """
        Handles post request for the registering any Aqua model.
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
        model = input_data.get("model")
        if not model:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("model"))
        os_path = input_data.get("os_path")
        if not os_path:
            raise HTTPError(400, Errors.MISSING_REQUIRED_PARAMETER.format("os_path"))

        inference_container = input_data.get("inference_container")
        finetuning_container = input_data.get("finetuning_container")
        compartment_id = input_data.get("compartment_id")
        project_id = input_data.get("project_id")
        model_file = input_data.get("model_file")

        return self.finish(
            AquaModelApp().register(
                model=model,
                os_path=os_path,
                inference_container=inference_container,
                finetuning_container=finetuning_container,
                compartment_id=compartment_id,
                project_id=project_id,
                model_file=model_file,
            )
        )


class AquaModelLicenseHandler(AquaAPIhandler):
    """Handler for Aqua Model license REST APIs."""

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""

        model_id = model_id.split("/")[0]
        return self.finish(AquaModelApp().load_license(model_id))


__handlers__ = [
    ("model/?([^/]*)", AquaModelHandler),
    ("model/?([^/]*)/license", AquaModelLicenseHandler),
]

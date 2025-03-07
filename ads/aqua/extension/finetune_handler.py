#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.common.decorator import handle_exceptions
from ads.aqua.extension.errors import Errors
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.extension.utils import validate_function_parameters
from ads.aqua.finetuning import AquaFineTuningApp
from ads.aqua.finetuning.entities import CreateFineTuningDetails


class AquaFineTuneHandler(AquaAPIhandler):
    """Handler for Aqua fine-tuning job REST APIs."""

    @handle_exceptions
    def get(self, id=""):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/finetuning/config"):
            if not id:
                raise HTTPError(
                    400, f"The request {self.request.path} requires model id."
                )
            return self.get_finetuning_config(id)
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def post(self, *args, **kwargs):
        """Handles post request for the fine-tuning API

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

        validate_function_parameters(
            data_class=CreateFineTuningDetails, input_data=input_data
        )

        self.finish(AquaFineTuningApp().create(CreateFineTuningDetails(**input_data)))

    def get_finetuning_config(self, model_id):
        """Gets the finetuning config for Aqua model."""
        return self.finish(AquaFineTuningApp().get_finetuning_config(model_id=model_id))


class AquaFineTuneParamsHandler(AquaAPIhandler):
    """Handler for Aqua finetuning params REST APIs."""

    @handle_exceptions
    def get(self, model_id):
        """Handle GET request."""
        return self.finish(
            AquaFineTuningApp().get_finetuning_default_params(model_id=model_id)
        )

    @handle_exceptions
    def post(self, *args, **kwargs):
        """Handles post request for the finetuning param handler API.

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

        params = input_data.get("params", None)
        return self.finish(
            AquaFineTuningApp().validate_finetuning_params(
                params=params,
            )
        )


__handlers__ = [
    ("finetuning/?([^/]*)/params", AquaFineTuneParamsHandler),
    ("finetuning/?([^/]*)", AquaFineTuneHandler),
    ("finetuning/config/?([^/]*)", AquaFineTuneHandler),
]

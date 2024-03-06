#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from tornado.web import HTTPError
from ads.aqua.exception import AquaError
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.extension.utils import validate_function_parameters
from ads.aqua.finetune import AquaFineTuningApp, CreateFineTuningDetails
from ads.aqua.decorator import handle_exceptions


class AquaFineTuneHandler(AquaAPIhandler):
    """Handler for Aqua fine-tuning job REST APIs."""

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

        self.finish(
            AquaFineTuningApp().create(
                CreateFineTuningDetails(**input_data)
            )
        )


__handlers__ = [("finetuning/?([^/]*)", AquaFineTuneHandler)]

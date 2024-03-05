#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from tornado.web import HTTPError
from ads.aqua.extension.base_handler import AquaAPIhandler, Errors
from ads.aqua.finetune import AquaFineTuningApp
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

        # todo: create CreateAquaFineTuningDetails class and validate inputs
        # validate_function_parameters(
        #     data_class=CreateAquaFineTuningDetails, input_data=input_data
        # )
        try:
            self.finish(AquaFineTuningApp().create(**input_data))
        except Exception as ex:
            raise HTTPError(500, str(ex))


__handlers__ = [("finetuning/?([^/]*)", AquaFineTuneHandler)]

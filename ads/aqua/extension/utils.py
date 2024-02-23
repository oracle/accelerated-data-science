#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import inspect
from typing import Dict
from requests import HTTPError

from ads.aqua.extension.base_handler import Errors

SKIPPED_PARAMETERS = ["self", "kwargs"]


def validate_function_parameters(function, input_data: Dict):
    """Validates if the required parameters are provided in input data."""
    function_signature = inspect.signature(function)
    
    required_parameters = [
        parameter.name for parameter in 
        function_signature.parameters.values()
        if (
            parameter.name not in SKIPPED_PARAMETERS
            and parameter.default is parameter.empty
        )
    ]

    for required_parameter in required_parameters:
        if not input_data.get(required_parameter):
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format(required_parameter)
            )

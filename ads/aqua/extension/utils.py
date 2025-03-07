#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from dataclasses import fields
from typing import Dict, Optional

from tornado.web import HTTPError

from ads.aqua.extension.errors import Errors


def validate_function_parameters(data_class, input_data: Dict):
    """Validates if the required parameters are provided in input data."""
    required_parameters = [
        field.name for field in fields(data_class) if field.type != Optional[field.type]
    ]

    for required_parameter in required_parameters:
        if not input_data.get(required_parameter):
            raise HTTPError(
                400, Errors.MISSING_REQUIRED_PARAMETER.format(required_parameter)
            )

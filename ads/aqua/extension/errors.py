#!/usr/bin/env python
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class Errors(str):
    INVALID_INPUT_DATA_FORMAT = "Invalid format of input data."
    NO_INPUT_DATA = "No input data provided."
    MISSING_REQUIRED_PARAMETER = "Missing required parameter: '{}'"
    MISSING_ONEOF_REQUIRED_PARAMETER = "Either '{}' or '{}' is required."
    INVALID_VALUE_OF_PARAMETER = "Invalid value of parameter: '{}'"

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator import __operators__
from ads.opctl.operator.common.errors import InvalidParameterError


class DataMismatchError(Exception):
    """Exception raised when there is an issue with the schema."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid operator specification. Check the YAML structure and ensure it "
            "complies with the required schema for the operator. \n"
            f"{error}"
        )


class InputDataError(Exception):
    """Exception raised when there is an issue with the input data."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid operator specification. Check the YAML structure and ensure it "
            "complies with the required schema for the operator. \n"
            f"{error}"
        )


class PermissionsError(Exception):
    """Exception raised when there is an issue with the input data."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid operator specification. Check the YAML structure and ensure it "
            "complies with the required schema for the operator. \n"
            f"{error}"
        )

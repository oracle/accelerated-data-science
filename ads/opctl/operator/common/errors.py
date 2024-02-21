#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator import __operators__


class InvalidParameterError(Exception):
    """Exception raised when there is an issue with the schema."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid operator specification. Check the YAML structure and ensure it "
            "complies with the required schema for the operator. \n"
            f"{error}"
        )


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The provided operator: `{operator}` is not found. You can pick up one from the "
            f"registered service operators: `{'`, `'.join(__operators__)}`."
        )


class OperatorImageNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The Docker image for the operator: `{operator}` nas not been built yet. "
            "Please ensure that you build the image before attempting to publish it. "
        )


class OperatorCondaNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The Conda environment for the operator: `{operator}` nas not been built yet. "
            "Please ensure that you build the conda environment before attempting to publish it. "
        )

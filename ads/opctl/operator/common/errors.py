#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.opctl.operator import __operators__


class OperatorNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The provided operator: `{operator}` "
            f"is not found. Available operators: {__operators__}"
        )


class OperatorImageNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The Docker image for the operator: `{operator}` nas not been built yet. "
            "Please ensure that you build the image before attempting to publish it. "
            f"Use the `ads opctl operator build-image --name {operator}` command "
            "to build the image."
        )


class OperatorCondaNotFoundError(Exception):
    def __init__(self, operator: str):
        super().__init__(
            f"The Conda environment for the operator: `{operator}` nas not been built yet. "
            "Please ensure that you build the conda environment before attempting to publish it. "
            f"Use the `ads opctl operator build-conda --name {operator}` "
            "command to build the conda environment."
        )

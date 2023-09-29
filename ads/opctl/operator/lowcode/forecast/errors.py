#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

class ForecastSchemaYamlError(Exception):
    """Exception raised when there is an issue with the schema."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid forecast operator specification. Check the YAML structure and ensure it "
            "complies with the required schema for forecast operator. \n"
            f"{error}"
        )


class ForecastInputDataError(Exception):
    """Exception raised when there is an issue with input data."""

    def __init__(self, error: str):
        super().__init__(
            "Invalid input data. Check the input data and ensure it "
            "complies with the validation criteria. \n"
            f"{error}"
        )

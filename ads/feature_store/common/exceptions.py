#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class NotMaterializedError(ValueError):
    """
    Exception raised when a resource is not materialized.

    Attributes:
        resource_name (str): The name of the resource that is not materialized.
    """

    def __init__(self, resource_type: str, resource_name: str):
        self.resource_name = resource_name
        super().__init__(f"{resource_type} {resource_name} is not materialized.")

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class DatasetError(BaseException):
    """Base class for dataset errors."""

    def __init__(self, *args, **kwargs):
        pass

    def __str__(self, *args, **kwargs):
        pass


class ValidationError(DatasetError):
    """Handles validation errors in dataset."""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self, *args, **kwargs):
        return self.msg

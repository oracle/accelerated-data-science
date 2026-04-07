#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod

from .base_model import RegressionOperatorBaseModel


class SharedRegressionOperatorModel(RegressionOperatorBaseModel):
    """Shared model template for regression models with a common pipeline flow."""

    @abstractmethod
    def _build_estimator(self):
        raise NotImplementedError

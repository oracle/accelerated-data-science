#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os

from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.model import AquaModelApp
from ads.aqua.exception import exception_handler

AQUA_MODEL_COMPARTMENT = "AQUA_MODEL_COMPARTMENT"


class AquaModelHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    @exception_handler
    def get(self, model_id=""):
        """Handle GET request."""

        if not model_id:
            return self.list()
        return self.read(model_id)

    @exception_handler
    def read(self, model_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaModelApp().get(model_id))

    # @exception_handler
    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        # default compartment will be stored in env var: AQUA_MODEL_COMPARTMENT
        # raise ValueError
        compartment_id = self.get_argument(
            "compartment_id", default=os.environ.get(AQUA_MODEL_COMPARTMENT)
        )
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        model_app = AquaModelApp()
        all_models = model_app.list(compartment_id, project_id)
        return self.finish(all_models)


__handlers__ = [("model/?([^/]*)", AquaModelHandler)]

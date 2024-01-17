#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.model import AquaModelApp


class AquaModelHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    def get(self, model_id=""):
        """Handle GET request."""
        if not model_id:
            return self.list()
        return self.read(model_id)

    def read(self, model_id):
        """Read the information of an Aqua model."""
        return self.finish(AquaModelApp().get(model_id))

    def list(self):
        """List Aqua models."""
        # If default is not specified,
        # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
        compartment_id = self.get_argument("compartment_id")
        # project_id is optional.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaModelApp().list(compartment_id, project_id))

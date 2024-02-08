#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.aqua.decorator import exception_handler
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.model import AquaModelApp


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

    @exception_handler
    def list(self):
        """List Aqua models."""
        compartment_id = self.get_argument("compartment_id", default=None)
        # project_id is no needed.
        project_id = self.get_argument("project_id", default=None)
        return self.finish(AquaModelApp().list(compartment_id, project_id))


__handlers__ = [("model/?([^/]*)", AquaModelHandler)]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.model import AquaModelApp
from ads.aqua.exception import AquaServiceError, AquaClientError
from tornado.web import HTTPError


class AquaModelHandler(AquaAPIhandler):
    """Handler for Aqua Model REST APIs."""

    def get(self, model_id=""):
        """Handle GET request."""
        try:
            if not model_id:
                return self.list()
            return self.read(model_id)
        except AquaServiceError as service_error:
            raise HTTPError(500, str(service_error))
        except AquaClientError as client_error:
            raise HTTPError(400, str(client_error))
        except Exception as internal_error:
            raise HTTPError(500, str(internal_error))

    def read(self, model_id):
        """Read the information of an Aqua model."""
        try:
            return self.finish(AquaModelApp().get(model_id))
        except AquaServiceError as service_error:
            raise HTTPError(500, str(service_error))
        except AquaClientError as client_error:
            raise HTTPError(400, str(client_error))
        except Exception as internal_error:
            raise HTTPError(500, str(internal_error))

    def list(self):
        """List Aqua models."""
        try:
            # If default is not specified,
            # jupyterlab will raise 400 error when argument is not provided by the HTTP request.
            compartment_id = self.get_argument("compartment_id")
            # project_id is optional.
            project_id = self.get_argument("project_id", default=None)
            return self.finish(AquaModelApp().list(compartment_id, project_id))
        except AquaServiceError as service_error:
            raise HTTPError(500, str(service_error))
        except AquaClientError as client_error:
            raise HTTPError(400, str(client_error))
        except Exception as internal_error:
            raise HTTPError(500, str(internal_error))


__handlers__ = [("model/?([^/]*)", AquaModelHandler)]

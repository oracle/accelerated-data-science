#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse

from tornado.web import HTTPError

from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.ui import AquaUIApp


class AquaUIHandler(AquaAPIhandler):
    """
    Handler for Aqua UI REST APIs.

    Methods
    -------
    get(self, id="")
        Routes the request to fetch log groups, log ids details or compartments
    list_log_groups(self, id: str)
        Reads the AQUA deployment information.
    list_logs(self, log_group_id: str, **kwargs)
        Lists the specified log group's log objects.
    list_compartments(self, **kwargs)
        Lists the compartments in a compartment specified by ODSC_MODEL_COMPARTMENT_OCID env variable.

    Raises
    ------
    HTTPError: For various failure scenarios such as invalid input format, missing data, etc.
    """

    def get(self, id=""):
        """Handle GET request."""
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/logging"):
            if not id:
                return self.list_log_groups()
            return self.list_logs(id)
        elif paths.startswith("aqua/compartments/default"):
            return self.get_default_compartment()
        elif paths.startswith("aqua/compartments"):
            return self.list_compartments()
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    def list_log_groups(self, **kwargs):
        """Lists all log groups for the specified compartment or tenancy."""
        compartment_id = self.get_argument("compartment_id")
        try:
            return self.finish(
                AquaUIApp().list_log_groups(compartment_id=compartment_id, **kwargs)
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list_logs(self, log_group_id: str, **kwargs):
        """Lists the specified log group's log objects."""
        try:
            return self.finish(
                AquaUIApp().list_logs(log_group_id=log_group_id, **kwargs)
            )
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def list_compartments(self, **kwargs):
        """Lists the compartments in a compartment specified by ODSC_MODEL_COMPARTMENT_OCID env variable."""
        try:
            return self.finish(AquaUIApp().list_compartments(**kwargs))
        except Exception as ex:
            raise HTTPError(500, str(ex))

    def get_default_compartment(self):
        """Returns user compartment ocid."""
        try:
            return self.finish(AquaUIApp().get_default_compartment())
        except Exception as ex:
            raise HTTPError(500, str(ex))


__handlers__ = [
    ("logging/?([^/]*)", AquaUIHandler),
    ("compartments/?([^/]*)", AquaUIHandler),
]

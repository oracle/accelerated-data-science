#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from urllib.parse import urlparse
from tornado.web import HTTPError
from ads.aqua.decorator import handle_exceptions
from ads.aqua.extension.base_handler import AquaAPIhandler
from ads.aqua.ui import AquaUIApp
from ads.config import COMPARTMENT_OCID


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
        elif paths.startswith("aqua/experiment"):
            return self.list_model_version_sets()
        elif paths.startswith("aqua/buckets"):
            return self.list_buckets()
        elif paths.startswith("aqua/job/shapes"):
            return self.list_job_shapes()
        elif paths.startswith("aqua/vcn"):
            return self.list_vcn()
        elif paths.startswith("aqua/subnets"):
            return self.list_subnets()
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def delete(self, id=""):
        """Handles DELETE request for clearing cache"""
        # todo: added for dev work, to be deleted if there's no feature to refresh cache in Aqua
        url_parse = urlparse(self.request.path)
        paths = url_parse.path.strip("/")
        if paths.startswith("aqua/compartments/cache"):
            return self.finish(AquaUIApp().clear_compartments_list_cache())
        else:
            raise HTTPError(400, f"The request {self.request.path} is invalid.")

    @handle_exceptions
    def list_log_groups(self, **kwargs):
        """Lists all log groups for the specified compartment or tenancy."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(
            AquaUIApp().list_log_groups(compartment_id=compartment_id, **kwargs)
        )

    @handle_exceptions
    def list_logs(self, log_group_id: str, **kwargs):
        """Lists the specified log group's log objects."""
        return self.finish(AquaUIApp().list_logs(log_group_id=log_group_id, **kwargs))

    @handle_exceptions
    def list_compartments(self):
        """Lists the compartments in a compartment specified by ODSC_MODEL_COMPARTMENT_OCID env variable."""
        return self.finish(AquaUIApp().list_compartments())

    @handle_exceptions
    def get_default_compartment(self):
        """Returns user compartment ocid."""
        return self.finish(AquaUIApp().get_default_compartment())

    @handle_exceptions
    def list_model_version_sets(self, **kwargs):
        """Lists all model version sets for the specified compartment or tenancy."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(
            AquaUIApp().list_model_version_sets(compartment_id=compartment_id, **kwargs)
        )

    @handle_exceptions
    def list_buckets(self, **kwargs):
        """Lists all model version sets for the specified compartment or tenancy."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(
            AquaUIApp().list_buckets(compartment_id=compartment_id, **kwargs)
        )

    @handle_exceptions
    def list_job_shapes(self, **kwargs):
        """Lists job shapes available in the specified compartment."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(
            AquaUIApp().list_job_shapes(compartment_id=compartment_id, **kwargs)
        )

    @handle_exceptions
    def list_vcn(self, **kwargs):
        """Lists the virtual cloud networks (VCNs) in the specified compartment."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        return self.finish(AquaUIApp.list_vcn(compartment_id=compartment_id, **kwargs))

    @handle_exceptions
    def list_subnets(self, **kwargs):
        """Lists the subnets in the specified VCN and the specified compartment."""
        compartment_id = self.get_argument("compartment_id", default=COMPARTMENT_OCID)
        vcn_id = self.get_argument("vcn_id")
        return self.finish(
            AquaUIApp.list_subnets(
                compartment_id=compartment_id, vcn_id=vcn_id, **kwargs
            )
        )


__handlers__ = [
    ("logging/?([^/]*)", AquaUIHandler),
    ("compartments/?([^/]*)", AquaUIHandler),
    ("experiment/?([^/]*)", AquaUIHandler),
    ("buckets/?([^/]*)", AquaUIHandler),
    ("job/shapes/?([^/]*)", AquaUIHandler),
    ("vcn/?([^/]*)", AquaUIHandler),
    ("subnets/?([^/]*)", AquaUIHandler),
]

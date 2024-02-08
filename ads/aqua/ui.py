#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict, List

from oci.exceptions import ServiceError

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaClientError, AquaServiceError
from ads.config import COMPARTMENT_OCID, TENANCY_OCID


class AquaUIApp(AquaApp):
    """Contains APIs for supporting Aqua UI.

    Attributes
    ----------

    Methods
    -------
    list_log_groups(self, **kwargs) -> List[Dict]
        Lists all log groups for the specified compartment or tenancy.
    list_logs(self, **kwargs) -> List[Dict]
        Lists the specified log group's log objects.
    list_compartments(self, **kwargs) -> List[Dict]
        Lists the compartments in a specified compartment.

    """

    def list_log_groups(self, **kwargs) -> List[Dict]:
        """Lists all log groups for the specified compartment or tenancy. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_log_groups <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_log_groups>`_

        Returns
        -------
        List[Dict]:
            Dict has json representation of oci.logging.models.log_group.LogGroup
        """

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        return self.logging_client.list_log_groups(
            compartment_id=compartment_id, **kwargs
        ).data.__repr__()

    def list_logs(self, **kwargs) -> List[Dict]:
        """Lists the specified log group's log objects. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as log_group_id, log_type
            for `list_logs <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_logs>`_

        Returns
        -------
        List[Dict]:
            Dict has json representation of oci.logging.models.log_summary.LogSummary
        """
        log_group_id = kwargs.pop("log_group_id")

        return self.logging_client.list_logs(
            log_group_id=log_group_id, **kwargs
        ).data.__repr__()

    def list_compartments(self, **kwargs) -> List[Dict]:
        """Lists the compartments in a compartment specified by TENANCY_OCID env variable. This is a pass through the OCI list_compartments
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_compartments <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.identity.IdentityClient.html#oci.identity.IdentityClient.list_compartments>`_

        Returns
        -------
        List[Dict]:
            Dict has json representation of oci.identity.models.Compartment
        """
        if not TENANCY_OCID:
            raise AquaClientError(
                f"TENANCY_OCID must be available in environment"
                " variables to list the sub compartments."
            )

        return self.identity_client.list_compartments(
            compartment_id=TENANCY_OCID, **kwargs
        ).data.__repr__()

    def get_default_compartment(self):
        """Returns user compartment OCID fetched from environment variables.

        Returns
        -------
        dict:
            The compartment ocid.
        """
        if not COMPARTMENT_OCID:
            logger.error("No compartment id found from environment variables.")
        return dict(compartment_id=COMPARTMENT_OCID)

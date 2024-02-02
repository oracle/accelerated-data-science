#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List
from ads.aqua.base import AquaApp
from oci.logging.models import LogGroupSummary, LogSummary
from oci.identity.models import Compartment
from ads.config import COMPARTMENT_OCID, ODSC_MODEL_COMPARTMENT_OCID
from oci.exceptions import ServiceError
from ads.aqua.exception import AquaServiceError, AquaClientError


class AquaUIApp(AquaApp):
    """Contains APIs for supporting Aqua UI.

    Attributes
    ----------

    Methods
    -------
    list_log_groups(self, **kwargs) -> List["LogGroupSummary"]
        Lists all log groups for the specified compartment or tenancy.
    list_logs(self, **kwargs) -> List[LogSummary]
        Lists the specified log group's log objects.
    list_compartments(self, **kwargs) -> List[Compartment]
        Lists the compartments in a specified compartment.

    """

    def list_log_groups(self, **kwargs) -> List["LogGroupSummary"]:
        """Lists all log groups for the specified compartment or tenancy. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_log_groups <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_log_groups>`_

        Returns
        -------
        List[LogGroupSummary]:
            A Response object with data of type list of LogGroupSummary
        """

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        try:
            return self.logging_client.list_log_groups(
                compartment_id=compartment_id, **kwargs
            ).data.__repr__()
        # todo : update this once exception handling is set up
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)

    def list_logs(self, **kwargs) -> List[LogSummary]:
        """Lists the specified log group's log objects. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as log_group_id, log_type
            for `list_logs <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_logs>`_

        Returns
        -------
        List[LogSummary]:
            A Response object with data of type list of LogSummary
        """
        log_group_id = kwargs.pop("log_group_id")

        try:
            return self.logging_client.list_logs(
                log_group_id=log_group_id, **kwargs
            ).data.__repr__()
        # todo : update this once exception handling is set up
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)

    def list_compartments(self, **kwargs) -> List[Compartment]:
        """Lists the compartments in a compartment specified by ODSC_MODEL_COMPARTMENT_OCID env variable. This is a pass through the OCI list_compartments
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_compartments <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.identity.IdentityClient.html#oci.identity.IdentityClient.list_compartments>`_

        Returns
        -------
        List[Compartments]:
            oci.identity.models.Compartment
        """
        try:
            if not ODSC_MODEL_COMPARTMENT_OCID:
                raise AquaClientError(
                    f"ODSC_MODEL_COMPARTMENT_OCID must be available in environment"
                    " variables to list the sub compartments."
                )

            return self.identity_client.list_compartments(
                compartment_id=ODSC_MODEL_COMPARTMENT_OCID, **kwargs
            ).data.__repr__()
        # todo : update this once exception handling is set up
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)

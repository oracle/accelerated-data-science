#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from oci.exceptions import ServiceError
from oci.identity.models import Compartment
from datetime import datetime, timedelta
from threading import Lock
from cachetools import TTLCache

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaValueError
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

    _compartments_cache = TTLCache(
        maxsize=10, ttl=timedelta(hours=2), timer=datetime.now
    )
    _cache_lock = Lock()

    def list_log_groups(self, **kwargs) -> str:
        """Lists all log groups for the specified compartment or tenancy. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as compartment_id,
            for `list_log_groups <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_log_groups>`_

        Returns
        -------
            str has json representation of oci.logging.models.log_group.LogGroup
        """

        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)

        return self.logging_client.list_log_groups(
            compartment_id=compartment_id, **kwargs
        ).data.__repr__()

    def list_logs(self, **kwargs) -> str:
        """Lists the specified log group's log objects. This is a pass through the OCI list_log_groups
        API.

        Parameters
        ----------
        kwargs
            Keyword arguments, such as log_group_id, log_type
            for `list_logs <https://docs.oracle.com/en-us/iaas/tools/python/2.119.1/api/logging/client/oci.logging.LoggingManagementClient.html#oci.logging.LoggingManagementClient.list_logs>`_

        Returns
        -------
        str:
            str has json representation of oci.logging.models.log_summary.LogSummary
        """
        log_group_id = kwargs.pop("log_group_id")

        return self.logging_client.list_logs(
            log_group_id=log_group_id, **kwargs
        ).data.__repr__()

    def list_compartments(self) -> str:
        """Lists the compartments in a compartment specified by TENANCY_OCID env variable. This is a pass through the OCI list_compartments
        API.

        Returns
        -------
        str:
            str has json representation of oci.identity.models.Compartment
        """
        try:
            if not TENANCY_OCID:
                raise AquaValueError(
                    f"TENANCY_OCID must be available in environment"
                    " variables to list the sub compartments."
                )

            if TENANCY_OCID in self._compartments_cache.keys():
                logger.info(
                    f"Returning compartments list in {TENANCY_OCID} from cache."
                )
                return self._compartments_cache.get(TENANCY_OCID)

            compartments = []
            # User may not have permissions to list compartment.
            try:
                compartments.extend(
                    self.list_resource(
                        list_func_ref=self.identity_client.list_compartments,
                        compartment_id=TENANCY_OCID,
                        compartment_id_in_subtree=True,
                        access_level="ANY",
                    )
                )
            except ServiceError as se:
                logger.error(
                    f"ERROR: Unable to list all sub compartment in tenancy {TENANCY_OCID}."
                )
                try:
                    compartments.append(
                        self.list_resource(
                            list_func_ref=self.identity_client.list_compartments,
                            compartment_id=TENANCY_OCID,
                        )
                    )
                except ServiceError as se:
                    logger.error(
                        f"ERROR: Unable to list all child compartment in tenancy {TENANCY_OCID}."
                    )
            try:
                root_compartment = self.identity_client.get_compartment(
                    TENANCY_OCID
                ).data
                compartments.insert(0, root_compartment)
            except ServiceError as se:
                logger.error(
                    f"ERROR: Unable to get details of the root compartment {TENANCY_OCID}."
                )
                compartments.insert(
                    0,
                    Compartment(id=TENANCY_OCID, name=" ** Root - Name N/A **"),
                )
            res = compartments.__repr__()
            self._compartments_cache.__setitem__(key=TENANCY_OCID, value=res)

            return res

        # todo : update this once exception handling is set up
        except ServiceError as se:
            raise se

    def get_default_compartment(self) -> dict:
        """Returns user compartment OCID fetched from environment variables.

        Returns
        -------
        dict:
            The compartment ocid.
        """
        if not COMPARTMENT_OCID:
            logger.error("No compartment id found from environment variables.")
        return dict(compartment_id=COMPARTMENT_OCID)

    def clear_compartments_list_cache(self) -> dict:
        """Allows caller to clear compartments list cache
        Returns
        -------
            dict with the key used, and True if cache has the key that needs to be deleted.
        """
        res = {}
        logger.info(f"Clearing list_compartments cache")
        with self._cache_lock:
            if TENANCY_OCID in self._compartments_cache.keys():
                self._compartments_cache.pop(key=TENANCY_OCID)
                res = {
                    "key": {
                        "tenancy_ocid": TENANCY_OCID,
                    },
                    "cache_deleted": True,
                }
        return res

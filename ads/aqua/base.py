#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import oci
from ads.common.auth import default_signer
from ads.aqua.exception import AquaServiceError, AquaClientError
from oci.exceptions import ServiceError, ClientError
from ads.common import oci_client as oc
from ads.common.utils import extract_region


class AquaApp:
    """Base Aqua App to contain common components."""

    def __init__(self) -> None:
        self._auth = default_signer()
        self.ds_client = oc.OCIClientFactory(**self._auth).data_science
        self.logging_client = oc.OCIClientFactory(**self._auth).logging_management
        self.identity_client = oc.OCIClientFactory(**self._auth).identity
        self.region = extract_region(self._auth)

    def list_resource(
        self,
        list_func_ref,
        **kwargs,
    ) -> list:
        """Generic method to list OCI Data Science resources.

        Parameters
        ----------
        list_func_ref : function
            A reference to the list operation which will be called.
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI Data Science resources.
        """
        try:
            items = oci.pagination.list_call_get_all_results(
                list_func_ref,
                **kwargs,
            ).data

            return items
        except ServiceError as se:
            raise AquaServiceError(opc_request_id=se.request_id, status_code=se.code)
        except ClientError as ce:
            raise AquaClientError(str(ce))

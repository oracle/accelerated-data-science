#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import List

import oci
from oci.exceptions import ClientError, ServiceError

from ads.aqua.exception import AquaClientError, AquaServiceError, oci_exception_handler
from ads.common.auth import default_signer
from ads.common.utils import extract_region


class AquaApp:
    """Base Aqua App to contain common components."""

    def __init__(self) -> None:
        self._auth = default_signer()
        self.client = oci.data_science.DataScienceClient(**self._auth)
        self.region = extract_region(self._auth)

    @oci_exception_handler
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
        items = oci.pagination.list_call_get_all_results(
            list_func_ref,
            **kwargs,
        ).data

        return items

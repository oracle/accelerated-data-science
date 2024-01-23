#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from typing import List

import logging
import oci
from ads.common.auth import default_signer

logger = logging.getLogger(__name__)


class AquaApp:
    """Base Aqua App to contain common components."""

    def __init__(self) -> None:
        self._auth = default_signer()
        self.client = oci.data_science.DataScienceClient(**self._auth)

    def list_resource(
        self,
        list_func_ref,
        **kwargs,
    ) -> List[dict]:
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
        except Exception as e:
            # show opc-request-id and status code
            logger.error(f"Failing to retreive resources in the given compartment. {e}")
            return []

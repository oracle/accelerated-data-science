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
        self.client = oci.data_science.DataScienceClient(**default_signer())

    def list_resource(
        self,
        compartment_id: str = None,
    ) -> List[dict]:
        """Generic method to list OCI Data Science resources.

        Parameters
        ----------
        compartment_id : str
            Compartment ID of the OCI resources. Defaults to None.
            If compartment_id is not specified,
            the value of NB_SESSION_COMPARTMENT_OCID in environment variable will be used.
        limit : int
            The maximum number of items to return. Defaults to 0, All items will be returned
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI resources

        Raises
        ------
        NotImplementedError
            List method is not supported or implemented.

        """
        try:
            # https://docs.oracle.com/en-us/iaas/tools/python-sdk-examples/2.118.1/datascience/list_models.py.html
            # list_call_get_all_results
            items = oci.pagination.list_call_get_all_results(
                self._find_oci_method("list"),
                self.check_compartment_id(compartment_id),
                **kwargs,
            ).data

            return items
        except Exception as e:
            # show opc-request-id and status code
            logger.error(f"Failing to retreive models in the given compartment. {e}")
            return []

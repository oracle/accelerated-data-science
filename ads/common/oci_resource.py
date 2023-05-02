#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Contains class wrapping OCI resource search service
"""
import oci.resource_search
from ads.common.oci_mixin import OCIClientMixin


class ResourceNotFoundError(Exception):   # pragma: no cover
    """Exception when an OCI resource is not found or user does not have permission to access it.
    This could mean the resource does not exist, or
    there is not enough permission to find/access the resource.
    """


class SEARCH_TYPE(str):
    FREETEXT = "FreeText"
    STRUCTURED = "Structured"


class OCIResource(OCIClientMixin):
    """Contains helper methods for getting information from OCIResourceSearch service.

    Usage:
    Find the compartment ID of an OCI resource:
    >>> OCIResource.get_compartment_id("YOUR_OCID")
    Search for OCI resource matching free text (Similar to the search box in OCI console):
    >>> OCIResource.search("YOUR_FREE_TEXT")
    Search for OCI resource matching structured text:
    >>> OCIResource.search("STRUCTURED_TEXT", type="Structured")
    """

    @classmethod
    def init_client(cls, **kwargs) -> oci.resource_search.ResourceSearchClient:
        return cls._init_client(
            client=oci.resource_search.ResourceSearchClient, **kwargs
        )

    @classmethod
    def get_compartment_id(cls, ocid) -> str:
        """Gets the compartment OCID of an OCI resource, given the resource's OCID.

        Parameters
        ----------
        ocid : str
            OCID of a resource

        Returns
        -------
        str:
            Compartment OCID of the resource


        """
        results = set([r.compartment_id for r in cls.search(ocid)])
        if len(results) == 1:
            return list(results)[0]
        elif not results:
            raise ResourceNotFoundError(
                f"Resource not found for {ocid}. It may not exist or you may not have permission to list/access it."
            )
        else:
            raise ValueError(
                f"Unable determine compartment ID, multiple matches: {results}"
            )

    @classmethod
    def search(
        cls,
        query: str,
        type: str = "FreeText",
        config: dict = None,
        tenant_id: str = None,
        limit: int = 500,
        page: str = None,
        **kwargs,
    ) -> list:
        """Search OCI resource by free text.

        Parameters
        ----------
        query : str
            The content to search for.
        type :  str (optional)
            The type of SearchDetails, whether "FreeText" or "Structured".
            Defaults to "FreeText".
        config: dict (optional)
            Configuration keys and values as per SDK and Tool Configuration.
            The from_file() method can be used to load configuration from a file.
            Alternatively, a dict can be passed. You can validate_config the dict
            using validate_config(). Defaults to None.
        tenant_id: str (optional)
            The tenancy ID, which can be used to specify a different tenancy
            (for cross-tenancy authorization) when searching for resources in
            a different tenancy. Defaults to None.
        limit: int (optional)
            The maximum number of items to return. The value must be between
            1 and 1000. Defaults to 500.
        page: str (optional)
            The page at which to start retrieving results.

        Returns
        -------
        list
            A list of search results

        """
        if not config:
            client = cls.init_client()
        else:
            client = oci.resource_search.ResourceSearchClient(config=config)

        if type == "Structured":
            search_details = oci.resource_search.models.StructuredSearchDetails(
                type="Structured",
                query=query,
                matching_context_type=None,
            )
        elif type == "FreeText":
            search_details = oci.resource_search.models.FreeTextSearchDetails(
                text=query, type="FreeText"
            )
        else:
            raise ValueError(f"Invalid Type: {type}")

        return client.search_resources(
            search_details,
            limit=limit,
            page=page,
            tenant_id=tenant_id,
            **kwargs,
        ).data.items

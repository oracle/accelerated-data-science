#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging

import feature_store_client.feature_store as fs

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin

from feature_store_client.feature_store.models import (
    Lineage,
)

logger = logging.getLogger(__name__)


class OCILineage(OCIFeatureStoreMixin, Lineage):
    """Represents an OCI Data Science Lineage Resource Class.
    This class contains all attributes of the `oci.data_science.models.Lineaage`.
    The main purpose of this class is to link the `oci.data_science.models.Lineaage`
    and the related client methods.
    Linking the `Lineaage` (payload) to Create methods.

    The `OCILineage` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "id": "Feature Store Resource Id, It can be dataset or feature group",
            "constructType": "Feature store lineage construct type for which lineage is supported",
        }
        Lineage = OCILineage(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        properties = {
            "id": "Feature Store Resource Id",
            "constructType": "Feature store lineage construct type for which lineage is supported",
        }
        Lineage = OCILineage(**properties)

    Methods
    -------
    create(self) -> "Lineage"
        Creates Lineage.
    from_id(cls, ocid: str) -> "Lineage":
        Gets existing Lineage by Id.
    Examples
    --------
    >>> lineage = OCILineage.from_id("<feature_store_construct_id>")
    >>> lineage.create()
    """

    def __init__(self, **kwargs) -> None:
        """Initialize a Lineage object.

        Parameters
        ----------
        kwargs:
            Same as kwargs in feature_store.models.Lineage.
            Keyword arguments are passed into OCI feature group Lineage model to initialize the properties.

        """

        super().__init__(**kwargs)

    def from_id(self, feature_store_lineage_resource_id: str, **kwargs) -> Lineage:
        """Gets lineage resource  by feature store id.

        Parameters
        -----------
        feature_store_lineage_resource_id: str
            The id of the feature store lineage resource.

        Returns
        -------
        Lineage
            Feature store lineage information.
        """
        if not feature_store_lineage_resource_id:
            raise ValueError("feature store lineage resource id not provided.")
        return super().client.get_lineage(feature_store_lineage_resource_id, **kwargs)

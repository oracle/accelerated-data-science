#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import feature_store_client.feature_store as fs
from feature_store_client.feature_store.models import (
    CreateEntityDetails,
    UpdateEntityDetails,
    Entity,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin
import logging

logger = logging.getLogger(__name__)


class OCIEntity(OCIFeatureStoreMixin, Entity):
    """Represents an OCI Data Science Entity.
    This class contains all attributes of the `oci.data_science.models.Entity`.
    The main purpose of this class is to link the `oci.data_science.models.Entity`
    and the related client methods.
    Linking the `Transformation` (payload) to Create/Update/Get/List/Delete methods.

    The `OCIEntity` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "name": "<entity_name>",
            "description": "<entity_description>",
            "feature_store_id":"<feature_store_id>",
        }
        transformation_model = OCIEntity(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "name": "<entity_name>",
            "description": "<entity_description>",
            "feature_store_id":"<feature_store_id>",
        }
        transformation_model = OCIEntity(**properties)

    Methods
    -------
    create(self) -> "OCIEntity"
        Creates entity.
    delete(self):
        Deletes entity.
    from_id(cls, ocid: str) -> "OCIEntity":
        Gets existing entity by Id.
    Examples
    --------
    >>> oci_entity = OCIEntity.from_id("<entity_id>")
    >>> oci_entity.delete()
    """

    # Overriding default behavior
    @classmethod
    def init_client(cls, **kwargs) -> fs.feature_store_client.FeatureStoreClient:
        client = super().init_client(**kwargs)

        # Define the list entities callable to list the resources
        cls.OCI_LIST_METHOD = client.list_entities

        return client

    def create(self) -> "OCIEntity":
        """Creates entity resource.

        Returns
        -------
        OCIEntity
            The `OCIEntity` instance (self), which allows chaining additional method.
        """
        if not self.compartment_id:
            raise ValueError("The `compartment_id` must be specified.")
        entity_details = self.to_oci_model(CreateEntityDetails)
        return self.update_from_oci_model(
            self.client.create_entity(entity_details).data
        )

    def update(self) -> "OCIEntity":
        """Updates entity.

        Returns
        -------
        OCIEntity
            The `OCIEntity` instance (self).
        """
        return self.update_from_oci_model(
            self.client.update_entity(
                self.id, self.to_oci_model(UpdateEntityDetails)
            ).data
        )

    @classmethod
    def from_id(cls, id: str) -> "OCIEntity":
        """Gets entity resource  by id.

        Parameters
        ----------
        id: str
            The id of the entity resource.

        Returns
        -------
        OCIEntity
            An instance of `OCIEntity`.
        """
        if not id:
            raise ValueError("Entity id not provided.")
        return super().from_ocid(id)

    def delete(self):
        """Removes entity"""

        self.client.delete_entity(self.id)

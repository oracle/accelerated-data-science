#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
from functools import wraps
from typing import Callable

from feature_store_client.feature_store.models import (
    CreateFeatureStoreDetails,
    UpdateFeatureStoreDetails,
    FeatureStore,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin
import logging

logger = logging.getLogger(__name__)

FEATURE_STORE_NEEDS_TO_BE_SAVED = (
    "Feature store needs to be saved before it can be accessed."
)


class FeatureStoreNotSavedError(Exception):
    """
    An exception raised when attempting to access a feature store that has not been saved.
    """

    def __init__(self, message="Feature store has not been saved."):
        """
        Initializes a new FeatureStoreNotSavedError exception with the specified message.

        Args:
            message (str): Optional. The message to include with the exception.
        """
        self.message = message
        super().__init__(self.message)


def check_for_feature_store_id(msg: str = FEATURE_STORE_NEEDS_TO_BE_SAVED):
    """The decorator helping to check if the ID attribute sepcified for a datascience feature store.

    Parameters
    ----------
    msg: str
        The message that will be thrown.

    Raises
    ------
    FeatureStoreNotSavedError
        In case if the ID attribute not specified.

    Examples
    --------
    >>> @check_for_feature_store_id(msg="Some message.")
    ... def test_function(self, name: str, last_name: str)
    ...     pass
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.id:
                raise FeatureStoreNotSavedError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class OCIFeatureStore(OCIFeatureStoreMixin, FeatureStore):
    """Represents an OCI Data Science feature store.
    This class contains all attributes of the `oci.data_science.models.FeatureStore`.
    The main purpose of this class is to link the `oci.data_science.models.FeatureStore`
    and the related client methods.
    Linking the `FeatureStore` (payload) to Create/Update/Get/List/Delete methods.

    The `OCIFeatureStore` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_store_name>",
            "description": "<feature_store_description>",
        }
        feature_store_model = OCIFeatureStore(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "displayName": "<feature_store_name>",
            "description": "<feature_store_description>",
        }
        feature_store_model = OCIFeatureStore(**payload)

    Methods
    -------
    create(self) -> "OCIFeatureStore"
        Creates feature store.
    delete(self):
        Deletes feature store.
    update(self) -> "OCIFeatureStore":
        Updates feature store.
    from_id(cls, ocid: str) -> "OCIFeatureStore":
        Gets feature store by OCID.
    Examples
    --------
    >>> oci_feature_store = OCIFeatureStore.from_id("<feature store_id>")
    >>> oci_feature_store.description = "A brand new description"
    >>> oci_feature_store.delete()
    """

    def create(self) -> "OCIFeatureStore":
        """Creates feature store resource.
        !!! note "Lazy"
            This method is lazy and does not persist any metadata or feature data in the
            feature store on its own. To persist the feature group and save feature data
            along the metadata in the feature store, call the `materialise()` method with a
            DataFrame or with a Datasource.

        Returns
        -------
        OCIFeatureStore
            The `OCIFeatureStore` instance (self), which allows chaining additional method.
        """
        if not self.display_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.display_name = f"feature-store-{timestamp}"

        if not self.compartment_id:
            raise ValueError(
                "`compartment_id` must be specified for the feature store."
            )

        feature_store_details = self.to_oci_model(CreateFeatureStoreDetails)
        return self.update_from_oci_model(
            self.client.create_feature_store(feature_store_details).data
        )

    def update(self) -> "OCIFeatureStore":
        """Updates feature store.

        Returns
        -------
        OCIFeatureStore
            The `OCIFeatureStore` instance (self).
        """
        return self.update_from_oci_model(
            self.client.update_feature_store(
                self.id, self.to_oci_model(UpdateFeatureStoreDetails)
            ).data
        )

    def delete(self):
        """Removes feature store"""
        self.client.delete_feature_store(self.id)

    @classmethod
    def from_id(cls, id: str) -> "OCIFeatureStore":
        """Gets feature store resource  by id.

        Parameters
        ----------
        id: str
            The id of the feature store resource.

        Returns
        -------
        OCIFeatureStore
            An instance of `OCIFeatureStore`.
        """
        if not id:
            raise ValueError("Feature store id not provided.")
        return super().from_ocid(id)

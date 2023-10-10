#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime

import feature_store_client.feature_store as fs
from feature_store_client.feature_store.models import (
    Dataset,
    CreateDatasetDetails,
    UpdateDatasetDetails,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin


class OCIDataset(OCIFeatureStoreMixin, Dataset):
    """Represents an OCI Data Science dataset.
    This class contains all attributes of the `oci.data_science.models.Dataset`.
    The main purpose of this class is to link the `oci.data_science.models.Dataset`
    and the related client methods.
    Linking the `Dataset` (payload) to Create/Update/Get/List/Delete methods.

    The `OCIDataset` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<dataset_name>",
            "description": "<dataset_description>"
        }
        dataset = OCIDataset(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "displayName": "<dataset_name>",
            "description": "<dataset_description>",
        }
        dataset = OCIDataset(**payload)

    Methods
    -------
    create(self) -> "OCIDataset"
        Creates dataset
    delete(self) -> "OCIDataset":
        Deletes dataset
    update(self) -> "OCIDataset":
        Updates dataset
    from_id(cls, ocid: str) -> "OCIDataset":
        Gets dataset by OCID.
    Examples
    --------
    >>> oci_dataset = OCIDataset.from_id("<dataset_id>")
    >>> oci_dataset.description = "A brand new description"
    >>> oci_dataset.delete()
    """

    @property
    def client(self) -> fs.feature_store_client.FeatureStoreClient:
        return super().client

    def create(self) -> "OCIDataset":
        """Creates dataset on OCI Data Science platform

        !!! note "Lazy"
            This method is lazy and does not persist any metadata or feature data in the
            feature store on its own. To persist the dataset and save dataset data
            along the metadata in the feature store, call the `materialise()`.

        Returns
        -------
        OCIDataset
            The OCIDataset instance (self), which allows chaining additional method.
        """
        if not self.name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.name = f"dataset-{timestamp}"

        if not self.compartment_id:
            raise ValueError("`compartment_id` must be specified for the dataset.")

        if not self.feature_store_id:
            raise ValueError("`feature_store_id` must be specified for the dataset.")

        if not self.entity_id:
            raise ValueError("`entity_id` must be specified for the dataset.")

        dataset_details = self.to_oci_model(CreateDatasetDetails)
        return self.update_from_oci_model(
            self.client.create_dataset(dataset_details).data
        )

    def update(self) -> "OCIDataset":
        """Updates dataset.

        Returns
        -------
        OCIDataset
            The `OCIDataset` instance (self).
        """
        return self.update_from_oci_model(
            self.client.update_dataset(
                self.to_oci_model(UpdateDatasetDetails), self.id
            ).data
        )

    def delete(self):
        """Removes dataset

        Returns
        -------
        None
        """
        self.client.delete_dataset(self.id)

    @classmethod
    def from_id(cls, id: str) -> "OCIDataset":
        """Gets dataset resource  by id.

        Parameters
        ----------
        id: str
            The id of the dataset resource.

        Returns
        -------
        OCIDataset
            An instance of `OCIDataset`.
        """
        if not id:
            raise ValueError("Dataset id not provided.")
        return super().from_ocid(id)

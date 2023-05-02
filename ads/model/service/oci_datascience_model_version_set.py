#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
from typing import Optional

import oci.data_science
from ads.common.oci_datascience import OCIDataScienceMixin
from ads.common.oci_mixin import (
    LIFECYCLE_STOP_STATE,
    OCIModelNotExists,
    OCIModelWithNameMixin,
    OCIWorkRequestMixin,
)

from ads.config import COMPARTMENT_OCID

from oci.data_science.models import (
    CreateModelVersionSetDetails,
    UpdateModelVersionSetDetails,
)


class ModelVersionSetNotExists(Exception):   # pragma: no cover
    pass


class ModelVersionSetNotSaved(Exception):   # pragma: no cover
    pass


class DataScienceModelVersionSet(
    OCIDataScienceMixin,
    OCIModelWithNameMixin,
    OCIWorkRequestMixin,
    oci.data_science.models.ModelVersionSet,
):
    """Represents an OCI Data Science Model Version Set
    This class contains all attributes of the oci.data_science.models.ModelVersionSet
    The main purpose of this class is to link the `oci.data_science.models.ModelVersionSet` model
    and the related client methods. Mainly, linking the ModelVersionSet model (payload) to
    Create/Update/Get/List/Delete methods.

    A `DataScienceModelVersionSet` can be initialized by unpacking the properties stored in a dictionary (payload):

    .. code-block:: python

        properties = {
            "name": "experiment1",
            "description": "my experiment"
        }
        experiment = DataScienceModelVersionSet(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "Id": "<model_version_set_ocid>",
            "compartmentId": "<compartment_ocid>",
            "Name": "<model_version_set_name>",
        }
        experiment = DataScienceModelVersionSet(**payload)
    """

    def create(self) -> "DataScienceModelVersionSet":
        """Creates model version set on OCI Data Science platform

        Returns
        -------
        DataScienceModelVersionSet
            The DataScienceModelVersionSet instance (self), which allows chaining additional method.
        """
        if not self.name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.name = f"model-version-set-{timestamp}"

        if not self.compartment_id:
            raise ValueError(
                "`compartment_id` must be specified for the model version set."
            )

        if not self.project_id:
            raise ValueError(
                "`project_id` must be specified for the model version set."
            )

        res = self.client.create_model_version_set(
            self.to_oci_model(CreateModelVersionSetDetails)
        )
        self.update_from_oci_model(res.data)

        return self

    def update(self) -> "DataScienceModelVersionSet":
        """Updates the Data Science Model Version Set."""
        res = self.client.update_model_version_set(
            self.id, self.to_oci_model(UpdateModelVersionSetDetails)
        )
        self.update_from_oci_model(res.data)

        return self

    def delete(
        self, delete_model: Optional[bool] = False
    ) -> "DataScienceModelVersionSet":
        """Deletes the model version set.

        Parameters
        ----------
        delete_model: (bool, optional). Defaults to False.
            By default, this parameter is false. A model version set can only be
            deleted if all the models associate with it are already in the DELETED state.
            You can optionally specify the deleteRelatedModels boolean query parameters to
            true, which deletes all associated models for you.

        Returns
        -------
        DataScienceModelVersionSet
            The DataScienceModelVersionSet instance (self), which allows chaining additional method.
        """
        response = self.client.delete_model_version_set(
            self.id, is_delete_related_models=delete_model
        )
        self.get_work_request_response(
            response,
            wait_for_state=LIFECYCLE_STOP_STATE,
            success_state="SUCCEEDED",
            wait_interval_seconds=1,
        )
        self.sync()
        return self

    @classmethod
    def from_ocid(cls, ocid: str) -> "DataScienceModelVersionSet":
        """Gets a Model Version Set by OCID.

        Parameters
        ----------
        ocid: str
            The OCID of the model version set.

        Returns
        -------
        DataScienceModelVersionSet
            An instance of DataScienceModelVersionSet.
        """
        return super().from_ocid(ocid)

    @classmethod
    def from_name(
        cls, name: str, compartment_id: Optional[str] = None
    ) -> "DataScienceModelVersionSet":
        """Gets a Model Version Set by name.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to None.
            Compartment OCID of the OCI resources. If `compartment_id` is not specified,
            the value will be taken from environment variables.
        name: str
            The name of the model version set.

        Returns
        -------
        DataScienceModelVersionSet
            An instance of DataScienceModelVersionSet.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        try:
            return super().from_name(name=name, compartment_id=compartment_id)
        except OCIModelNotExists:
            raise ModelVersionSetNotExists(f"The model version set `{name}` not found.")

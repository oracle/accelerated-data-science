#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import pandas as pd
from ads.common import utils

from feature_store_client.feature_store.models import (
    CreateFeatureGroupDetails,
    UpdateFeatureGroupDetails,
    FeatureGroup,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin


class OCIFeatureGroup(OCIFeatureStoreMixin, FeatureGroup):
    """Represents an OCI Data Science feature group.
    This class contains all attributes of the `oci.data_science.models.FeatureDefinition`.
    The main purpose of this class is to link the `oci.data_science.models.FeatureDefinition`
    and the related client methods.
    Linking the `FeatureGroup` (payload) to Create/Update/Get/List/Delete methods.

    The `OCIFeatureGroup` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_group_name>",
            "description": "<feature_group_description>",
            "feature_store_id": <feature_store_id>
            "entity_id": "<entity id>",
            "input_feature_details": "",
            "primary_keys": []
        }
        feature_group = OCIFeatureGroup(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        payload = {
            "compartmentId": "<compartment_ocid>",
            "displayName": "<feature_group_name>",
            "description": "<feature_group_description>",
            "featureStoreId": <feature_store_id>
            "entityId": "<entity id>",
            "inputFeatureDetails": "",
            "primaryKeys": []
        }
        feature_group = OCIFeatureGroup(**payload)

    Methods
    -------
    create(self) -> "OCIFeatureGroup"
        Creates feature group
    delete(self) -> "OCIFeatureGroup":
        Deletes feature group
    update(self) -> "OCIFeatureGroup":
        Updates feature group
    from_id(cls, ocid: str) -> "OCIFeatureGroup":
        Gets feature group by OCID.
    Examples
    --------
    >>> self = OCIFeatureGroup.from_id("<feature_group_id>")
    >>> self.description = "A brand new description"
    >>> self.delete()
    """

    def create(self) -> "OCIFeatureGroup":
        """Creates feature group on OCI Data Science platform

        Returns
        -------
        OCIFeatureGroup
            The OCIFeatureGroup instance (self), which allows chaining additional method.
        """
        if not self.compartment_id:
            raise ValueError(
                "`compartment_id` must be specified for the feature group."
            )

        if not self.feature_store_id:
            raise ValueError(
                "`feature_store_id` must be specified for the feature group."
            )

        if not self.entity_id:
            raise ValueError("`entity_id` must be specified for the feature group.")

        if not self.name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.name = f"feature-group-{timestamp}"

        feature_group_details = self.to_oci_model(CreateFeatureGroupDetails)
        return self.update_from_oci_model(
            self.client.create_feature_group(feature_group_details).data
        )

    def update(self) -> "OCIFeatureGroup":
        """Updates feature group.

        Returns
        -------
        OCIFeatureGroup
            The `OCIFeatureGroup` instance (self).
        """
        return self.update_from_oci_model(
            self.client.update_feature_group(
                self.to_oci_model(UpdateFeatureGroupDetails), self.id
            ).data
        )

    def delete(self):
        """Removes feature group

        Returns
        -------
        None
        """
        self.client.delete_feature_group(self.id)

    def to_df(self):
        return pd.DataFrame.from_records([self.to_df_record()])

    def to_df_record(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "time_created": self.time_created.strftime(utils.date_format),
            "time_updated": self.time_updated.strftime(utils.date_format),
            "lifecycle_state": self.lifecycle_state,
            "created_by": f"...{self.created_by[-6:]}",
            "compartment_id": f"...{self.compartment_id[-6:]}",
            "primary_keys": self.primary_keys,
            "feature_store_id": self.feature_store_id,
            "entity_id": self.entity_id,
            "input_feature_details": self.input_feature_details,
            "expectation_details": self.expectation_details,
            "statistics_config": self.statistics_config,
        }

    @classmethod
    def from_id(cls, id: str) -> "OCIFeatureGroup":
        """Gets feature group resource  by id.

        Parameters
        ----------
        id: str
            The id of the feature group resource.

        Returns
        -------
        OCIFeatureStore
            An instance of `OCIFeatureStore`.
        """
        if not id:
            raise ValueError("feature group id not provided.")
        return super().from_ocid(id)

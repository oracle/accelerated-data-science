#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from feature_store_client.feature_store.models import (
    CreateTransformationDetails,
    Transformation,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin
import logging

logger = logging.getLogger(__name__)


class OCITransformation(OCIFeatureStoreMixin, Transformation):
    """Represents an OCI Data Science Transformation.
    This class contains all attributes of the `oci.data_science.models.Transformation`.
    The main purpose of this class is to link the `oci.data_science.models.Transformation`
    and the related client methods.
    Linking the `Transformation` (payload) to Create/Update/Get/List/Delete methods.

    The `OCITransformation` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_store_name>",
            "description": "<feature_store_description>",
            "transformation_mode":"<transformation_mode>",
            "source_code":"<transformation_source_code>"
        }
        transformation_model = OCITransformation(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_store_name>",
            "description": "<feature_store_description>",
            "transformation_mode":"<transformation_mode>",
            "source_code":"<transformation_source_code>"
        }
        transformation_model = OCITransformation(**properties)

    Methods
    -------
    create(self) -> "OCITransformation"
        Creates transformation.
    delete(self) -> "OCITransformation":
        Deletes transformation.
    from_id(cls, ocid: str) -> "OCITransformation":
        Gets existing transformation by Id.
    Examples
    --------
    >>> oci_transformation = OCITransformation.from_id("<transformation_id>")
    >>> oci_transformation.delete()
    """

    def create(self) -> "OCITransformation":
        """Creates transformation resource.

        Returns
        -------
        OCITransformation
            The `OCITransformation` instance (self), which allows chaining additional method.
        """
        if not self.compartment_id:
            raise ValueError("The `compartment_id` must be specified.")

        if not self.feature_store_id:
            raise ValueError("The `feature_store_id` must be specified.")

        transformation_details = self.to_oci_model(CreateTransformationDetails)
        return self.update_from_oci_model(
            self.client.create_transformation(transformation_details).data
        )

    def delete(self):
        """Removes transformation"""
        self.client.delete_transformation(self.id)

    @classmethod
    def from_id(cls, id: str) -> "OCITransformation":
        """Gets transformation resource  by id.

        Parameters
        ----------
        id: str
            The id of the transformation resource.

        Returns
        -------
        OCIFeatureStore
            An instance of `OCITransformation`.
        """
        if not id:
            raise ValueError("Transformation id not provided.")
        return super().from_ocid(id)

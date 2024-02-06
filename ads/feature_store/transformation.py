#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import inspect
import logging
import re
import types
from copy import deepcopy
from typing import Dict, List

import pandas

from ads.common import utils
from ads.feature_store.common.enums import TransformationMode
from ads.feature_store.common.utils.base64_encoder_decoder import Base64EncoderDecoder
from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.service.oci_transformation import OCITransformation
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class Transformation(Builder):
    """Represents a Transformation Resource.

    Methods
    -------
    create(self, **kwargs) -> "Transformation"
        Creates transformation resource.
    delete(self) -> "Transformation":
        Removes transformation resource.
    to_dict(self) -> dict
        Serializes transformation to a dictionary.
    from_id(cls, id: str) -> "Transformation"
        Gets an existing transformation resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["Transformation"]
        Lists transformation resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists transformation resources as a pandas dataframe.
    with_description(self, description: str) -> "Transformation"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "Transformation"
        Sets the compartment ID.
    with_feature_store_id(self, feature_store_id: str) -> "Transformation"
        Sets the feature store ID.
    with_name(self, name: str) -> "Transformation"
        Sets the name.
    with_transformation_mode(self, transformation_mode: TransformationMode) -> "Transformation"
        Sets the transformation mode.
    with_source_code_function(self, source_code_func) -> "Transformation"
        Sets the transformation source code function.

    Examples
    --------
    >>> from ads.feature_store import transformation
    >>> import oci
    >>> import os
    >>> def transactions_df(transactions_batch):
    >>>        sql_query = f"select id, cc_num, amount from {transactions_batch}"
    >>>        return sql_query
    >>>
    >>> transformation = transformation.Transformation()
    >>>     .with_description("Feature store description")
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_name("FeatureStore")
    >>>     .with_feature_store_id("feature_store_id")
    >>>     .with_transformation_mode(TransformationMode.SQL)
    >>>     .with_source_code_function(transactions_df)
    >>> transformation.create()
    """

    _PREFIX = "transformation_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_FEATURE_STORE_ID = "featureStoreId"
    CONST_TRANSFORMATION_MODE = "transformationMode"
    CONST_SOURCE_CODE = "sourceCode"
    CONST_FUNCTION_REF = "functionRef"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_FEATURE_STORE_ID: "feature_store_id",
        CONST_TRANSFORMATION_MODE: "transformation_mode",
        CONST_SOURCE_CODE: "source_code",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes Transformation Resource.

        Parameters
        ----------
        spec: (Dict, optional). Defaults to None.
            Object specification.

        kwargs: Dict
            Specification as keyword arguments.
            If 'spec' contains the same key as the one in kwargs,
            the value from kwargs will be used.
        """
        super().__init__(spec=spec, **deepcopy(kwargs))
        # Specify oci Transformation instance
        self.oci_fs_transformation = self._to_oci_fs_transformation(**kwargs)
        self._transformation_function_name = None

    def _to_oci_fs_transformation(self, **kwargs):
        """Creates an `OCITransformation` instance from the  `Transformation`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Transformation` accepts.

        Returns
        -------
        OCITransformation
            The instance of the OCITransformation.
        """
        fs_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCITransformation(**fs_spec)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "transformation"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "Transformation":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def feature_store_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STORE_ID)

    @feature_store_id.setter
    def feature_store_id(self, value: str):
        self.with_feature_store_id(value)

    def with_feature_store_id(self, feature_store_id: str) -> "Transformation":
        """Sets the feature_store_id.

        Parameters
        ----------
        feature_store_id: str
            The featurestore id.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_STORE_ID, feature_store_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, name: str):
        self.with_name(name)

    def with_name(self, name: str) -> "Transformation":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name of Transformation resource.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def transformation_mode(self) -> str:
        return self.get_spec(self.CONST_TRANSFORMATION_MODE)

    @transformation_mode.setter
    def transformation_mode(self, transformation_mode: TransformationMode) -> None:
        self.with_transformation_mode(transformation_mode)

    def with_transformation_mode(
        self, transformation_mode: TransformationMode
    ) -> "Transformation":
        """Sets the mode of the transformation.

        Parameters
        ----------
        transformation_mode: TransformationMode
            The mode of the transformation.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """
        return self.set_spec(self.CONST_TRANSFORMATION_MODE, transformation_mode.value)

    @property
    def source_code_function(self) -> str:
        return self.get_spec(self.CONST_SOURCE_CODE)

    @source_code_function.setter
    def source_code_function(self, source_code_func):
        self.with_source_code_function(source_code_func)

    def with_source_code_function(self, source_code_func) -> "Transformation":
        """Sets the source code function for the transformation.

        Parameters
        ----------
        source_code_func: function
            source code for the transformation.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """

        if isinstance(source_code_func, types.FunctionType):
            source_code = inspect.getsource(source_code_func)
        else:
            source_code = source_code_func

        pattern = r"def\s+(\w+)\("
        match = re.search(pattern, source_code)

        # Set the function reference to the transformation
        self._transformation_function_name = match.group(1)
        return self.set_spec(
            self.CONST_SOURCE_CODE,
            Base64EncoderDecoder.encode(source_code),
        )

    def with_id(self, id: str) -> "Transformation":
        return self.set_spec(self.CONST_ID, id)

    @property
    def id(self) -> str:
        return self.get_spec(self.CONST_ID)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "Transformation":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the transformation resource.

        Returns
        -------
        FeatureStore
            The Transformation instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @classmethod
    def from_id(cls, id: str) -> "Transformation":
        """Gets an existing Transformation resource by Id.

        Parameters
        ----------
        id: str
            The Transformation id.

        Returns
        -------
        FeatureStore
            An instance of Transformation resource.
        """
        return cls()._update_from_oci_fs_transformation_model(
            OCITransformation.from_id(id)
        )

    def create(self, **kwargs) -> "Transformation":
        """Creates transformation  resource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Transformation` accepts.

        Returns
        -------
        FeatureStore
            The Transformation instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        self.compartment_id = OCIModelMixin.check_compartment_id(self.compartment_id)

        if not self.feature_store_id:
            raise ValueError("FeatureStore id must be provided.")

        if not self.source_code_function:
            raise ValueError("Transformation source code function must be provided.")

        if not self.transformation_mode:
            raise ValueError("Transformation Mode must be provided.")

        if not self.name:
            self.name = self._transformation_function_name

        if self.name != self._transformation_function_name:
            raise ValueError("Transformation name and function name must be same.")

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a transformation resource with payload {payload}")

        # Create transformation
        logger.info("Saving transformation.")
        self.oci_fs_transformation = self._to_oci_fs_transformation(**kwargs).create()
        self.with_id(self.oci_fs_transformation.id)
        return self

    def delete(self):
        """Removes transformation resource.

        Returns
        -------
        None
        """
        self.oci_fs_transformation.delete()

    def _update_from_oci_fs_transformation_model(
        self, oci_fs_transformation: OCITransformation
    ) -> "Transformation":
        """Update the properties from an OCITransformation object.

        Parameters
        ----------
        oci_fs_transformation: OCITransformation
            An instance of OCITransformation.

        Returns
        -------
        Transformation
            The Transformation instance (self).
        """

        # Update the main properties
        self.oci_fs_transformation = oci_fs_transformation
        transformation_details = oci_fs_transformation.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in transformation_details:
                if infra_attr == self.CONST_SOURCE_CODE:
                    encoded_code = Base64EncoderDecoder.encode(
                        transformation_details[infra_attr]
                    )
                    self.set_spec(infra_attr, encoded_code)
                else:
                    self.set_spec(infra_attr, transformation_details[infra_attr])

        return self

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists transformation resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the transformation resources in a pandas dataframe format.
        """
        records = []
        for oci_fs_transformation in OCITransformation.list_resource(
            compartment_id, **kwargs
        ):
            records.append(
                {
                    "id": oci_fs_transformation.id,
                    "display_name": oci_fs_transformation.display_name,
                    "description": oci_fs_transformation.description,
                    "time_created": oci_fs_transformation.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_fs_transformation.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_fs_transformation.lifecycle_state,
                    "created_by": f"...{oci_fs_transformation.created_by[-6:]}",
                    "compartment_id": f"...{oci_fs_transformation.compartment_id[-6:]}",
                    "feature_store_id": oci_fs_transformation.feature_store_id,
                    "transformation_mode": oci_fs_transformation.transformation_mode,
                    "source_code": oci_fs_transformation.source_code,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["Transformation"]:
        """Lists transformation resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Transformation.

        Returns
        -------
        List[Transformation]
            The list of the Transformation Resources.
        """
        return [
            cls()._update_from_oci_fs_transformation_model(oci_fs_transformation)
            for oci_fs_transformation in OCITransformation.list_resource(
                compartment_id, **kwargs
            )
        ]

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes transformation  to a dictionary.

        Returns
        -------
        dict
            The Transformation resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

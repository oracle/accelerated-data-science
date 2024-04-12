#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from copy import deepcopy
from typing import Dict, List, Union

import pandas
import pandas as pd
from great_expectations.core import ExpectationSuite

from ads.common.oci_mixin import OCIModelMixin
from pyspark.sql import DataFrame
from ads.common import utils
from ads.feature_store.common.enums import ExpectationType
from ads.feature_store.common.utils.utility import get_input_features_from_df
from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.input_feature_detail import FeatureDetail
from ads.feature_store.statistics_config import StatisticsConfig
from ads.feature_store.service.oci_entity import OCIEntity
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class Entity(Builder):
    """Represents an Entity Resource.

    Methods
    --------

    create(self, **kwargs) -> "Entity"
        Creates entity resource.
    delete(self) -> "Entity":
        Removes entity resource.
    to_dict(self) -> dict
        Serializes entity to a dictionary.
    from_id(cls, id: str) -> "Entity"
        Gets an existing entity resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["Entity"]
        Lists entity resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists entity resources as a pandas dataframe.
    with_description(self, description: str) -> "Entity"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "Entity"
        Sets the compartment ID.
    with_feature_store_id(self, feature_store_id: str) -> "Entity"
        Sets the feature store ID.
    with_display_name(self, name: str) -> "Entity"
        Sets the name.

    Examples
    --------
    >>> from ads.feature_store import entity
    >>> import oci
    >>> import os
    >>> entity = entity.Entity()
    >>>     .with_description("Feature store description")
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_name("FeatureStore")
    >>>     .with_feature_store_id("feature_store_id")
    >>> entity.create()
    """

    _PREFIX = "entity_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_FEATURE_STORE_ID = "featureStoreId"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_FEATURE_STORE_ID: "feature_store_id",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes Entity Resource.

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
        # Specify oci Entity instance
        self.oci_feature_group = None
        self.oci_fs_dataset = None
        self.oci_fs_entity = self._to_oci_fs_entity(**kwargs)

    def _to_oci_fs_entity(self, **kwargs):
        """Creates an `OCIEntity` instance from the  `Entity`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Entity` accepts.

        Returns
        -------
        OCIEntity
            The instance of the OCIEntity.
        """
        fs_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCIEntity(**fs_spec)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "entity"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "Entity":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def feature_store_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STORE_ID)

    @feature_store_id.setter
    def feature_store_id(self, value: str):
        self.with_feature_store_id(value)

    def with_feature_store_id(self, feature_store_id: str) -> "Entity":
        """Sets the feature_store_id.

        Parameters
        ----------
        feature_store_id: str
            The featurestore id.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_STORE_ID, feature_store_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, name: str) -> "Entity":
        return self.with_name(name)

    def with_name(self, name: str) -> "Entity":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name of entity resource.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def id(self) -> str:
        return self.get_spec(self.CONST_ID)

    def with_id(self, id: str) -> "Entity":
        return self.set_spec(self.CONST_ID, id)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "Entity":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the entity resource.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @classmethod
    def from_id(cls, id: str) -> "Entity":
        """Gets an existing entity resource by Id.

        Parameters
        ----------
        id: str
            The entity id.

        Returns
        -------
        Entity
            An instance of Entity resource.
        """
        return cls()._update_from_oci_fs_entity_model(OCIEntity.from_id(id))

    def create(self, **kwargs) -> "Entity":
        """Creates entity  resource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Entity` accepts.

        Returns
        -------
        Entity
            The Entity instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        self.compartment_id = OCIModelMixin.check_compartment_id(self.compartment_id)

        if not self.name:
            self.name = self._random_display_name()

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a entity resource with payload {payload}")

        # Create entity
        logger.info("Saving entity.")
        self.oci_fs_entity = self._to_oci_fs_entity(**kwargs).create()
        self.with_id(self.oci_fs_entity.id)
        return self

    def _build_feature_group(
        self,
        primary_keys,
        partition_keys,
        input_feature_details,
        expectation_suite: ExpectationSuite = None,
        expectation_type: ExpectationType = ExpectationType.NO_EXPECTATION,
        statistics_config: Union[StatisticsConfig, bool] = True,
        transformation_id: str = None,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
        transformation_kwargs: Dict = None,
    ):
        feature_group_resource = (
            FeatureGroup()
            .with_feature_store_id(self.feature_store_id)
            .with_name(name)
            .with_description(description)
            .with_compartment_id(
                compartment_id if compartment_id else self.compartment_id
            )
            .with_entity_id(self.id)
            .with_transformation_id(transformation_id)
            .with_partition_keys(partition_keys)
            .with_transformation_kwargs(transformation_kwargs)
            .with_primary_keys(primary_keys)
            .with_input_feature_details(input_feature_details)
            .with_statistics_config(statistics_config)
        )

        if expectation_suite and expectation_type:
            feature_group_resource.with_expectation_suite(
                expectation_suite, expectation_type
            )
        return feature_group_resource

    def create_feature_group(
        self,
        primary_keys: List[str],
        partition_keys: List[str] = None,
        input_feature_details: List[FeatureDetail] = None,
        schema_details_dataframe: Union[DataFrame, pd.DataFrame] = None,
        expectation_suite: ExpectationSuite = None,
        expectation_type: ExpectationType = ExpectationType.NO_EXPECTATION,
        statistics_config: Union[StatisticsConfig, bool] = True,
        transformation_id: str = None,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
        transformation_kwargs: Dict = None,
    ) -> "FeatureGroup":
        """Creates FeatureGroup  resource.

        Parameters
        ----------
        primary_keys: List[str]
            List of primary keys.
        partition_keys: List[str]
            List of partition_keys to partition the materialized data.
        input_feature_details: List[FeatureDetail]
            Raw feature schema for the input features.
        schema_details_dataframe: Union[DataFrame, pd.DataFrame]
            Dataframe from which raw schema will be inferred.
        expectation_suite: ExpectationSuite = None
            Expectation details for the validation.
        statistics_config: StatisticsConfig = None
            Config details for the Statistics.
        expectation_type: ExpectationType
            Type of the expectation.
        transformation_id: str = None
            Transformation Mode.
        name: str = None
            Name for the resource.
        description: str = None
            Description about the Resource.
        compartment_id: str = None
            compartment_id
        transformation_kwargs: Dict
            Arguments for the transformation.


        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        if not self.id:
            raise ValueError(
                "Entity Resource must be created or saved before creating the FeatureDefinition."
            )

        if input_feature_details is None and schema_details_dataframe is None:
            raise ValueError(
                "At least one of input_feature_details or schema_details_dataframe must be provided to create the Feature Group."
            )

        raw_feature_details = (
            input_feature_details
            if input_feature_details
            else get_input_features_from_df(
                schema_details_dataframe, self.feature_store_id
            )
        )

        self.oci_feature_group = self._build_feature_group(
            primary_keys,
            partition_keys,
            raw_feature_details,
            expectation_suite,
            expectation_type,
            statistics_config,
            transformation_id,
            name,
            description,
            compartment_id,
            transformation_kwargs,
        )

        return self.oci_feature_group.create()

    def delete_feature_group(self):
        """Removes FeatureGroup Resource.

        Returns
        -------
        None
        """
        if not self.oci_feature_group:
            raise ValueError(
                "FeatureGroup must be created or exist before deleting it."
            )

        self.oci_feature_group.delete()

    @classmethod
    def list_feature_group(
        cls, compartment_id: str = None, **kwargs
    ) -> List["FeatureGroup"]:
        """Lists FeatureGroup resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        List["FeatureGroup"]
            The list of the FeatureGroup Resources.
        """

        return FeatureGroup.list(compartment_id, **kwargs)

    @classmethod
    def list_feature_group_df(
        cls, compartment_id: str = None, **kwargs
    ) -> "pandas.DataFrame":
        """Lists FeatureGroup resources in a given compartment as pandas dataframe.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        "pandas.DataFrame"
            The list of the FeatureGroup Resources.
        """

        return FeatureGroup.list_df(compartment_id, **kwargs)

    def _build_dataset(
        self,
        query: str,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
        expectation_suite: ExpectationSuite = None,
        expectation_type: ExpectationType = ExpectationType.NO_EXPECTATION,
        statistics_config: Union[StatisticsConfig, bool] = True,
        partition_keys: List[str] = None,
    ):
        dataset_resource = (
            Dataset()
            .with_name(name)
            .with_description(description)
            .with_feature_store_id(self.feature_store_id)
            .with_entity_id(self.id)
            .with_query(query)
            .with_compartment_id(
                compartment_id if compartment_id else self.compartment_id
            )
            .with_statistics_config(statistics_config)
            .with_partition_keys(partition_keys)
        )

        if expectation_suite:
            dataset_resource.with_expectation_suite(expectation_suite, expectation_type)

        if statistics_config is not None:
            dataset_resource.with_statistics_config(statistics_config)

        return dataset_resource

    def create_dataset(
        self,
        query: str,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
        expectation_suite: ExpectationSuite = None,
        expectation_type: ExpectationType = ExpectationType.NO_EXPECTATION,
        statistics_config: Union[StatisticsConfig, bool] = True,
        partition_keys: List[str] = None,
    ) -> "Dataset":
        """Creates Dataset resource.

        Parameters
        ----------
        query: str
            SQL Query that will be used to create the dataset by joining FeatureGroups.
        name: str = None
             Name for the resource.
        description: str = None
            Description about the Resource.
        compartment_id: str = None
            compartment_id
        expectation_suite: ExpectationSuite = None
            Expectation details for the validation.
        expectation_type: ExpectationType
            Type of the expectation.
        statistics_config: StatisticsConfig = None
            Config details for the Statistics.
        partition_keys: List[str]
            Partition keys for the datset.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        if not self.id:
            raise ValueError(
                "Entity Resource must be created or saved before creating the Dataset."
            )

        self.oci_fs_dataset = self._build_dataset(
            query,
            name,
            description,
            compartment_id,
            expectation_suite,
            expectation_type,
            statistics_config,
            partition_keys,
        )

        return self.oci_fs_dataset.create()

    def delete_dataset(self):
        """Removes Dataset Resource.

        Returns
        -------
        None
        """
        if not self.oci_fs_dataset:
            raise ValueError("Dataset must be created or exist before deleting it.")

        self.oci_fs_dataset.delete()

    @classmethod
    def list_dataset(cls, compartment_id: str = None, **kwargs) -> List["Dataset"]:
        """Lists Dataset resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Dataset.

        Returns
        -------
        List["Dataset"]
            The list of the Dataset Resources.
        """

        return Dataset.list(compartment_id, **kwargs)

    @classmethod
    def list_dataset_df(
        cls, compartment_id: str = None, **kwargs
    ) -> "pandas.DataFrame":
        """Lists Dataset resources in a given compartment as pandas dataframe.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Dataset.

        Returns
        -------
        "pandas.DataFrame"
            The list of the Dataset Resources.
        """

        return Dataset.list_df(compartment_id, **kwargs)

    def delete(self):
        """Removes entity resource.

        Returns
        -------
        None
        """
        self.oci_fs_entity.delete()

    def _update_from_oci_fs_entity_model(self, oci_fs_entity: OCIEntity) -> "Entity":
        """Update the properties from an OCIEntity object.

        Parameters
        ----------
        oci_fs_entity: OCIEntity
            An instance of OCIEntity.

        Returns
        -------
        Entity
            The Entity instance (self).
        """

        # Update the main properties
        self.oci_fs_entity = oci_fs_entity
        entity_details = oci_fs_entity.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in entity_details:
                self.set_spec(infra_attr, entity_details[infra_attr])

        return self

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists entity resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the entity resources in a pandas dataframe format.
        """
        records = []
        for oci_fs_entity in OCIEntity.list_resource(compartment_id, **kwargs):
            records.append(
                {
                    "id": oci_fs_entity.id,
                    "display_name": oci_fs_entity.display_name,
                    "description": oci_fs_entity.description,
                    "time_created": oci_fs_entity.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_fs_entity.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_fs_entity.lifecycle_state,
                    "created_by": f"...{oci_fs_entity.created_by[-6:]}",
                    "compartment_id": f"...{oci_fs_entity.compartment_id[-6:]}",
                    "feature_store_id": oci_fs_entity.feature_store_id,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["Entity"]:
        """Lists Entity Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        List[Entity]
            The list of the Entity Resources.
        """
        return [
            cls()._update_from_oci_fs_entity_model(oci_fs_entity)
            for oci_fs_entity in OCIEntity.list_resource(compartment_id, **kwargs)
        ]

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes entity  to a dictionary.

        Returns
        -------
        dict
            The entity resource serialized as a dictionary.
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

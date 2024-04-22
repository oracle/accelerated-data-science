#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from copy import deepcopy
from typing import Dict, Any, List, Optional

import pandas
from ads.common.oci_mixin import OCIModelMixin

from ads.common import utils
from ads.feature_store.common.enums import DataFrameType
from ads.feature_store.common.spark_session_singleton import SparkSessionSingleton

from ads.feature_store.entity import Entity
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine
from ads.feature_store.service.oci_feature_store import OCIFeatureStore
from ads.feature_store.transformation import Transformation, TransformationMode
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class FeatureStore(Builder):
    """Represents a Feature Store Resource.

    Methods
    -------
    create(self, **kwargs) -> "FeatureStore"
        Creates feature store resource.
    delete(self) -> "FeatureStore":
        Removes feature store resource.
    to_dict(self) -> dict
        Serializes feature store to a dictionary.
    from_id(cls, id: str) -> "FeatureStore"
        Gets an existing feature store resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["FeatureStore"]
        Lists feature store resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists feature store resources as a pandas dataframe.
    with_description(self, description: str) -> "FeatureStore"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "FeatureStore"
        Sets the compartment ID.
    with_display_name(self, name: str) -> "FeatureStore"
        Sets the name.
    with_offline_config(self, metastore_id: str, **kwargs: Dict[str, Any]) -> "FeatureStore"
        Sets the offline config.

    Examples
    --------
    >>> from ads.feature_store import feature_store
    >>> import oci
    >>> import os
    >>> feature_store = feature_store.FeatureStore()
    >>>     .with_description("Feature store description")
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_name("FeatureStore")
    >>>     .with_offline_config(
    >>>         metastore_id="metastore id")
    >>> feature_store.create()
    """

    _PREFIX = "featurestore_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_OFFLINE_CONFIG = "offlineConfig"
    CONST_METASTORE_ID = "metastoreId"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_OFFLINE_CONFIG: "offline_config",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes FeatureStore Resource.

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
        # Specify oci FeatureStore instance
        self.spark_engine = None
        self.oci_transformation = None
        self.oci_fs_entity = None
        self.oci_fs = self._to_oci_fs(**kwargs)

    def _to_oci_fs(self, **kwargs):
        """Creates an `OCIFeatureStore` instance from the  `FeatureStore`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureStore` accepts.

        Returns
        -------
        OCIFeatureStore
            The instance of the OCIFeatureStore.
        """
        fs_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCIFeatureStore(**fs_spec)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "featurestore"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "FeatureStore":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, name: str) -> "FeatureStore":
        return self.with_name(name)

    def with_name(self, name: str) -> "FeatureStore":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name of feature store.

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def id(self) -> str:
        return self.get_spec(self.CONST_ID)

    def with_id(self, id: str) -> "FeatureStore":
        return self.set_spec(self.CONST_ID, id)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "FeatureStore":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the feature store.

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    def init(self):
        """Initialize the feature store with spark session."""

        # Initialize the Spark Session
        return SparkSessionSingleton(
            self.offline_config.get(self.CONST_METASTORE_ID)
            if self.offline_config
            else None
        )

    @property
    def offline_config(self) -> dict:
        return self.get_spec(self.CONST_OFFLINE_CONFIG)

    @offline_config.setter
    def offline_config(self, metastore_id: str, **kwargs: Dict[str, Any]):
        self.with_offline_config(metastore_id, **kwargs)

    def with_offline_config(
        self, metastore_id: str, **kwargs: Dict[str, Any]
    ) -> "FeatureStore":
        """Sets the offline config.

        Parameters
        ----------
        metastore_id: str
            The metastore id for offline store
        kwargs: Dict[str, Any]
            Additional key value arguments

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self)
        """
        return self.set_spec(
            self.CONST_OFFLINE_CONFIG,
            {
                self.CONST_METASTORE_ID: metastore_id,
                **kwargs,
            },
        )

    @classmethod
    def from_id(cls, id: str) -> "FeatureStore":
        """Gets an existing feature store resource by Id.

        Parameters
        ----------
        id: str
            The feature store id.

        Returns
        -------
        FeatureStore
            An instance of FeatureStore resource.
        """
        return cls()._update_from_oci_fs_model(OCIFeatureStore.from_id(id))

    def create(self, **kwargs) -> "FeatureStore":
        """Creates feature store  resource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureStore` accepts.

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        self.compartment_id = OCIModelMixin.check_compartment_id(self.compartment_id)

        if (
            not self.offline_config
            or not self.CONST_METASTORE_ID in self.offline_config
        ):
            raise ValueError("OfflineConfig must be provided with valid metastore_id.")

        if not self.name:
            self.name = self._random_display_name()

        payload = deepcopy(self._spec)
        # TODO: Remove when no longer needed
        payload.pop("id", None)
        logger.debug(f"Creating a feature store resource with payload {payload}")

        # Create feature store
        logger.info("Saving feature store.")
        self.oci_fs = self._to_oci_fs(**kwargs).create()
        self.with_id(self.oci_fs.id)
        return self

    def _build_entity(
        self,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
    ):
        """Creates entity class.

        Parameters
        ----------
        name: str
            Name for the entity.
        description: str
            Description for the entity.
        compartment_id: str
            compartment_id for the entity.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        entity = (
            Entity()
            .with_name(name)
            .with_description(description)
            .with_compartment_id(
                compartment_id if compartment_id else self.compartment_id
            )
            .with_feature_store_id(self.id)
        )

        return entity

    def create_entity(
        self,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
    ):
        """Creates entity resource from feature store.

        Parameters
        ----------
        name: str
             name for the entity.
        description: str
             description for the entity.
        compartment_id: str
             compartment_id for the entity.

        Returns
        -------
        Entity
            The Entity instance (self)
        """
        if not self.id:
            raise ValueError(
                "FeatureStore Resource must be created or saved before creating the entity."
            )

        self.oci_fs_entity = self._build_entity(name, description, compartment_id)
        return self.oci_fs_entity.create()

    def delete_entity(self):
        """Removes FeatureStore Resource.

        Returns
        -------
        None
        """
        if not self.oci_fs_entity:
            raise ValueError("Entity must be created or exist before deleting it.")

        self.oci_fs_entity.delete()

    @classmethod
    def list_entities(cls, compartment_id: str = None, **kwargs) -> List["Entity"]:
        """Lists Entity resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        List["Entity"]
            The list of the Entity Resources.
        """

        return Entity.list(compartment_id, **kwargs)

    @classmethod
    def list_entities_df(
        cls, compartment_id: str = None, **kwargs
    ) -> "pandas.DataFrame":
        """Lists Entity resources in a given compartment as pandas dataframe.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        "pandas.DataFrame"
            The list of the Entity Resources.
        """

        return Entity.list_df(compartment_id, **kwargs)

    def _build_transformation(
        self,
        source_code_func,
        transformation_mode,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
    ):
        transformation = (
            Transformation()
            .with_transformation_mode(transformation_mode)
            .with_description(description)
            .with_name(name)
            .with_source_code_function(source_code_func)
            .with_compartment_id(
                compartment_id if compartment_id else self.compartment_id
            )
            .with_feature_store_id(self.id)
        )

        return transformation

    def create_transformation(
        self,
        source_code_func,
        transformation_mode: TransformationMode,
        name: str = None,
        description: str = None,
        compartment_id: str = None,
    ) -> "Transformation":
        """Creates transformation resource from feature store.

        Parameters
        ----------
        source_code_func
            Transformation source code.
        transformation_mode: TransformationMode
            Transformation mode.
        name: str
             name for the entity.
        description: str
             description for the entity.
        compartment_id: str
             compartment_id for the entity.

        Returns
        -------
        Transformation
            The Transformation instance (self)
        """
        if not self.id:
            raise ValueError(
                "FeatureStore Resource must be created or saved before creating the transformation."
            )

        self.oci_transformation = self._build_transformation(
            source_code_func,
            transformation_mode,
            name,
            description,
            compartment_id,
        )

        return self.oci_transformation.create()

    def delete_transformation(self):
        """Removes Transformation Resource.

        Returns
        -------
        None
        """
        if not self.oci_transformation:
            raise ValueError(
                "Transformation must be created or exist before deleting it."
            )

        self.oci_transformation.delete()

    @classmethod
    def list_transformation(
        cls, compartment_id: str = None, **kwargs
    ) -> List["Transformation"]:
        """Lists Transformation resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        List["Transformation"]
            The list of the Transformation Resources.
        """

        return Transformation.list(compartment_id, **kwargs)

    @classmethod
    def list_transformations_df(
        cls, compartment_id: str = None, **kwargs
    ) -> "pandas.DataFrame":
        """Lists Transformation resources in a given compartment as pandas dataframe.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Entity.

        Returns
        -------
        "pandas.DataFrame"
            The list of the Transformation Resources.
        """

        return Transformation.list_df(compartment_id, **kwargs)

    def delete(self):
        """Removes FeatureStore Resource.

        Returns
        -------
        None
        """
        self.oci_fs.delete()

    def _update_from_oci_fs_model(self, oci_fs: OCIFeatureStore) -> "FeatureStore":
        """Update the properties from an OCIFeatureStore object.

        Parameters
        ----------
        oci_fs: OCIFeatureStore
            An instance of OCIFeatureStore.

        Returns
        -------
        FeatureStore
            The FeatureStore instance (self).
        """

        # Update the main properties
        self.oci_fs = oci_fs
        feature_store_details = oci_fs.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in feature_store_details:
                self.set_spec(infra_attr, feature_store_details[infra_attr])

        return self

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists featurestore resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the featurestore resources in a pandas dataframe format.
        """
        records = []
        for oci_feature_store in OCIFeatureStore.list_resource(
            compartment_id, **kwargs
        ):
            records.append(
                {
                    "id": oci_feature_store.id,
                    "display_name": oci_feature_store.display_name,
                    "description": oci_feature_store.description,
                    "time_created": oci_feature_store.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_feature_store.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_feature_store.lifecycle_state,
                    "created_by": f"...{oci_feature_store.created_by[-6:]}",
                    "compartment_id": f"...{oci_feature_store.compartment_id[-6:]}",
                    "offline_config": oci_feature_store.offline_config,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["FeatureStore"]:
        """Lists FeatureStore Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering FeatureStore.

        Returns
        -------
        List[FeatureStore]
            The list of the FeatureStore Resources.
        """
        return [
            cls()._update_from_oci_fs_model(oci_feature_store)
            for oci_feature_store in OCIFeatureStore.list_resource(
                compartment_id, **kwargs
            )
        ]

    def sql(
        self,
        query: str,
        dataframe_type: DataFrameType = DataFrameType.SPARK,
        is_online: Optional[bool] = False,
    ):
        """
        Execute the specified SQL query on the offline or online feature store database.

        Args:
            query: The SQL query to execute.
            dataframe_type: The type of DataFrame to be returned. Defaults to `DataFrameType.SPARK`.
            is_online: A flag indicating whether to execute the query against the online feature store.
                       Defaults to `False`.

        Returns:
            A DataFrame object that depends on the chosen type.
        """
        if self.id is None:
            raise ValueError(
                "Cannot query a FeatureStore resource that has not been created or saved."
            )

        if not self.spark_engine:
            self.spark_engine = SparkEngine(
                self.offline_config.get(self.CONST_METASTORE_ID)
                if self.offline_config
                else None
            )

        return self.spark_engine.sql(query, dataframe_type, is_online)

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes feature store  to a dictionary.

        Returns
        -------
        dict
            The feature store resource serialized as a dictionary.
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

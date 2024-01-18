#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from great_expectations.core import ExpectationSuite

from ads import deprecated
from ads.common import utils
from ads.common.decorator.runtime_dependency import OptionalDependency
from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.common.enums import (
    ExpectationType,
    EntityType,
    StreamingIngestionMode,
    BatchIngestionMode,
)
from ads.feature_store.common.exceptions import (
    NotMaterializedError,
)
from ads.feature_store.common.utils.base64_encoder_decoder import Base64EncoderDecoder
from ads.feature_store.common.utils.utility import (
    get_metastore_id,
    get_execution_engine_type,
    validate_delta_format_parameters,
    get_schema_from_df,
    convert_expectation_suite_to_expectation,
)
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine
from ads.feature_store.execution_strategy.execution_strategy_provider import (
    OciExecutionStrategyProvider,
    ExecutionEngine,
)
from ads.feature_store.feature import Feature
from ads.feature_store.feature_group_expectation import Expectation
from ads.feature_store.feature_group_job import FeatureGroupJob
from ads.feature_store.feature_option_details import FeatureOptionDetails
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.query.filter import Filter, Logic
from ads.feature_store.query.query import Query
from ads.feature_store.service.oci_feature_group import OCIFeatureGroup
from ads.feature_store.service.oci_feature_group_job import OCIFeatureGroupJob
from ads.feature_store.service.oci_lineage import OCILineage
from ads.feature_store.statistics.statistics import Statistics
from ads.feature_store.statistics_config import StatisticsConfig
from ads.feature_store.validation_output import ValidationOutput

from ads.jobs.builders.base import Builder
from ads.feature_store.feature_lineage.graphviz_service import (
    GraphService,
    GraphOrientation,
)

try:
    from pyspark.sql import DataFrame
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise

logger = logging.getLogger(__name__)


class FeatureGroup(Builder):
    """ Represents a FeatureGroup Resource.

    Methods
    -------
    create(self, **kwargs) -> "FeatureGroup"
        Creates feature group resource.
    delete(self) -> "FeatureGroup":
        Removes feature group resource.
    to_dict(self) -> dict
        Serializes feature group to a dictionary.
    from_id(cls, id: str) -> "Transformation"
        Gets an existing feature group resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["FeatureGroup"]
        Lists feature groups resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists feature groups resources as a pandas dataframe.
    with_description(self, description: str) -> "FeatureGroup"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "FeatureGroup"
        Sets the compartment ID.
    with_feature_store_id(self, feature_store_id: str) -> "FeatureGroup"
        Sets the feature store ID.
    with_name(self, name: str) -> "FeatureGroup"
        Sets the name.
    with_entity_id(self, entity_id: str) -> "FeatureGroup"
        Sets the entity id.
    with_input_feature_details(self, **schema_details: Dict[str, str]) -> "FeatureGroup"
        Sets the raw input feature details for the feature group.
    with_statistics_config(self, statistics_config: Union[StatisticsConfig, bool])   -> "FeatureGroup"
        Sets the statistics config details

    Examples
    --------
    >>> from ads.feature_store import feature_group
    >>> import oci
    >>> import os
    >>> input_feature_detail = [FeatureDetail("cc_num").with_feature_type(FeatureType.STRING).with_order_number(1)]
    >>> feature_group = feature_group.FeatureGroup()
    >>>     .with_description("feature group description")
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_name("FeatureGroup")
    >>>     .with_entity_id("<entity_id>") \
    >>>     .with_feature_store_id("<feature_store_id>") \
    >>>     .with_primary_keys(["key1", "key2"]) \
    >>>     .with_input_feature_details(input_feature_detail) \
    >>>     .with_statistics_config(StatisticsConfig(True,columns=["column1","column2"]))
    >>> feature_group.create()
    """

    _PREFIX = "featuregroup_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_DESCRIPTION = "description"
    CONST_FEATURE_STORE_ID = "featureStoreId"
    CONST_ENTITY_ID = "entityId"
    CONST_ITEMS = "items"
    CONST_PRIMARY_KEYS = "primaryKeys"
    CONST_PARTITION_KEYS = "partitionKeys"
    CONST_EXPECTATION_DETAILS = "expectationDetails"
    CONST_INPUT_FEATURE_DETAILS = "inputFeatureDetails"
    CONST_OUTPUT_FEATURE_DETAILS = "outputFeatureDetails"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_TRANSFORMATION_ID = "transformationId"
    CONST_STATISTICS_CONFIG = "statisticsConfig"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_LAST_JOB_ID = "jobId"
    CONST_INFER_SCHEMA = "isInferSchema"
    CONST_TRANSFORMATION_KWARGS = "transformationParameters"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_DESCRIPTION: "description",
        CONST_FEATURE_STORE_ID: "feature_store_id",
        CONST_ENTITY_ID: "entity_id",
        CONST_PRIMARY_KEYS: "primary_keys",
        CONST_EXPECTATION_DETAILS: "expectation_details",
        CONST_ITEMS: "items",
        CONST_INPUT_FEATURE_DETAILS: "input_feature_details",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_TRANSFORMATION_ID: "transformation_id",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_OUTPUT_FEATURE_DETAILS: "output_feature_details",
        CONST_STATISTICS_CONFIG: "statistics_config",
        CONST_INFER_SCHEMA: "is_infer_schema",
        CONST_PARTITION_KEYS: "partition_keys",
        CONST_TRANSFORMATION_KWARGS: "transformation_parameters",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes FeatureGroup Resource.

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
        # Specify oci FeatureGroup instance
        self.feature_group_job = None
        self._spark_engine = None
        self.oci_feature_group: OCIFeatureGroup = self._to_oci_feature_group(**kwargs)
        self.dsc_job = OCIFeatureGroupJob()
        self.lineage = OCILineage(**kwargs)

    def _to_oci_feature_group(self, **kwargs):
        """Creates an `OCIFeatureGroup` instance from the  `FeatureGroup`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `oci.feature_group.models.FeatureGroup` accepts.

        Returns
        -------
        OCIFeatureGroup
            The instance of the OCIFeatureGroup.
        """

        fs_spec = {}

        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCIFeatureGroup(**fs_spec)

    @property
    def spark_engine(self):
        if not self._spark_engine:
            self._spark_engine = SparkEngine(get_metastore_id(self.feature_store_id))
        return self._spark_engine

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "FeatureGroup"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "FeatureGroup":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, name: str):
        self.with_name(name)

    def with_name(self, name: str) -> "FeatureGroup":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name of feature group.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def id(self) -> str:
        """The id of the feature group.

        Returns
        -------
        str
            The id of the feature group.
        """
        return self.get_spec(self.CONST_ID)

    def with_id(self, id: str) -> "FeatureGroup":
        return self.set_spec(self.CONST_ID, id)

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "FeatureGroup":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the feature group.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def primary_keys(self) -> List[str]:
        return self.get_spec(self.CONST_PRIMARY_KEYS)

    @primary_keys.setter
    def primary_keys(self, value: List[str]):
        self.with_primary_keys(value)

    def with_primary_keys(self, primary_keys: List[str]) -> "FeatureGroup":
        """Sets the primary keys of the feature group.

        Parameters
        ----------
        primary_keys: str
            The description of the feature group.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(
            self.CONST_PRIMARY_KEYS,
            {
                self.CONST_ITEMS: [
                    {self.CONST_NAME: primary_key} for primary_key in primary_keys or []
                ]
            },
        )

    @property
    def transformation_kwargs(self) -> str:
        return self.get_spec(self.CONST_TRANSFORMATION_KWARGS)

    @transformation_kwargs.setter
    def transformation_kwargs(self, value: Dict):
        self.with_transformation_kwargs(value)

    def with_transformation_kwargs(
        self, transformation_kwargs: Dict = ()
    ) -> "FeatureGroup":
        """Sets the primary keys of the feature group.

        Parameters
        ----------
        transformation_kwargs: Dict
            Dictionary containing the transformation arguments.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(
            self.CONST_TRANSFORMATION_KWARGS,
            Base64EncoderDecoder.encode(json.dumps(transformation_kwargs or {})),
        )

    @property
    def partition_keys(self) -> List[str]:
        return self.get_spec(self.CONST_PARTITION_KEYS)

    @partition_keys.setter
    def partition_keys(self, value: List[str]):
        self.with_partition_keys(value)

    def with_partition_keys(self, partition_keys: List[str]) -> "FeatureGroup":
        """Sets the partition keys of the feature group.

        Parameters
        ----------
        partition_keys: List[str]
            The List of partition keys for the feature group.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(
            self.CONST_PARTITION_KEYS,
            {
                self.CONST_ITEMS: [
                    {self.CONST_NAME: partition_key}
                    for partition_key in partition_keys or []
                ]
            },
        )

    @property
    def feature_store_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STORE_ID)

    @feature_store_id.setter
    def feature_store_id(self, value: str):
        self.with_feature_store_id(value)

    def with_feature_store_id(self, feature_store_id: str) -> "FeatureGroup":
        """Sets the feature_store_id.

        Parameters
        ----------
        feature_store_id: str
            The feature_store_id.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_STORE_ID, feature_store_id)

    @property
    def transformation_id(self) -> str:
        return self.get_spec(self.CONST_TRANSFORMATION_ID)

    @transformation_id.setter
    def transformation_id(self, value: str):
        self.with_feature_store_id(value)

    def with_transformation_id(self, transformation_id: str) -> "FeatureGroup":
        """Sets the transformation_id.

        Parameters
        ----------
        transformation_id: str
            The transformation_id.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """

        # Initialize the empty dictionary as transformation arguemnts if not specified
        if not self.transformation_kwargs:
            self.with_transformation_kwargs()

        return self.set_spec(self.CONST_TRANSFORMATION_ID, transformation_id)

    def _with_lifecycle_state(self, lifecycle_state: str) -> "FeatureGroup":
        """Sets the lifecycle_state.

        Parameters
        ----------
        lifecycle_state: str
            The lifecycle_state.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_LIFECYCLE_STATE, lifecycle_state)

    @property
    def entity_id(self) -> str:
        return self.get_spec(self.CONST_ENTITY_ID)

    @entity_id.setter
    def entity_id(self, value: str):
        self.with_entity_id(value)

    def with_entity_id(self, entity_id: str) -> "FeatureGroup":
        """Sets the entity_id.

        Parameters
        ----------
        entity_id: str
            The entity_id.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_ENTITY_ID, entity_id)

    @property
    def expectation_details(self) -> "Expectation":
        """The expectation details of the feature group.

        Returns
        -------
        list
            The step details of the feature group.
        """
        return self.get_spec(self.CONST_EXPECTATION_DETAILS)

    def with_expectation_suite(
        self, expectation_suite: ExpectationSuite, expectation_type: ExpectationType
    ) -> "FeatureGroup":
        """Sets the expectation details for the feature group.

        Parameters
        ----------
        expectation_suite: ExpectationSuite
            A list of rules in the feature store.
        expectation_type: ExpectationType
            Type of the expectation.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self).
        """
        return self.set_spec(
            self.CONST_EXPECTATION_DETAILS,
            convert_expectation_suite_to_expectation(
                expectation_suite, expectation_type
            ).to_dict(),
        )

    @property
    def input_feature_details(self) -> list:
        return self.get_spec(self.CONST_INPUT_FEATURE_DETAILS)

    @input_feature_details.setter
    def input_feature_details(self, input_feature_details: List[FeatureDetail]):
        self.with_input_feature_details(input_feature_details)

    def with_input_feature_details(
        self, input_feature_details: List[FeatureDetail]
    ) -> "FeatureGroup":
        """Sets the input feature details.

        Parameters
        ----------
        input_feature_details: List[FeatureDetail]
            The input_feature_details for the Feature Group.
        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        if not self.is_infer_schema:
            self.with_is_infer_schema(False)
        return self.set_spec(
            self.CONST_INPUT_FEATURE_DETAILS,
            [feature_details.to_dict() for feature_details in input_feature_details],
        )

    def with_schema_details_from_dataframe(
        self, data_frame: Union[DataFrame, pd.DataFrame]
    ) -> "FeatureGroup":
        if not self.feature_store_id:
            raise ValueError(
                "FeatureStore id must be set before calling `with_schema_details_from_dataframe`"
            )

        schema_details = get_schema_from_df(data_frame, self.feature_store_id)
        feature_details = []

        for schema_detail in schema_details:
            feature_details.append(FeatureDetail(**schema_detail))
        self.with_is_infer_schema(True)
        return self.with_input_feature_details(feature_details)

    def _with_features(self, features: List[Feature]):
        """Sets the output_features.

        Parameters
        ----------
        features: List[Feature]
            The features for the Feature Group.
        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(
            self.CONST_OUTPUT_FEATURE_DETAILS,
            {self.CONST_ITEMS: [feature.to_dict() for feature in features]},
        )

    @property
    def statistics_config(self) -> "StatisticsConfig":
        """The statistics config deatils of the feature group.

        Returns
        -------
        list
            The step details of the feature group.
        """
        return self.get_spec(self.CONST_STATISTICS_CONFIG)

    @statistics_config.setter
    def statistics_config(self, statistics_config: Union[StatisticsConfig, bool]):
        self.with_statistics_config(statistics_config)

    def with_statistics_config(
        self, statistics_config: Union[StatisticsConfig, bool]
    ) -> "FeatureGroup":
        """Sets the expectation details for the feature group.

        Parameters
        ----------
        statistics_config: StatisticsConfig
            statistics config

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self).
        """
        if isinstance(statistics_config, StatisticsConfig):
            statistics_config_in = statistics_config
        elif isinstance(statistics_config, bool):
            statistics_config_in = StatisticsConfig(statistics_config)
        else:
            raise TypeError(
                "The argument `statistics_config` has to be of type `StatisticsConfig` or `bool`, "
                "but is of type: `{}`".format(type(statistics_config))
            )
        return self.set_spec(
            self.CONST_STATISTICS_CONFIG, statistics_config_in.to_dict()
        )

    @property
    def features(self) -> List[Feature]:
        return [
            Feature(**feature_dict)
            for feature_dict in self.get_spec(self.CONST_OUTPUT_FEATURE_DETAILS)[
                self.CONST_ITEMS
            ]
            or []
        ]

    def with_job_id(self, feature_group_job_id: str) -> "FeatureGroup":
        """Sets the job_id for the last running job.

        Parameters
        ----------
        feature_group_job_id: str
            FeatureGroup job id.
        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_LAST_JOB_ID, feature_group_job_id)

    @property
    def is_infer_schema(self) -> bool:
        return self.get_spec(self.CONST_INFER_SCHEMA)

    @is_infer_schema.setter
    def is_infer_schema(self, value: bool):
        self.with_is_infer_schema(value)

    def with_is_infer_schema(self, is_infer_schema: bool) -> "FeatureGroup":
        """Sets the job_id for the last running job.

        Parameters
        ----------
        is_infer_schema: bool
            Infer Schema or not.
        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)
        """
        return self.set_spec(self.CONST_INFER_SCHEMA, is_infer_schema)

    def target_delta_table(self):
        """
        Returns the fully-qualified name of the target table for storing delta data.

        The name of the target table is constructed by concatenating the entity ID
        and the name of the table, separated by a dot. The resulting string has the
        format 'entity_id.table_name'.

        Returns:
            str: The fully-qualified name of the target delta table.
        """
        target_table = f"{self.entity_id}.{self.name}"
        return target_table

    @property
    def job_id(self) -> str:
        return self.get_spec(self.CONST_LAST_JOB_ID)

    def create(self, **kwargs) -> "FeatureGroup":
        """Creates feature group  resource.

        !!! note "Lazy"
            This method is lazy and does not persist any metadata or feature data in the
            feature store on its own. To persist the feature group and save feature data
            along the metadata in the feature store, call the `materialise()` method with a
            DataFrame or with a Datasource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureGroup` accepts.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """
        self.compartment_id = OCIModelMixin.check_compartment_id(self.compartment_id)

        if not self.feature_store_id:
            raise ValueError("FeatureStore id must be provided.")

        if not self.entity_id:
            raise ValueError("Entity id must be provided.")

        if not self.name:
            self.name = self._random_display_name()

        if self.statistics_config is None:
            self.statistics_config = StatisticsConfig()

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a feature group resource with payload {payload}")

        # Create feature group
        logger.info("Saving feature group.")
        self.oci_feature_group = self._to_oci_feature_group(**kwargs).create()
        self.with_id(self.oci_feature_group.id)
        return self

    def get_features(self) -> List[Feature]:
        """
        Returns all the features in the feature group.

        Returns:
            List[Feature]
        """

        return self.features

    def get_features_df(self) -> "pd.DataFrame":
        """
        Returns all the features as pandas dataframe.

        Returns:
            pandas.DataFrame
        """
        records = []
        for feature in self.features:
            records.append(
                {
                    "name": feature.feature_name,
                    "type": feature.feature_type,
                }
            )
        return pd.DataFrame.from_records(records)

    def get_input_features_df(self) -> "pd.DataFrame":
        """
        Returns all the input features details as pandas dataframe.

        Returns:
            pandas.DataFrame
        """
        records = []
        for input_feature in self.input_feature_details:
            records.append(
                {
                    "name": input_feature.get("name"),
                    "type": input_feature.get("featureType"),
                    "order_number": input_feature.get("orderNumber"),
                    "is_event_timestamp": input_feature.get("isEventTimestamp"),
                    "event_timestamp_format": input_feature.get("eventTimestampFormat"),
                }
            )
        return pd.DataFrame.from_records(records)

    def update(self, **kwargs) -> "FeatureGroup":
        """Updates FeatureGroup in the feature store.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureGroup` accepts.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self).
        """

        if not self.id:
            raise ValueError(
                "FeatureGroup needs to be saved to the feature store before it can be updated."
            )

        self.oci_feature_group = self._to_oci_feature_group(**kwargs).update()
        return self

    def _update_from_oci_feature_group_model(
        self, oci_feature_group: OCIFeatureGroup
    ) -> "FeatureGroup":
        """Update the properties from an OCIFeatureGroup object.

        Parameters
        ----------
        oci_feature_group: OCIFeatureGroup
            An instance of OCIFeatureGroup.

        Returns
        -------
        FeatureGroup
            The FeatureGroup instance (self).
        """

        # Update the main properties
        self.oci_feature_group = oci_feature_group
        feature_group_details = oci_feature_group.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in feature_group_details:
                if infra_attr == self.CONST_OUTPUT_FEATURE_DETAILS:
                    # May not need if we fix the backend and add feature_group_id to the output_feature
                    features_list = []
                    for output_feature in feature_group_details[infra_attr]["items"]:
                        output_feature["featureGroupId"] = feature_group_details[
                            self.CONST_ID
                        ]
                        features_list.append(output_feature)

                    value = {self.CONST_ITEMS: features_list}
                else:
                    value = feature_group_details[infra_attr]

                self.set_spec(infra_attr, value)

        return self

    def _build_feature_group_job(
        self,
        ingestion_mode,
        from_timestamp: str = None,
        to_timestamp: str = None,
        feature_option_details=None,
    ):
        feature_group_job = (
            FeatureGroupJob()
            .with_feature_group_id(self.id)
            .with_compartment_id(self.compartment_id)
            .with_ingestion_mode(ingestion_mode)
            .with_time_from(from_timestamp)
            .with_time_to(to_timestamp)
        )

        if feature_option_details:
            feature_group_job = feature_group_job.with_feature_option_details(
                feature_option_details
            )

        return feature_group_job

    def materialise(
        self,
        input_dataframe: Union[DataFrame, pd.DataFrame],
        ingestion_mode: BatchIngestionMode = BatchIngestionMode.OVERWRITE,
        from_timestamp: str = None,
        to_timestamp: str = None,
        feature_option_details: FeatureOptionDetails = None,
    ):
        """
        Executes a feature group job to materialize feature data into the feature store.

        Args:
            input_dataframe. A pandas/spark DataFrame containing the input data for the feature group job.
            ingestion_mode: Optional. An instance of the IngestionMode enum indicating how to ingest the data into the feature store.
            from_timestamp: Optional. A string representing the lower bound of the time range of data to include in the job.
            to_timestamp: Optional. A string representing the upper bound of the time range of data to include in the job.
            feature_option_details: Optional. An instance of the FeatureOptionDetails class containing feature options.

        Returns:
            None. This method does not return anything.

        Raises:
            Any exceptions thrown by the underlying execution strategy or feature store.

        """

        # Create Feature Definition Job and persist it
        feature_group_job = self._build_feature_group_job(
            ingestion_mode=ingestion_mode,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            feature_option_details=feature_option_details,
        )

        # Create the Job
        feature_group_job.create()
        # Update the feature group with corresponding job so that user can see the details about the job
        self.with_job_id(feature_group_job.id)

        feature_group_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=get_execution_engine_type(input_dataframe),
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        feature_group_execution_strategy.ingest_feature_definition(
            self, feature_group_job, input_dataframe
        )

    def materialise_stream(
        self,
        input_dataframe: Union[DataFrame],
        checkpoint_dir: str,
        query_name: Optional[str] = None,
        ingestion_mode: StreamingIngestionMode = StreamingIngestionMode.APPEND,
        await_termination: Optional[bool] = False,
        timeout: Optional[int] = None,
        feature_option_details: FeatureOptionDetails = None,
    ):
        """Ingest a Spark Structured Streaming Dataframe to the feature store.

        This method creates a long running Spark Streaming Query, you can control the
        termination of the query through the arguments.

        It is possible to stop the returned query with the `.stop()` and check its
        status with `.isActive`.

        !!! warning "Engine Support"
            **Spark only**

            Stream ingestion using Pandas/Python as engine is currently not supported.
            Python/Pandas has no notion of streaming.

        !!! warning "Data Validation Support"
            `materialise_stream` does not perform any data validation using Great Expectations
            even when a expectation suite is attached.

        # Arguments
            input_dataframe: Features in Streaming Dataframe to be saved.
            query_name: It is possible to optionally specify a name for the query to
                make it easier to recognise in the Spark UI. Defaults to `None`.
            ingestion_mode: Specifies how data of a streaming DataFrame/Dataset is
                written to a streaming sink. (1) `"append"`: Only the new rows in the
                streaming DataFrame/Dataset will be written to the sink. (2)
                `"complete"`: All the rows in the streaming DataFrame/Dataset will be
                written to the sink every time there is some update. (3) `"update"`:
                only the rows that were updated in the streaming DataFrame/Dataset will
                be written to the sink every time there are some updates.
                If the query doesn’t contain aggregations, it will be equivalent to
                append mode. Defaults to `"append"`.
            await_termination: Waits for the termination of this query, either by
                query.stop() or by an exception. If the query has terminated with an
                exception, then the exception will be thrown. If timeout is set, it
                returns whether the query has terminated or not within the timeout
                seconds. Defaults to `False`.
            timeout: Only relevant in combination with `await_termination=True`.
                Defaults to `None`.
            checkpoint_dir: Checkpoint directory location. This will be used to as a reference to
                from where to resume the streaming job.

        # Returns
            `StreamingQuery`: Spark Structured Streaming Query object.
        """

        # Create Feature Definition Job and persist it
        feature_group_job = self._build_feature_group_job(
            ingestion_mode=ingestion_mode,
            feature_option_details=feature_option_details,
        )

        # Create the Job
        feature_group_job.create()

        # Update the feature group with corresponding job so that user can see the details about the job
        self.with_job_id(feature_group_job.id)

        feature_group_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=get_execution_engine_type(input_dataframe),
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        return feature_group_execution_strategy.ingest_feature_definition_stream(
            self,
            feature_group_job,
            input_dataframe,
            query_name,
            await_termination,
            timeout,
            checkpoint_dir,
        )

    def get_last_job(self) -> "FeatureGroupJob":
        """Gets the Job details for the last running job.

        Returns:
            FeatureGroupJob
        """

        if not self.id:
            raise ValueError(
                "FeatureGroup needs to be saved to the feature store before getting associated jobs."
            )

        if not self.job_id:
            fg_job = FeatureGroupJob.list(
                feature_group_id=self.id,
                compartment_id=self.compartment_id,
                sort_by="timeCreated",
                limit="1",
            )
            if not fg_job:
                raise ValueError(
                    "Unable to retrieve the associated last job. Please make sure you materialized the data."
                )
            self.with_job_id(fg_job[0].id)
            return fg_job[0]
        return FeatureGroupJob.from_id(self.job_id)

    def select(self, features: Optional[List[str]] = ()) -> Query:
        """
        Selects a subset of features from the feature group and returns a Query object that can be used to view the
        resulting dataframe.

        Args:
            features (Optional[List[str]], optional): A list of feature names to be selected. Defaults to [].

        Returns:
            Query: A Query object that includes the selected features from the feature group.
        """
        self.check_resource_materialization()

        if features:
            self.__validate_features_exist(features)

        return Query(
            left_feature_group=self,
            left_features=features,
            feature_store_id=self.feature_store_id,
            entity_id=self.entity_id,
        )

    def delete(self):
        """Removes FeatureGroup Resource.
        Returns
        -------
        None
        """
        # Create Feature Definition Job and persist it
        feature_group_job = self._build_feature_group_job(BatchIngestionMode.DEFAULT)

        # Create the Job
        feature_group_job.create()
        feature_group_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=ExecutionEngine.SPARK,
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        feature_group_execution_strategy.delete_feature_definition(
            self, feature_group_job
        )

    def filter(self, f: Union[Filter, Logic]):
        """Apply filter to the feature group.

        Selects all features and returns the resulting `Query` with the applied filter.

        ```python
        fg.filter((fg.feature1 == 1) | (fg.feature2 >= 2))
        ```

        # Arguments
            f: Filter object.

        # Returns
            `Query`. The query object with the applied filter.
        """
        return self.select().filter(f)

    @deprecated(details="preview functionality is deprecated. Please use as_of.")
    def preview(
        self,
        row_count: int = 10,
        version_number: int = None,
        timestamp: datetime = None,
    ):
        """preview the feature definition and return the response in dataframe.

        Parameters
        ----------
        timestamp: datetime
            commit date time to preview in format yyyy-MM-dd or yyyy-MM-dd HH:mm:ss
            commit date time is maintained for every ingestion commit using delta lake
        version_number: int
            commit version number for the preview. Version numbers are automatically versioned for every ingestion
            commit using delta lake
        row_count: int
            preview row count

        Returns
        -------
        spark dataframe
            The preview result in spark dataframe
        """
        self.check_resource_materialization()

        validate_delta_format_parameters(timestamp, version_number)
        target_table = self.target_delta_table()

        if version_number is not None:
            logger.warning("Time travel queries are not supported in current version")

        sql_query = f"select * from {target_table} LIMIT {row_count}"

        return self.spark_engine.sql(sql_query)

    def as_of(
        self,
        version_number: int = None,
        commit_timestamp: datetime = None,
    ):
        """preview the feature definition and return the response in dataframe.

        Parameters
        ----------
        commit_timestamp: datetime
            commit date time to preview in format yyyy-MM-dd or yyyy-MM-dd HH:mm:ss
            commit date time is maintained for every ingestion commit using delta lake
        version_number: int
            commit version number for the preview. Version numbers are automatically versioned for every ingestion
            commit using delta lake

        Returns
        -------
        spark dataframe
            The preview result in spark dataframe
        """
        self.check_resource_materialization()

        validate_delta_format_parameters(commit_timestamp, version_number)
        target_table = self.target_delta_table()

        return self.spark_engine.get_time_version_data(
            target_table, version_number, commit_timestamp
        )

    def profile(self):
        """get the profile information for feature definition and return the response in dataframe.

        Returns
        -------
        spark dataframe
            The profile result in spark dataframe
        """
        self.check_resource_materialization()

        sql_query = f"DESCRIBE DETAIL {self.target_delta_table()}"

        return self.spark_engine.sql(sql_query)

    def restore(self, version_number: int = None, timestamp: datetime = None):
        """restore  the feature definition and return the response in dataframe.

        Parameters
        ----------
        timestamp: datetime
            commit date time to restore in format yyyy-MM-dd or yyyy-MM-dd HH:mm:ss.
             commit date time is maintained for every ingestion commit using delta lake
        version_number: int
            commit version number to restore. Version numbers are automatically versioned for every ingestion
            commit using delta lake
        Returns
        -------
        spark dataframe
            The restore output as spark dataframe
        """
        self.check_resource_materialization()

        validate_delta_format_parameters(timestamp, version_number, True)
        target_table = self.target_delta_table()
        if version_number is not None:
            sql_query = (
                f"RESTORE TABLE {target_table} TO VERSION AS OF {version_number}"
            )
        else:
            sql_query = f"RESTORE TABLE {target_table} TO TIMESTAMP AS OF {timestamp}"

        restore_output = self.spark_engine.sql(sql_query)

        feature_group_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=ExecutionEngine.SPARK,
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        feature_group_execution_strategy.update_feature_definition_features(
            self, target_table
        )

        return restore_output

    def check_resource_materialization(self):
        """Checks whether the target Delta table for this resource has been materialized in Spark.
        If the target Delta table doesn't exist, raises a NotMaterializedError with the type and name of this resource.
        """
        if not self.spark_engine.is_delta_table_exists(self.target_delta_table()):
            raise NotMaterializedError(self.type, self.name)

    def history(self):
        """get the feature definition commit history.

        Returns
        -------
        spark dataframe
            The history output as spark dataframe
        """
        target_table = self.target_delta_table()
        sql_query = f"DESCRIBE HISTORY {target_table}"
        return self.spark_engine.sql(sql_query)

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pd.DataFrame":
        """Lists FeatureGroup resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the FeatureGroup resources in a pandas dataframe format.
        """
        records = []
        for oci_feature_group in OCIFeatureGroup.list_resource(
            compartment_id, **kwargs
        ):
            oci_feature_group: OCIFeatureGroup = oci_feature_group
            records.append(oci_feature_group.to_df_record())

        return pd.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["FeatureGroup"]:
        """Lists FeatureGroup Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering FeatureGroup.

        Returns
        -------
        List[FeatureGroup]
            The list of the FeatureGroup Resources.
        """
        return [
            cls()._update_from_oci_feature_group_model(oci_feature_group)
            for oci_feature_group in OCIFeatureGroup.list_resource(
                compartment_id, **kwargs
            )
        ]

    @classmethod
    def from_id(cls, id: str) -> "FeatureGroup":
        """Gets an existing feature group resource by Id.

        Parameters
        ----------
        id: str
            The feature group id.

        Returns
        -------
        FeatureGroup
            An instance of FeatureGroup resource.
        """
        return cls()._update_from_oci_feature_group_model(OCIFeatureGroup.from_id(id))

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes feature group  to a dictionary.

        Returns
        -------
        dict
            The feature group resource serialized as a dictionary.
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

    def show(self, rankdir: str = GraphOrientation.LEFT_RIGHT) -> None:
        """
        Show the lineage tree for the feature_group instance.

        Raises:
            ValueError: If  lineage graph cannot be plotted due to missing lineage information.
        """
        lineage = self.lineage.from_id(self.id)
        if lineage:
            GraphService.view_lineage(lineage.data, EntityType.FEATURE_GROUP, rankdir)
        else:
            raise ValueError(
                f"Can't get lineage information for Feature group id {self.id}"
            )

    def __validate_features_exist(self, features: List[str]) -> None:
        """
        Validates whether each feature in the input list is present in the output features list.

        Args:
            features (List[str]): A list of feature names to validate.

        Raises:
            ValueError: If any feature in the input list is not present in the output features list.
        """
        # Get a list of output feature names
        output_feature_names = [
            output_feature.feature_name for output_feature in self.features
        ]

        # Initialize an empty list to store non-existing features
        non_existing_features = []

        # Check if each feature in the input list is present in the output features list
        for feature in features:
            if feature not in output_feature_names:
                non_existing_features.append(feature)

        # If there are any non-existing features, raise a ValueError
        if len(non_existing_features) != 0:
            raise ValueError(
                f"Features {non_existing_features} are not defined in the feature group."
            )

    def get_feature(self, name: str):
        """Retrieve a `Feature` object from the schema of the feature group.

        There are several ways to access features of a feature group:

        ```python
        fg.feature1
        fg.get_feature("feature1")
        ```

        Args:
            name (str): [description]

        Returns:
            [type]: [description]
        """
        try:
            return self.__getitem__(name)
        except KeyError:
            raise ValueError(f"'FeatureGroup' object has no feature called '{name}'.")

    def _get_job_id(self, job_id: str = None) -> str:
        """
        Helper function to determine the job ID based on the given input or the last run job.

        Args:
            job_id (str): Job ID provided by the user.

        Returns:
            str: Job ID to be used.
        """
        if job_id is not None:
            return job_id

        if self.job_id is None:
            raise ValueError(
                "Unable to retrieve the last job. Please provide the job ID and make sure you materialized the data."
            )

        return self.job_id

    def get_statistics(self, job_id: str = None) -> "Statistics":
        """Retrieve Statistics object for the job with job_id
        if job_id is not specified the last run job will be considered.
        Args:
            job_id (str): [job id of the job for which the statistics need to be calculated]

        Returns:
            [type]: [Statistics]
        """

        if not self.id:
            raise ValueError(
                "FeatureGroup needs to be saved to the feature store before retrieving the statistics"
            )

        stat_job_id = job_id if job_id is not None else self.get_last_job().id

        # TODO: take the one in memory or will list down job ids and find the latest
        fg_job = FeatureGroupJob.from_id(stat_job_id)
        if self.id != fg_job.feature_group_id:
            raise ValueError(
                "The specified job id does not belong to this feature group"
            )
        output_details = fg_job.job_output_details
        feature_statistics = (
            output_details.get("featureStatistics") if output_details else None
        )
        stat_version = output_details.get("version") if output_details else None
        version = stat_version if stat_version is not None else 1

        return Statistics(feature_statistics, version)

    def get_validation_output(self, job_id: str = None) -> "ValidationOutput":
        """Retrieve validation report for the job with job_id
        if job_id is not specified the last run job will be considered.
        Args:
            job_id (str): [job id of the job for which the validation report need to be retrieved]

        Returns:
            ValidationOutput
        """

        if not self.id:
            raise ValueError(
                "FeatureGroup needs to be saved to the feature store before retrieving the validation report"
            )

        validation_job_id = job_id if job_id is not None else self.get_last_job().id

        # Retrieve the validation output JSON from data_flow_batch_execution_output.
        fg_job = FeatureGroupJob.from_id(validation_job_id)
        output_details = fg_job.job_output_details
        validation_output = (
            output_details.get("validationOutput") if output_details else None
        )

        return ValidationOutput(validation_output)

    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(
                f"'FeatureGroup' object has no attribute '{name}'. "
                "If you are trying to access a feature, fall back on "
                "using the `get_feature` method."
            )

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError(
                f"Expected type `str`, got `{type(name)}`. "
                "Features are accessible by name."
            )
        feature = [
            feature
            for feature in self.__getattribute__("features")
            if feature.feature_name == name
        ]
        if len(feature) == 1:
            return feature[0]
        else:
            raise KeyError(f"'FeatureGroup' object has no feature called '{name}'.")

#!/usr/bin/env python
# -*- coding: utf-8; -*-
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Union

import pandas
import pandas as pd
from great_expectations.core import ExpectationSuite

from ads import deprecated
from feature_store_client.feature_store.models import (
    DatasetFeatureGroupCollection,
    DatasetFeatureGroupSummary,
)

from ads.common import utils
from ads.common.oci_mixin import OCIModelMixin
from ads.feature_store.common.enums import (
    ExecutionEngine,
    ExpectationType,
    EntityType,
    BatchIngestionMode,
)
from ads.feature_store.common.exceptions import NotMaterializedError
from ads.feature_store.common.utils.utility import (
    get_metastore_id,
    validate_delta_format_parameters,
    convert_expectation_suite_to_expectation,
)
from ads.feature_store.dataset_job import DatasetJob
from ads.feature_store.execution_strategy.engine.spark_engine import SparkEngine
from ads.feature_store.execution_strategy.execution_strategy_provider import (
    OciExecutionStrategyProvider,
)
from ads.feature_store.feature import DatasetFeature
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.feature_group_expectation import Expectation
from ads.feature_store.feature_option_details import FeatureOptionDetails
from ads.feature_store.service.oci_dataset import OCIDataset
from ads.feature_store.statistics.statistics import Statistics
from ads.feature_store.statistics_config import StatisticsConfig
from ads.feature_store.service.oci_lineage import OCILineage
from ads.feature_store.model_details import ModelDetails
from ads.jobs.builders.base import Builder
from ads.feature_store.feature_lineage.graphviz_service import (
    GraphService,
    GraphOrientation,
)
from ads.feature_store.validation_output import ValidationOutput

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

logger = logging.getLogger(__name__)


class Dataset(Builder):
    """ Represents a Dataset Resource.

    Methods
    -------
    create(self, **kwargs) -> "Dataset"
        Creates dataset resource.
    delete(self) -> "Dataset":
        Removes dataset resource.
    to_dict(self) -> dict
        Serializes dataset to a dictionary.
    from_id(cls, id: str) -> "Dataset"
        Gets an existing dataset resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["Dataset"]
        Lists dataset resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists datasets resources as a pandas dataframe.
    with_description(self, description: str) -> "Dataset"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "Dataset"
        Sets the compartment ID.
    with_feature_store_id(self, feature_store_id: str) -> "Dataset"
        Sets the feature store ID.
    with_entity_id(self, entity_id: str) -> "Dataset"
        Sets the entity ID.
    with_query(self, query: str) -> "Dataset"
        Sets the SQL query.
    with_dataset_ingestion_mode(self, dataset_ingestion_mode: str) -> "Dataset"
        Sets the ingestion mode for dataset.
    with_statistics_config(self, statistics_config: Union[StatisticsConfig, bool]) -> "Dataset"
        Sets the statistics config details
    Examples
    --------
    >>> from ads.feature_store import dataset
    >>> import oci
    >>> import os
    >>> dataset = dataset.Dataset()
    >>>     .with_description("dataset description")
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_name("Dataset")
    >>>     .with_entity_id("<entity id>") \
    >>>     .with_feature_store_id("<feature_store_id>") \
    >>>     .with_query('SELECT feature_gr_1.name FROM feature_gr_1') \
    >>>     .with_statistics_config(StatisticsConfig(True,columns=["column1","column2"]))
    >>> dataset.create()
    """

    _PREFIX = "Dataset_resource"
    CONST_LINEAGE_CONSTRUCT_TYPE = "construct_type"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_NAME = "name"
    CONST_QUERY = "query"
    CONST_ENTITY_ID = "entityId"
    CONST_FEATURE_STORE_ID = "featureStoreId"
    CONST_DESCRIPTION = "description"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_PARTITION_KEYS = "partitionKeys"
    CONST_OUTPUT_FEATURE_DETAILS = "outputFeatureDetails"
    CONST_EXPECTATION_DETAILS = "expectationDetails"
    CONST_STATISTICS_CONFIG = "statisticsConfig"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_ITEMS = "items"
    CONST_LAST_JOB_ID = "jobId"
    CONST_MODEL_DETAILS = "modelDetails"
    CONST_FEATURE_GROUP = "datasetFeatureGroups"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_NAME: "name",
        CONST_FEATURE_STORE_ID: "feature_store_id",
        CONST_ENTITY_ID: "entity_id",
        CONST_QUERY: "query",
        CONST_DESCRIPTION: "description",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_EXPECTATION_DETAILS: "expectation_details",
        CONST_STATISTICS_CONFIG: "statistics_config",
        CONST_OUTPUT_FEATURE_DETAILS: "output_feature_details",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_MODEL_DETAILS: "model_details",
        CONST_PARTITION_KEYS: "partition_keys",
        CONST_FEATURE_GROUP: "dataset_feature_groups",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes Dataset Resource.

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
        # Specify oci Dataset instance
        self.dataset_job = None
        self._is_manual_association: bool = False
        self._spark_engine = None
        self.oci_dataset = self._to_oci_dataset(**kwargs)
        self.lineage = OCILineage(**kwargs)

    def _to_oci_dataset(self, **kwargs) -> OCIDataset:
        """Creates an `OCIDataset` instance from the  `Dataset`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `oci.dataset.models.Dataset` accepts.

        Returns
        -------
        OCIDataset
            The instance of the OCIDataset.
        """
        fs_spec = {}

        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value

        fs_spec.update(**kwargs)

        return OCIDataset(**fs_spec)

    @property
    def spark_engine(self):
        if not self._spark_engine:
            self._spark_engine = SparkEngine(get_metastore_id(self.feature_store_id))
        return self._spark_engine

    @property
    def is_manual_association(self):
        collection: DatasetFeatureGroupCollection = self.get_spec(
            self.CONST_FEATURE_GROUP
        )
        if collection and collection.is_manual_association is not None:
            return collection.is_manual_association
        else:
            return self._is_manual_association

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "Dataset"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "Dataset":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def name(self) -> str:
        return self.get_spec(self.CONST_NAME)

    @name.setter
    def name(self, name: str):
        self.with_name(name)

    def with_name(self, name: str) -> "Dataset":
        """Sets the name.

        Parameters
        ----------
        name: str
            The name of dataset.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_NAME, name)

    @property
    def id(self) -> str:
        """The id of the dataset.

        Returns
        -------
        str
            The id of the dataset.
        """
        return self.get_spec(self.CONST_ID)

    @property
    def features(self) -> List[DatasetFeature]:
        return [
            DatasetFeature(**feature_dict)
            for feature_dict in self.get_spec(self.CONST_OUTPUT_FEATURE_DETAILS)[
                self.CONST_ITEMS
            ]
            or []
        ]

    def with_id(self, id: str) -> "Dataset":
        return self.set_spec(self.CONST_ID, id)

    def with_job_id(self, dataset_job_id: str) -> "Dataset":
        """Sets the job_id for the last running job.

        Parameters
        ----------
        dataset_job_id: str
            Dataset job id.
        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_LAST_JOB_ID, dataset_job_id)

    @property
    def job_id(self) -> str:
        return self.get_spec(self.CONST_LAST_JOB_ID)

    @property
    def query(self) -> str:
        return self.get_spec(self.CONST_QUERY)

    @query.setter
    def query(self, query: str):
        self.with_query(query)

    def with_query(self, query: str) -> "Dataset":
        """Sets the dataset query.

        Parameters
        ----------
        query: str
            SQL Query that will be used to join the FeatureGroups to form the dataset.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_QUERY, query)

    def _with_lifecycle_state(self, lifecycle_state: str) -> "Dataset":
        """Sets the lifecycle_state.

        Parameters
        ----------
        lifecycle_state: str
            The lifecycle_state.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_LIFECYCLE_STATE, lifecycle_state)

    def _with_features(self, features: List[DatasetFeature]):
        """Sets the output_features.

        Parameters
        ----------
        features: List[DatasetFeature]
            The features for the Dataset.
        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(
            self.CONST_OUTPUT_FEATURE_DETAILS,
            {self.CONST_ITEMS: [feature.to_dict() for feature in features]},
        )

    @property
    def description(self) -> str:
        return self.get_spec(self.CONST_DESCRIPTION)

    @description.setter
    def description(self, value: str):
        self.with_description(value)

    def with_description(self, description: str) -> "Dataset":
        """Sets the description.

        Parameters
        ----------
        description: str
            The description of the dataset.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_DESCRIPTION, description)

    @property
    def feature_store_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STORE_ID)

    @feature_store_id.setter
    def feature_store_id(self, value: str):
        self.with_feature_store_id(value)

    def with_feature_store_id(self, feature_store_id: str) -> "Dataset":
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
    def expectation_details(self) -> "Expectation":
        """The expectation details of the dataset.

        Returns
        -------
        list
            The step details of the feature group.
        """
        return self.get_spec(self.CONST_EXPECTATION_DETAILS)

    def with_expectation_suite(
        self, expectation_suite: ExpectationSuite, expectation_type: ExpectationType
    ) -> "Dataset":
        """Sets the expectation details for the feature group.

        Parameters
        ----------
        expectation_suite: ExpectationSuite
            A list of rules in the feature store.
        expectation_type: ExpectationType
            Type of the expectation.

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """
        return self.set_spec(
            self.CONST_EXPECTATION_DETAILS,
            convert_expectation_suite_to_expectation(
                expectation_suite, expectation_type
            ).to_dict(),
        )

    @property
    def entity_id(self) -> str:
        return self.get_spec(self.CONST_ENTITY_ID)

    @entity_id.setter
    def entity_id(self, value: str):
        self.with_entity_id(value)

    @classmethod
    def from_id(cls, id: str) -> "Dataset":
        """Gets an existing dataset resource by Id.

        Parameters
        ----------
        id: str
            The dataset id.

        Returns
        -------
        Dataset
            An instance of Dataset resource.
        """
        return cls()._update_from_oci_dataset_model(OCIDataset.from_id(id))

    def with_entity_id(self, entity_id: str) -> "Dataset":
        """Sets the entity_id.

        Parameters
        ----------
        entity_id: str
            The entity_id.

        Returns
        -------
        Dataset
            The Dataset instance (self)
        """
        return self.set_spec(self.CONST_ENTITY_ID, entity_id)

    @property
    def statistics_config(self) -> "StatisticsConfig":
        return self.get_spec(self.CONST_STATISTICS_CONFIG)

    @statistics_config.setter
    def statistics_config(self, statistics_config: StatisticsConfig):
        self.with_statistics_config(statistics_config)

    def with_statistics_config(
        self, statistics_config: Union[StatisticsConfig, bool]
    ) -> "Dataset":
        """Sets the statistics details for the dataset.

        Parameters
        ----------
        statistics_config: StatisticsConfig
            statistics config

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """
        statistics_config_in = None
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
    def model_details(self) -> "ModelDetails":
        return self.get_spec(self.CONST_MODEL_DETAILS)

    @model_details.setter
    def model_details(self, model_details: ModelDetails):
        self.with_model_details(model_details)

    def with_model_details(self, model_details: ModelDetails) -> "Dataset":
        """Sets the model details for the dataset.

        Parameters
        ----------
        model_details: ModelDetails

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """
        if not isinstance(model_details, ModelDetails):
            raise TypeError(
                "The argument `model_details` has to be of type `ModelDetails`"
                "but is of type: `{}`".format(type(model_details))
            )

        return self.set_spec(self.CONST_MODEL_DETAILS, model_details.to_dict())

    @property
    def feature_groups(self) -> List["FeatureGroup"]:
        collection: "DatasetFeatureGroupCollection" = self.get_spec(
            self.CONST_FEATURE_GROUP
        )
        feature_groups: List["FeatureGroup"] = []
        if collection and collection.items:
            for datasetFGSummary in collection.items:
                feature_groups.append(
                    FeatureGroup.from_id(datasetFGSummary.feature_group_id)
                )

        return feature_groups

    @feature_groups.setter
    def feature_groups(self, feature_groups: List["FeatureGroup"]):
        self.with_feature_groups(feature_groups)

    def with_feature_groups(self, feature_groups: List["FeatureGroup"]) -> "Dataset":
        """Sets the model details for the dataset.

        Parameters
        ----------
        feature_groups: List of feature groups
        Returns
        -------
        Dataset
            The Dataset instance (self).

        """
        collection: List["DatasetFeatureGroupSummary"] = []
        for group in feature_groups:
            collection.append(DatasetFeatureGroupSummary(feature_group_id=group.id))

        self._is_manual_association = True
        return self.set_spec(
            self.CONST_FEATURE_GROUP,
            DatasetFeatureGroupCollection(items=collection, is_manual_association=True),
        )

    def feature_groups_to_df(self):
        return pd.DataFrame.from_records(
            [
                feature_group.oci_feature_group.to_df_record()
                for feature_group in self.feature_groups
            ]
        )

    @property
    def partition_keys(self) -> List[str]:
        return self.get_spec(self.CONST_PARTITION_KEYS)

    @partition_keys.setter
    def partition_keys(self, value: List[str]):
        self.with_partition_keys(value)

    def with_partition_keys(self, partition_keys: List[str]) -> "Dataset":
        """Sets the partition keys of the dataset.

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

    def add_models(self, model_details: ModelDetails) -> "Dataset":
        """Add model details to the dataset, Append to the existing model id list

        Parameters
        ----------
        model_details: ModelDetails to be appended to the existing model details

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """
        existing_model_details = self.model_details
        if existing_model_details and existing_model_details.items:
            items = existing_model_details["items"]
            for item in items:
                if item not in model_details.items:
                    model_details.items.append(item)
        self.with_model_details(model_details)
        try:
            return self.update()
        except Exception as ex:
            logger.error(
                f"Dataset update Failed with : {type(ex)} with error message: {ex}"
            )
            if existing_model_details:
                self.with_model_details(
                    ModelDetails().with_items(existing_model_details["items"])
                )
            else:
                self.with_model_details(ModelDetails().with_items([]))
                return self

    def remove_models(self, model_details: ModelDetails) -> "Dataset":
        """remove model details from the dataset, remove from the existing dataset model id list

        Parameters
        ----------
        model_details: ModelDetails to be removed from the existing model details

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """
        existing_model_details = self.model_details
        if existing_model_details.items:
            items = existing_model_details["items"]
            if model_details.items and all(
                item in items for item in model_details.items
            ):
                model_details_input = list(set(items) - set(model_details.items))
                self.with_model_details(ModelDetails().with_items(model_details_input))
                return self.update()
            else:
                raise ValueError(
                    f"Can't get find the model details in associated dataset model ids {self.model_details}"
                )

        else:
            raise ValueError(
                f"Can't get find the model details in associated dataset model ids {self.model_details}"
            )

    def show(self, rankdir: str = GraphOrientation.LEFT_RIGHT) -> None:
        """
        Show the lineage tree for the dataset instance.

        Raises:
            ValueError: If  lineage graph cannot be plotted due to missing lineage information.
        """
        lineage_type = {self.CONST_LINEAGE_CONSTRUCT_TYPE: EntityType.DATASET.value}
        lineage = self.lineage.from_id(self.id, **lineage_type)
        if lineage:
            GraphService.view_lineage(lineage.data, EntityType.DATASET, rankdir)
        else:
            raise ValueError(
                f"Can't get lineage information for Feature group id {self.id}"
            )

    def create(self, validate_sql=False, **kwargs) -> "Dataset":
        """Creates dataset  resource.

        !!! note "Lazy"
            This method is lazy and does not persist any metadata or feature data in the
            feature store on its own. To persist the dataset and save dataset data
            along the metadata in the feature store, call the `materialise()`.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Dataset` accepts.
        validate_sql:
            Boolean value indicating whether to validate sql before creating dataset

        Returns
        -------
        Dataset
            The Dataset instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        self.compartment_id = OCIModelMixin.check_compartment_id(self.compartment_id)

        if not self.name:
            self.name = self._random_display_name()

        if self.statistics_config is None:
            self.statistics_config = StatisticsConfig()

        if validate_sql is True:
            self.spark_engine.sql(self.get_spec(self.CONST_QUERY))

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a dataset resource with payload {payload}")

        # Create dataset
        logger.info("Saving dataset.")
        self.oci_dataset = self._to_oci_dataset(**kwargs).create()
        self._update_from_oci_dataset_model(self.oci_dataset)
        self.with_id(self.oci_dataset.id)
        return self

    def _build_dataset_job(self, ingestion_mode, feature_option_details=None):
        dataset_job = (
            DatasetJob()
            .with_dataset_id(self.id)
            .with_compartment_id(self.compartment_id)
            .with_ingestion_mode(ingestion_mode)
        )

        if feature_option_details:
            dataset_job = dataset_job.with_feature_option_details(
                feature_option_details
            )
        return dataset_job

    def delete(self):
        """Removes Dataset Resource.

        Returns
        -------
        None
        """
        # Create DataSet Job and persist it
        dataset_job = self._build_dataset_job(BatchIngestionMode.DEFAULT)

        # Create the Job
        dataset_job.create()
        dataset_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=ExecutionEngine.SPARK,
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        dataset_execution_strategy.delete_dataset(self, dataset_job)

    def get_features(self) -> List[DatasetFeature]:
        """
        Returns all the features in the dataset.

        Returns:
            List[DatasetFeature]
        """

        return self.features

    def get_features_df(self) -> "pandas.DataFrame":
        """
        Returns all the features as pandas dataframe.

        Returns:
            pandas.DataFrame
        """
        records = []
        for feature in self.features:
            records.append({"name": feature.feature_name, "type": feature.feature_type})
        return pandas.DataFrame.from_records(records)

    def update(self, **kwargs) -> "Dataset":
        """Updates Dataset in the feature store.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.Dataset` accepts.

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """

        if not self.id:
            raise ValueError(
                "Dataset needs to be saved to the feature store before it can be updated."
            )

        self.oci_dataset = self._to_oci_dataset(**kwargs).update()
        return self

    def _update_from_oci_dataset_model(self, oci_dataset: OCIDataset) -> "Dataset":
        """Update the properties from an OCIDataset object.

        Parameters
        ----------
        oci_dataset: OCIDataset
            An instance of OCIDataset.

        Returns
        -------
        Dataset
            The Dataset instance (self).
        """

        # Update the main properties
        self.oci_dataset = oci_dataset
        dataset_details = oci_dataset.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in dataset_details:
                if infra_attr == self.CONST_OUTPUT_FEATURE_DETAILS:
                    # May not need if we fix the backend and add dataset_id to the output_feature
                    features_list = []
                    for output_feature in dataset_details[infra_attr]["items"]:
                        output_feature["datasetId"] = dataset_details[self.CONST_ID]
                        features_list.append(output_feature)

                    value = {self.CONST_ITEMS: features_list}
                elif infra_attr == self.CONST_FEATURE_GROUP:
                    value = getattr(self.oci_dataset, dsc_attr)
                else:
                    value = dataset_details[infra_attr]
                self.set_spec(infra_attr, value)
        return self

    def materialise(
        self,
        ingestion_mode: BatchIngestionMode = BatchIngestionMode.OVERWRITE,
        feature_option_details: FeatureOptionDetails = None,
    ):
        """Creates a dataset job.

        Parameters
        ----------
        ingestion_mode: dict(str, str), optional
            The IngestionMode is used to specify the expected behavior of saving a DataFrame.
            Defaults to OVERWRITE.
        feature_option_details: FeatureOptionDetails
            An instance of the FeatureOptionDetails class containing feature options.

        Returns
        -------
        None

        """

        # Build the job and persist it.
        dataset_job = self._build_dataset_job(ingestion_mode, feature_option_details)
        dataset_job = dataset_job.create()
        # Update the dataset with corresponding job so that user can see the details about the job
        self.with_job_id(dataset_job.id)

        dataset_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=ExecutionEngine.SPARK,
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        dataset_execution_strategy.ingest_dataset(self, dataset_job)

    def get_last_job(self) -> "DatasetJob":
        """Gets the Job details for the last running Dataset job.

        Returns:
            DatasetJob
        """

        if not self.id:
            raise ValueError(
                "Dataset needs to be saved to the feature store before getting associated jobs."
            )

        if not self.job_id:
            ds_job = DatasetJob.list(
                dataset_id=self.id,
                compartment_id=self.compartment_id,
                sort_by="timeCreated",
                limit="1",
            )
            if not ds_job:
                raise ValueError(
                    "Unable to retrieve the associated last job. Please make sure you materialized the data."
                )
            self.with_job_id(ds_job[0].id)
            return ds_job[0]
        return DatasetJob.from_id(self.job_id)

    @deprecated(details="preview functionality is deprecated. Please use as_of.")
    def preview(
        self,
        row_count: int = 10,
        version_number: int = None,
        timestamp: datetime = None,
    ):
        """preview the dataset and return the response in dataframe.

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
        target_table = f"{self.entity_id}.{self.name}"

        if version_number or timestamp is not None:
            logger.warning("Time travel queries are not supported in current version")
        sql_query = f"select * from {target_table} LIMIT {row_count}"

        return self.spark_engine.sql(sql_query)

    def check_resource_materialization(self):
        """Checks whether the target Delta table for this resource has been materialized in Spark.
        If the target Delta table doesn't exist, raises a NotMaterializedError with the type and name of this resource.
        """
        if not self.spark_engine.is_delta_table_exists(self.target_delta_table()):
            raise NotMaterializedError(self.type, self.name)

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
        """Get the dataset profile information and return the response in dataframe.

        Returns
        -------
        spark dataframe
            The profile result in spark dataframe
        """
        self.check_resource_materialization()

        target_table = f"{self.entity_id}.{self.name}"
        sql_query = f"DESCRIBE DETAIL {target_table}"

        return self.spark_engine.sql(sql_query)

    def restore(self, version_number: int = None, timestamp: datetime = None):
        """restore the dataset and return the response in dataframe.

        Parameters
        ----------
        timestamp: datetime
            commit date time in format yyyy-MM-dd or yyyy-MM-dd HH:mm:ss to restore
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
        target_table = f"{self.entity_id}.{self.name}"
        if version_number is not None:
            sql_query = (
                f"RESTORE TABLE {target_table} TO VERSION AS OF {version_number}"
            )
        else:
            iso_timestamp = timestamp.isoformat(" ", "seconds").__str__()
            sql_query = (
                f"RESTORE TABLE {target_table} TO TIMESTAMP AS OF {iso_timestamp}"
            )

        restore_output = self.spark_engine.sql(sql_query)

        feature_group_execution_strategy = (
            OciExecutionStrategyProvider.provide_execution_strategy(
                execution_engine=ExecutionEngine.SPARK,
                metastore_id=get_metastore_id(self.feature_store_id),
            )
        )

        feature_group_execution_strategy.update_dataset_features(self, target_table)
        return restore_output

    def history(self):
        """get the dataset commit history.

        Returns
        -------
        spark dataframe
            The history output as spark dataframe
        """
        target_table = f"{self.entity_id}.{self.name}"
        sql_query = f"DESCRIBE HISTORY {target_table}"
        return self.spark_engine.sql(sql_query)

    def get_statistics(self, job_id: str = None) -> "Statistics":
        """Retrieve Statistics object for the job with job_id
        if job_id is not specified the last run job will be considered.
        Args:
            job_id (str): [job id of the job for which the statistics need to be retrieved]

        Returns:
            [type]: [Statistics]
        """
        if not self.id:
            raise ValueError(
                "Dataset needs to be saved to the feature store before retrieving the statistics"
            )

        stat_job_id = job_id if job_id is not None else self.get_last_job().id

        # TODO: take the one in memory or will list down job ids and find the latest
        dataset_job = DatasetJob.from_id(stat_job_id)
        if self.id != dataset_job.dataset_id:
            raise ValueError("The specified job id does not belong to this dataset")
        output_details = dataset_job.job_output_details
        feature_statistics = (
            output_details.get("featureStatistics") if output_details else None
        )
        stat_version = output_details.get("version") if output_details else None
        version = stat_version if stat_version is not None else 1

        return Statistics(feature_statistics, version)

    def get_validation_output(self, job_id: str = None) -> "ValidationOutput":
        """Retrieve Statistics object for the job with job_id
        if job_id is not specified the last run job will be considered.
        Args:
            job_id (str): [job id of the job for which the validation report need to be retrieved]

        Returns:
           ValidationOutput -- The validation output data in DataFrame format.
        """

        if not self.id:
            raise ValueError(
                "Dataset needs to be saved to the feature store before retrieving the validation report"
            )

        validation_job_id = job_id if job_id is not None else self.get_last_job().id

        # retrieve the validation output JSON from data_flow_batch_execution_output
        dataset_job = DatasetJob.from_id(validation_job_id)
        output_details = dataset_job.job_output_details
        validation_output = (
            output_details.get("validationOutput") if output_details else None
        )
        return ValidationOutput(validation_output)

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists dataset resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the dataset resources in a pandas dataframe format.
        """
        records = []
        for oci_dataset in OCIDataset.list_resource(compartment_id, **kwargs):
            records.append(
                {
                    "id": oci_dataset.id,
                    "name": oci_dataset.name,
                    "description": oci_dataset.description,
                    "time_created": oci_dataset.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_dataset.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_dataset.lifecycle_state,
                    "created_by": f"...{oci_dataset.created_by[-6:]}",
                    "compartment_id": f"...{oci_dataset.compartment_id[-6:]}",
                    "feature_store_id": oci_dataset.feature_store_id,
                    "entity_id": oci_dataset.entity_id,
                    "query": oci_dataset.query,
                    "dataset_ingestion_mode": oci_dataset.dataset_ingestion_mode,
                    "expectation_details": oci_dataset.expectation_details,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["Dataset"]:
        """Lists Dataset Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering Dataset.

        Returns
        -------
        List[Dataset]
            The list of the Dataset Resources.
        """
        return [
            cls()._update_from_oci_dataset_model(oci_dataset)
            for oci_dataset in OCIDataset.list_resource(compartment_id, **kwargs)
        ]

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes dataset  to a dictionary.

        Returns
        -------
        dict
            The dataset resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            if key == self.CONST_FEATURE_GROUP:
                spec[
                    key
                ] = self.oci_dataset.client.base_client.sanitize_for_serialization(
                    value
                )
            else:
                spec[key] = value
        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

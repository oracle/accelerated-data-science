# coding: utf-8
# Copyright (c) 2016, 2024, Oracle and/or its affiliates.  All rights reserved.
# This software is dual-licensed to you under the Universal Permissive License (UPL) 1.0 as shown at https://oss.oracle.com/licenses/upl or Apache License 2.0 as shown at http://www.apache.org/licenses/LICENSE-2.0. You may choose either license.

from __future__ import absolute_import

from .complete_dataset_job_details import CompleteDatasetJobDetails
from .complete_feature_group_job_details import CompleteFeatureGroupJobDetails
from .create_dataset_details import CreateDatasetDetails
from .create_dataset_feature_group_collection_details import CreateDatasetFeatureGroupCollectionDetails
from .create_dataset_job_details import CreateDatasetJobDetails
from .create_entity_details import CreateEntityDetails
from .create_feature_group_details import CreateFeatureGroupDetails
from .create_feature_group_job_details import CreateFeatureGroupJobDetails
from .create_feature_store_details import CreateFeatureStoreDetails
from .create_rule_detail import CreateRuleDetail
from .create_transformation_details import CreateTransformationDetails
from .dataset import Dataset
from .dataset_collection import DatasetCollection
from .dataset_feature_group_collection import DatasetFeatureGroupCollection
from .dataset_feature_group_summary import DatasetFeatureGroupSummary
from .dataset_job import DatasetJob
from .dataset_job_categorical_feature_statistics import DatasetJobCategoricalFeatureStatistics
from .dataset_job_collection import DatasetJobCollection
from .dataset_job_numerical_feature_statistics import DatasetJobNumericalFeatureStatistics
from .dataset_job_output_details import DatasetJobOutputDetails
from .dataset_job_statistics_collection import DatasetJobStatisticsCollection
from .dataset_job_statistics_details import DatasetJobStatisticsDetails
from .dataset_job_statistics_item import DatasetJobStatisticsItem
from .dataset_job_statistics_summary import DatasetJobStatisticsSummary
from .dataset_job_summary import DatasetJobSummary
from .dataset_job_validation_collection import DatasetJobValidationCollection
from .dataset_job_validation_output_details import DatasetJobValidationOutputDetails
from .dataset_job_validation_summary import DatasetJobValidationSummary
from .dataset_summary import DatasetSummary
from .entity import Entity
from .entity_collection import EntityCollection
from .entity_summary import EntitySummary
from .expectation_details import ExpectationDetails
from .feature_group import FeatureGroup
from .feature_group_collection import FeatureGroupCollection
from .feature_group_job import FeatureGroupJob
from .feature_group_job_categorical_feature_statistics import FeatureGroupJobCategoricalFeatureStatistics
from .feature_group_job_collection import FeatureGroupJobCollection
from .feature_group_job_numerical_feature_statistics import FeatureGroupJobNumericalFeatureStatistics
from .feature_group_job_output_details import FeatureGroupJobOutputDetails
from .feature_group_job_statistics_collection import FeatureGroupJobStatisticsCollection
from .feature_group_job_statistics_details import FeatureGroupJobStatisticsDetails
from .feature_group_job_statistics_item import FeatureGroupJobStatisticsItem
from .feature_group_job_statistics_summary import FeatureGroupJobStatisticsSummary
from .feature_group_job_summary import FeatureGroupJobSummary
from .feature_group_job_validation_collection import FeatureGroupJobValidationCollection
from .feature_group_job_validation_output_details import FeatureGroupJobValidationOutputDetails
from .feature_group_job_validation_summary import FeatureGroupJobValidationSummary
from .feature_group_summary import FeatureGroupSummary
from .feature_option_details import FeatureOptionDetails
from .feature_option_read_config_details import FeatureOptionReadConfigDetails
from .feature_option_write_config_details import FeatureOptionWriteConfigDetails
from .feature_store import FeatureStore
from .feature_store_collection import FeatureStoreCollection
from .feature_store_summary import FeatureStoreSummary
from .lineage import Lineage
from .lineage_detail import LineageDetail
from .lineage_summary import LineageSummary
from .lineage_summary_collection import LineageSummaryCollection
from .model_collection import ModelCollection
from .offline_config import OfflineConfig
from .online_config import OnlineConfig
from .output_feature_detail import OutputFeatureDetail
from .output_feature_detail_collection import OutputFeatureDetailCollection
from .partition_key_collection import PartitionKeyCollection
from .partition_key_summary import PartitionKeySummary
from .patch_feature_group_details import PatchFeatureGroupDetails
from .primary_key_collection import PrimaryKeyCollection
from .primary_key_summary import PrimaryKeySummary
from .raw_feature_detail import RawFeatureDetail
from .search_context import SearchContext
from .search_details import SearchDetails
from .search_summary import SearchSummary
from .search_summary_collection import SearchSummaryCollection
from .statistics_config import StatisticsConfig
from .transformation import Transformation
from .transformation_collection import TransformationCollection
from .transformation_summary import TransformationSummary
from .update_dataset_details import UpdateDatasetDetails
from .update_entity_details import UpdateEntityDetails
from .update_feature_group_details import UpdateFeatureGroupDetails
from .update_feature_store_details import UpdateFeatureStoreDetails
from .validation_output_details import ValidationOutputDetails

# Maps type names to classes for feature_store services.
feature_store_type_mapping = {
    "CompleteDatasetJobDetails": CompleteDatasetJobDetails,
    "CompleteFeatureGroupJobDetails": CompleteFeatureGroupJobDetails,
    "CreateDatasetDetails": CreateDatasetDetails,
    "CreateDatasetFeatureGroupCollectionDetails": CreateDatasetFeatureGroupCollectionDetails,
    "CreateDatasetJobDetails": CreateDatasetJobDetails,
    "CreateEntityDetails": CreateEntityDetails,
    "CreateFeatureGroupDetails": CreateFeatureGroupDetails,
    "CreateFeatureGroupJobDetails": CreateFeatureGroupJobDetails,
    "CreateFeatureStoreDetails": CreateFeatureStoreDetails,
    "CreateRuleDetail": CreateRuleDetail,
    "CreateTransformationDetails": CreateTransformationDetails,
    "Dataset": Dataset,
    "DatasetCollection": DatasetCollection,
    "DatasetFeatureGroupCollection": DatasetFeatureGroupCollection,
    "DatasetFeatureGroupSummary": DatasetFeatureGroupSummary,
    "DatasetJob": DatasetJob,
    "DatasetJobCategoricalFeatureStatistics": DatasetJobCategoricalFeatureStatistics,
    "DatasetJobCollection": DatasetJobCollection,
    "DatasetJobNumericalFeatureStatistics": DatasetJobNumericalFeatureStatistics,
    "DatasetJobOutputDetails": DatasetJobOutputDetails,
    "DatasetJobStatisticsCollection": DatasetJobStatisticsCollection,
    "DatasetJobStatisticsDetails": DatasetJobStatisticsDetails,
    "DatasetJobStatisticsItem": DatasetJobStatisticsItem,
    "DatasetJobStatisticsSummary": DatasetJobStatisticsSummary,
    "DatasetJobSummary": DatasetJobSummary,
    "DatasetJobValidationCollection": DatasetJobValidationCollection,
    "DatasetJobValidationOutputDetails": DatasetJobValidationOutputDetails,
    "DatasetJobValidationSummary": DatasetJobValidationSummary,
    "DatasetSummary": DatasetSummary,
    "Entity": Entity,
    "EntityCollection": EntityCollection,
    "EntitySummary": EntitySummary,
    "ExpectationDetails": ExpectationDetails,
    "FeatureGroup": FeatureGroup,
    "FeatureGroupCollection": FeatureGroupCollection,
    "FeatureGroupJob": FeatureGroupJob,
    "FeatureGroupJobCategoricalFeatureStatistics": FeatureGroupJobCategoricalFeatureStatistics,
    "FeatureGroupJobCollection": FeatureGroupJobCollection,
    "FeatureGroupJobNumericalFeatureStatistics": FeatureGroupJobNumericalFeatureStatistics,
    "FeatureGroupJobOutputDetails": FeatureGroupJobOutputDetails,
    "FeatureGroupJobStatisticsCollection": FeatureGroupJobStatisticsCollection,
    "FeatureGroupJobStatisticsDetails": FeatureGroupJobStatisticsDetails,
    "FeatureGroupJobStatisticsItem": FeatureGroupJobStatisticsItem,
    "FeatureGroupJobStatisticsSummary": FeatureGroupJobStatisticsSummary,
    "FeatureGroupJobSummary": FeatureGroupJobSummary,
    "FeatureGroupJobValidationCollection": FeatureGroupJobValidationCollection,
    "FeatureGroupJobValidationOutputDetails": FeatureGroupJobValidationOutputDetails,
    "FeatureGroupJobValidationSummary": FeatureGroupJobValidationSummary,
    "FeatureGroupSummary": FeatureGroupSummary,
    "FeatureOptionDetails": FeatureOptionDetails,
    "FeatureOptionReadConfigDetails": FeatureOptionReadConfigDetails,
    "FeatureOptionWriteConfigDetails": FeatureOptionWriteConfigDetails,
    "FeatureStore": FeatureStore,
    "FeatureStoreCollection": FeatureStoreCollection,
    "FeatureStoreSummary": FeatureStoreSummary,
    "Lineage": Lineage,
    "LineageDetail": LineageDetail,
    "LineageSummary": LineageSummary,
    "LineageSummaryCollection": LineageSummaryCollection,
    "ModelCollection": ModelCollection,
    "OfflineConfig": OfflineConfig,
    "OnlineConfig": OnlineConfig,
    "OutputFeatureDetail": OutputFeatureDetail,
    "OutputFeatureDetailCollection": OutputFeatureDetailCollection,
    "PartitionKeyCollection": PartitionKeyCollection,
    "PartitionKeySummary": PartitionKeySummary,
    "PatchFeatureGroupDetails": PatchFeatureGroupDetails,
    "PrimaryKeyCollection": PrimaryKeyCollection,
    "PrimaryKeySummary": PrimaryKeySummary,
    "RawFeatureDetail": RawFeatureDetail,
    "SearchContext": SearchContext,
    "SearchDetails": SearchDetails,
    "SearchSummary": SearchSummary,
    "SearchSummaryCollection": SearchSummaryCollection,
    "StatisticsConfig": StatisticsConfig,
    "Transformation": Transformation,
    "TransformationCollection": TransformationCollection,
    "TransformationSummary": TransformationSummary,
    "UpdateDatasetDetails": UpdateDatasetDetails,
    "UpdateEntityDetails": UpdateEntityDetails,
    "UpdateFeatureGroupDetails": UpdateFeatureGroupDetails,
    "UpdateFeatureStoreDetails": UpdateFeatureStoreDetails,
    "ValidationOutputDetails": ValidationOutputDetails
}

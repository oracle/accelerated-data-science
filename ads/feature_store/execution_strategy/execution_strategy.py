#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from __future__ import annotations

from abc import ABC, abstractmethod

from ads.common.decorator.runtime_dependency import OptionalDependency

try:
    from pyspark.sql import DataFrame
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        f"The `pyspark` module was not found. Please run `pip install "
        f"{OptionalDependency.SPARK}`."
    )
except Exception as e:
    raise
from ads.feature_store.dataset_job import DatasetJob
from ads.feature_store.feature_group_job import FeatureGroupJob


class Strategy(ABC):
    """
    An abstract base class that defines a strategy for ingesting and deleting feature definitions and datasets.
    """

    @abstractmethod
    def ingest_feature_definition(
        self, feature_group, feature_group_job: FeatureGroupJob, dataframe
    ):
        """
        Ingests a feature definition into the system.

        Args:
            feature_group (object): An object representing a group of related features.
            feature_group_job (FeatureGroupJob): An object representing a job responsible for processing the feature group.
            dataframe (pandas.DataFrame): A Pandas DataFrame object containing the feature data.
        """
        pass

    @abstractmethod
    def ingest_feature_definition_stream(
        self,
        feature_group,
        feature_group_job: FeatureGroupJob,
        dataframe,
        query_name,
        await_termination,
        timeout,
        checkpoint_dir,
    ):
        pass

    @abstractmethod
    def ingest_dataset(self, dataset, dataset_job: DatasetJob):
        """
        Ingests a dataset into the system.

        Args:
            dataset (Dataset): An object representing a dataset.
            dataset_job (DatasetJob): An object representing a job responsible for processing the dataset.
        """
        pass

    @abstractmethod
    def update_feature_definition_features(self, feature_group, target_table):
        """
        Updates the output features of a feature definition.

        Args:
            feature_group (object): An object representing a group of related features.
            target_table: The target table of feature definition.
        """
        pass

    @abstractmethod
    def update_dataset_features(self, dataset, target_table):
        """
        Updates the output features of a feature definition.

        Args:
            dataset (Dataset): An object representing a dataset.
            target_table: The target table of dataset.
        """
        pass

    @abstractmethod
    def delete_feature_definition(
        self, feature_group, feature_group_job: FeatureGroupJob
    ):
        """
        Deletes a feature definition from the system.

        Args:
            feature_group (object): An object representing a group of related features.
            feature_group_job (FeatureGroupJob): An object representing a job responsible for processing the feature group.
        """
        pass

    @abstractmethod
    def delete_dataset(self, dataset, dataset_job: DatasetJob):
        """
        Deletes a dataset from the system.

        Args:
            dataset (Dataset): An object representing a dataset.
            dataset_job (DatasetJob): An object representing a job responsible for processing the dataset.
        """
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from datetime import datetime
from random import random

import oci
import pandas as pd
from ads.feature_store.entity import Entity
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import ads
import os
from ads.feature_store.common.enums import FeatureType, TransformationMode
from ads.feature_store.dataset import Dataset
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.input_feature_detail import FeatureDetail
from ads.feature_store.statistics_config import StatisticsConfig


client_kwargs = dict(
    retry_strategy=oci.retry.NoneRetryStrategy(),
    fs_service_endpoint=os.getenv("service_endpoint"),
)
ads.set_auth(client_kwargs=client_kwargs)

try:
    from ads.feature_store.feature_store import FeatureStore
except (ImportError, AttributeError) as e:
    raise unittest.SkipTest("FeatureStore not available.")

MAXIMUM_TIMEOUT = 43
SLEEP_INTERVAL = 60


def transformation_with_kwargs(data_frame, **kwargs):
    is_area_enabled = kwargs.get("is_area_enabled")

    if is_area_enabled:
        # Calculate petal area and sepal area
        data_frame["petal_area"] = (
            data_frame["petal_length"] * data_frame["petal_width"]
        )
        data_frame["sepal_area"] = (
            data_frame["sepal_length"] * data_frame["sepal_width"]
        )

    # Return the updated DataFrame
    return data_frame


class FeatureStoreTestCase:
    # networks compartment in feature store
    TIME_NOW = str.format(
        "{}_{}", datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S"), int(random() * 1000)
    )
    TENANCY_ID = os.getenv("TENANCY_ID")
    COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
    METASTORE_ID = os.getenv("METASTORE_ID")
    INPUT_FEATURE_DETAILS = [
        FeatureDetail("flower")
        .with_feature_type(FeatureType.STRING)
        .with_order_number(1),
        FeatureDetail("sepal_length")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(1),
        FeatureDetail("sepal_width")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(1),
        FeatureDetail("petal_width")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(1),
        FeatureDetail("class")
        .with_feature_type(FeatureType.STRING)
        .with_order_number(1),
        FeatureDetail("petal_length")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(1),
    ]
    data = pd.DataFrame(
        {
            "flower": [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "f11",
                "f12",
                "f13",
                "f14",
            ],
            "sepal_length": [
                5.0,
                5.0,
                4.4,
                5.5,
                5.5,
                5.1,
                6.9,
                6.5,
                5.2,
                6.1,
                5.4,
                6.3,
                7.3,
                6.7,
            ],
            "sepal_width": [
                3.6,
                3.4,
                2.9,
                4.2,
                3.5,
                3.8,
                3.1,
                2.8,
                2.7,
                2.8,
                3,
                2.9,
                2.9,
                2.5,
            ],
            "petal_width": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                1.5,
                1.5,
                1.4,
                1.2,
                1.5,
                1.8,
                1.8,
                1.8,
            ],
            "class": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
            ],
            "petal_length": [
                1.4,
                1.5,
                1.4,
                1.4,
                1.3,
                1.6,
                4.9,
                4.6,
                3.9,
                4.7,
                4.5,
                5.6,
                6.3,
                5.8,
            ],
        }
    )

    data2 = pd.DataFrame(
        {
            "flower": [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "f11",
                "f12",
                "f13",
                "f14",
            ],
            "sepal_length": [
                5.4,
                5.0,
                4.4,
                5.5,
                5.5,
                5.1,
                6.9,
                6.5,
                5.2,
                6.1,
                5.4,
                6.3,
                7.3,
                6.7,
            ],
            "sepal_width": [
                3.6,
                3.4,
                2.9,
                4.2,
                3.5,
                3.8,
                3.1,
                2.8,
                2.7,
                2.8,
                3,
                2.9,
                2.9,
                2.5,
            ],
            "petal_width": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                1.5,
                1.5,
                1.4,
                1.2,
                1.5,
                1.8,
                1.8,
                1.8,
            ],
            "class": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
            ],
            "petal_length": [
                1.4,
                1.5,
                1.5,
                1.4,
                1.3,
                1.6,
                4.9,
                4.6,
                3.9,
                4.7,
                4.5,
                5.6,
                6.3,
                5.8,
            ],
        }
    )

    data3 = pd.DataFrame(
        {
            "flower": [
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "f11",
                "f12",
                "f13",
                "f14",
            ],
            "sepal_length": [
                5.1,
                5.0,
                4.4,
                5.5,
                5.5,
                5.1,
                6.9,
                6.5,
                5.2,
                6.1,
                5.4,
                6.3,
                7.3,
                6.7,
            ],
            "sepal_width": [
                3.6,
                3.4,
                2.9,
                4.2,
                3.5,
                3.8,
                3.1,
                2.8,
                2.7,
                2.8,
                3,
                2.9,
                2.9,
                2.5,
            ],
            "petal_width": [
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                0.2,
                1.5,
                1.5,
                1.4,
                1.2,
                1.5,
                1.8,
                1.8,
                1.8,
            ],
            "class": [
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "setosa",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "versicolor",
                "virginica",
                "virginica",
                "virginica",
            ],
        }
    )

    def define_feature_store_resource(self) -> "FeatureStore":
        name = self.get_name("FeatureStore1")
        feature_store_resource = (
            FeatureStore()
            .with_description("Feature Store Description")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(name)
            .with_offline_config(metastore_id=self.METASTORE_ID)
        )
        return feature_store_resource

    def create_entity_resource(self, feature_store) -> "Entity":
        entity = feature_store.create_entity(display_name=self.get_name("entity"))
        return entity

    def create_transformation_resource(self, feature_store) -> "Transformation":
        transformation = feature_store.create_transformation(
            source_code_func=transformation_with_kwargs,
            display_name="transformation_with_kwargs",
            transformation_mode=TransformationMode.PANDAS,
        )
        return transformation

    def define_feature_group_resource(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group description")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals1"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys(["flower"])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(
                StatisticsConfig(True, columns=["sepal_length", "petal_width"])
            )
        )
        return feature_group_resource

    def define_dataset_resource(
        self, entity_id, feature_store_id, feature_group_name
    ) -> "Dataset":
        name = self.get_name("petals_ds")
        dataset_resource = (
            Dataset()
            .with_description("dataset description")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(name)
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_query(f"SELECT * FROM `{entity_id}`.{feature_group_name}")
            .with_statistics_config(
                StatisticsConfig(True, columns=["sepal_length", "petal_width"])
            )
        )
        return dataset_resource

    def define_expectation_suite_single(self) -> ExpectationSuite:
        expectation_suite_test1 = ExpectationSuite(
            expectation_suite_name="test_feature_group"
        )
        expectation_suite_test1.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "class"},
            )
        )
        return expectation_suite_test1

    def define_expectation_suite_multiple(self) -> ExpectationSuite:
        expectation_suite_test2 = ExpectationSuite(
            expectation_suite_name="test_feature_group"
        )
        expectation_suite_test2.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "class"},
            )
        )
        expectation_suite_test2.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": "petal_width", "min_value": 0, "max_value": 2},
            )
        )
        return expectation_suite_test2

    @staticmethod
    def clean_up_feature_store(feature_store):
        try:
            feature_store.delete()
        except Exception as ex:
            print("Failed to delete feature store: ", str(ex))
            exit(1)

    @staticmethod
    def clean_up_entity(entity):
        try:
            entity.delete()
        except Exception as ex:
            print("Failed to delete entity: ", str(ex))
            exit(1)

    @staticmethod
    def clean_up_feature_group(feature_group):
        try:
            feature_group.delete()
        except Exception as ex:
            print("Failed to delete feature group: ", str(ex))
            exit(1)

    @staticmethod
    def clean_up_transformation(transformation):
        try:
            transformation.delete()
        except Exception as ex:
            print("Failed to delete transformation: ", str(ex))
            exit(1)

    @staticmethod
    def clean_up_dataset(dataset):
        try:
            dataset.delete()
        except Exception as ex:
            print("Failed to delete dataset: ", str(ex))
            exit(1)

    @staticmethod
    def assert_dataset_job(dataset, dataset_job):
        assert dataset_job.display_name == dataset.name
        assert dataset_job.id is not None
        assert dataset_job.compartment_id == dataset.compartment_id

    def get_name(self, name):
        return "_".join((name, self.TIME_NOW))

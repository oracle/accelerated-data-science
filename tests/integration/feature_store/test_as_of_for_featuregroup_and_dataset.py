import ast
import json
import random
import string

import pandas as pd
import pytest

from ads.feature_store.common.exceptions import NotMaterializedError
from ads.feature_store.dataset import Dataset
from ads.feature_store.statistics_config import StatisticsConfig
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.feature_group import FeatureGroup


class TestAsOfForFeatureGroupAndDataset(FeatureStoreTestCase):
    """Contains integration tests for as_of support for feature groups and datasets"""

    # Generate random data
    test_as_of_data = {
        'Name': [random.choice(['Alice', 'Bob', 'Charlie', 'David']) for _ in range(10)],
        'Age': [random.randint(20, 40) for _ in range(10)],
        'Score': [round(random.uniform(0, 100), 2) for _ in range(10)],
        'City': [random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston']) for _ in range(10)],
        'Gender': [random.choice(['Male', 'Female']) for _ in range(10)],
        'ID': [''.join(random.choices(string.ascii_uppercase, k=5)) for _ in range(10)]
    }

    as_of_data_frame = pd.DataFrame(test_as_of_data)

    def define_feature_group_resource(
            self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group with statistics disabled")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals2"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys(['ID'])
            .with_schema_details_from_dataframe(self.as_of_data_frame)
            .with_statistics_config(False)
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

    def test_as_of_for_non_materialized_feature_group(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()
        assert fg.oci_feature_group.id

        with pytest.raises(NotMaterializedError):
            fg.as_of()

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)
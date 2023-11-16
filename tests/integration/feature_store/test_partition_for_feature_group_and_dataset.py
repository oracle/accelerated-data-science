import ast
import json

from ads.feature_store.dataset import Dataset
from ads.feature_store.statistics_config import StatisticsConfig
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.feature_group import FeatureGroup


class TestPartitioningForFeatureGroupAndDataset(FeatureStoreTestCase):
    """Contains integration tests for partitioning of feature groups and datasets"""

    def define_feature_group_resource_with_partitioning(
        self, entity_id, feature_store_id, partitioning_keys
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group with statistics disabled")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals2"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_partition_keys(partitioning_keys)
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def define_dataset_resource_with_partitioning(
        self, entity_id, feature_store_id, feature_group_name, partitioning_keys
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
            .with_partition_keys(partitioning_keys)
        )
        return dataset_resource

    def test_feature_group_materialization_with_partitioning_keys(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource_with_partitioning(
            entity.oci_fs_entity.id, fs.oci_fs.id, ["class"]
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        history_df = fg.history()
        history_df_dict = json.loads(history_df.toJSON().collect()[0])
        materialized_partition_keys = ast.literal_eval(
            history_df_dict.get("operationParameters").get("partitionBy")
        )

        assert len(materialized_partition_keys) == 1
        assert materialized_partition_keys[0] == "class"

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_materialization_without_partitioning_keys(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource_with_partitioning(
            entity.oci_fs_entity.id, fs.oci_fs.id, None
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        history_df = fg.history()
        history_df_dict = json.loads(history_df.toJSON().collect()[0])
        materialized_partition_keys = ast.literal_eval(
            history_df_dict.get("operationParameters").get("partitionBy")
        )

        assert len(materialized_partition_keys) == 0

        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_materialization_with_partitioning_keys(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)

        dataset = self.define_dataset_resource_with_partitioning(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name, ["class"]
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        history_df = dataset.history()
        history_df_dict = json.loads(history_df.toJSON().collect()[0])
        materialized_partition_keys = ast.literal_eval(
            history_df_dict.get("operationParameters").get("partitionBy")
        )

        assert len(materialized_partition_keys) == 1
        assert materialized_partition_keys[0] == "class"

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_dataset_materialization_without_partitioning_keys(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        fg = self.define_feature_group_resource(
            entity.oci_fs_entity.id, fs.oci_fs.id
        ).create()

        assert fg.oci_feature_group.id
        fg.materialise(self.data)

        dataset = self.define_dataset_resource_with_partitioning(
            entity.oci_fs_entity.id, fs.oci_fs.id, fg.oci_feature_group.name, None
        ).create()
        assert dataset.oci_dataset.id

        dataset.materialise()
        history_df = dataset.history()
        history_df_dict = json.loads(history_df.toJSON().collect()[0])
        materialized_partition_keys = ast.literal_eval(
            history_df_dict.get("operationParameters").get("partitionBy")
        )

        assert len(materialized_partition_keys) == 0

        self.clean_up_dataset(dataset)
        self.clean_up_feature_group(fg)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

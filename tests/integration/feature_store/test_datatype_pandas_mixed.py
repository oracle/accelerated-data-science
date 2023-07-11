from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
import pandas as pd
import numpy as np
import pytest


class TestDataTypePandasMixed(FeatureStoreTestCase):
    data_mixed = {"MixedColumn": ["John", 25, "Emma", 30, "Michael", 35]}
    data_mixed_nan = {
        "MixedColumn": [
            "John",
            float("nan"),
            "Emma",
            float("nan"),
            "Michael",
            float("nan"),
        ]
    }
    pandas_mixed_df = pd.DataFrame(data_mixed)
    pandas_mixed_df_nan = pd.DataFrame(data_mixed_nan)

    input_feature_details_mixed = [
        FeatureDetail("MixedColumn")
        .with_feature_type(FeatureType.STRING)
        .with_order_number(1)
    ]

    def define_feature_group_resource_with_pandas_mixed_infer_schema(
        self, entity_id, feature_store_id
    ):
        feature_group_pandas_mixed = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_mixed"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.pandas_mixed_df)
            .with_statistics_config(False)
        )

        return feature_group_pandas_mixed

    def define_feature_group_resource_with_pandas_mixed_infer_schema_nan(
        self, entity_id, feature_store_id
    ):
        feature_group_pandas_mixed_1 = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_mixed_1"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.pandas_mixed_df_nan)
            .with_statistics_config(False)
        )

        return feature_group_pandas_mixed_1

    def define_feature_group_resource_with_pandas_mixed_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_mixed_schema = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_mixed_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_mixed)
            .with_statistics_config(False)
        )
        return feature_group_pandas_mixed_schema

    def test_feature_group_pandas_mixed_infer_schema(self):
        """Tests  pandas mixed data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id
        try:
            feature_group = (
                self.define_feature_group_resource_with_pandas_mixed_infer_schema(
                    entity.oci_fs_entity.id, fs.oci_fs.id
                )
            )
        except TypeError as e:
            assert (
                e.__str__()
                == "field MixedColumn: Can not merge type <class 'pyspark.sql.types.StringType'> "
                "and <class 'pyspark.sql.types.LongType'>"
            )
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_mixed_infer_schema_nan(self):
        """Tests  pandas mixed data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id
        feature_group = (
            self.define_feature_group_resource_with_pandas_mixed_infer_schema_nan(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()
        feature_group.materialise(self.pandas_mixed_df_nan)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_mixed_with_schema(self):
        """Tests  pandas mixed data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_mixed_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.pandas_mixed_df)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

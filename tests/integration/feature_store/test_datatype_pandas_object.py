from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
import pandas as pd
import numpy as np
import pytest
from decimal import Decimal


class TestDataTypePandasObject(FeatureStoreTestCase):
    a = ["value1", "value2"]
    b = ["value3", "value4"]
    df_string = pd.DataFrame({"A": a, "B": b})

    data1 = Decimal("1.23")
    data2 = Decimal("3.45")
    df_decimal = pd.DataFrame({"A": pd.Series(data1), "B": pd.Series(data2)})

    datetime_data1 = "2021-11-15 21:04:15"
    datetime_data2 = "2021-12-16 21:04:15"
    df_date = pd.DataFrame(
        {
            "A": pd.Series(pd.to_datetime(datetime_data1).date()),
            "B": pd.Series(pd.to_datetime(datetime_data2).date()),
        }
    )

    input_feature_details_string = [
        FeatureDetail("A").with_feature_type(FeatureType.STRING).with_order_number(1),
        FeatureDetail("B").with_feature_type(FeatureType.STRING).with_order_number(2),
    ]

    input_feature_details_decimal = [
        FeatureDetail("A").with_feature_type(FeatureType.DECIMAL).with_order_number(1),
        FeatureDetail("B").with_feature_type(FeatureType.DECIMAL).with_order_number(2),
    ]

    input_feature_details_date = [
        FeatureDetail("A").with_feature_type(FeatureType.DATE).with_order_number(1),
        FeatureDetail("B").with_feature_type(FeatureType.DATE).with_order_number(2),
    ]

    def define_feature_group_resource_with_pandas_object_string_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_string = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_object_string"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_string)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_string

    def define_feature_group_resource_with_pandas_object_string_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_string_schema = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_string_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_string)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_string_schema

    def define_feature_group_resource_with_pandas_object_decimal_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_string = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_object_decimal"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_decimal)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_string

    def define_feature_group_resource_with_pandas_object_decimal_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_decimal_schema = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_object_decimal_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_decimal)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_decimal_schema

    def define_feature_group_resource_with_pandas_object_date_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_date = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_object_date"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_date)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_date

    def define_feature_group_resource_with_pandas_object_date_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_object_date_schema = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_object_date_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_date)
            .with_statistics_config(False)
        )
        return feature_group_pandas_object_date_schema

    def test_feature_group_pandas_object_string_infer_schema(self):
        """Tests  pandas object string data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_string_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()

        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_string)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_object_string_with_schema(self):
        """Tests  pandas object string data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_string_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_string)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_object_decimal_infer_schema(self):
        """Tests  pandas object decimal data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_decimal_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()

        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_decimal)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_object_decimal_with_schema(self):
        """Tests  pandas object decimal data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_decimal_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_decimal)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_object_date_infer_schema(self):
        """Tests  pandas object date data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_date_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()

        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_date)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_object_date_with_schema(self):
        """Tests  pandas object date data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_object_date_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_date)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

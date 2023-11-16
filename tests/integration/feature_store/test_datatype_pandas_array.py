import ads.feature_store.common.exceptions
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
import pandas as pd
import numpy as np
import pytest


class TestDataTypePandasArray(FeatureStoreTestCase):
    input_feature_details_pandas_array = [
        FeatureDetail("Strings")
        .with_feature_type(FeatureType.STRING_ARRAY)
        .with_order_number(1),
        FeatureDetail("Int32")
        .with_feature_type(FeatureType.INTEGER_ARRAY)
        .with_order_number(2),
        FeatureDetail("Int64")
        .with_feature_type(FeatureType.LONG_ARRAY)
        .with_order_number(3),
        FeatureDetail("Float32")
        .with_feature_type(FeatureType.FLOAT_ARRAY)
        .with_order_number(4),
        FeatureDetail("Float64")
        .with_feature_type(FeatureType.DOUBLE_ARRAY)
        .with_order_number(5),
        FeatureDetail("Timestamps")
        .with_feature_type(FeatureType.TIMESTAMP_ARRAY)
        .with_order_number(6),
        FeatureDetail("Boolean")
        .with_feature_type(FeatureType.BOOLEAN_ARRAY)
        .with_order_number(7),
        FeatureDetail("Dates")
        .with_feature_type(FeatureType.DATE_ARRAY)
        .with_order_number(8),
    ]

    input_feature_details_numpy_array = [
        FeatureDetail("A")
        .with_feature_type(FeatureType.FLOAT_ARRAY)
        .with_order_number(2),
    ]

    strings = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
    int32 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    int64 = [[92233720368547758072030], [40, 50, 60], [70, 80, 90]]
    float32 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    float64 = [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]
    timestamps = [
        [
            pd.Timestamp("2023-06-01").timestamp(),
            pd.Timestamp("2023-06-02").timestamp(),
        ],
        [
            pd.Timestamp("2023-06-01").timestamp(),
            pd.Timestamp("2023-06-02").timestamp(),
        ],
        [
            pd.Timestamp("2023-06-03").timestamp(),
            pd.Timestamp("2023-06-03").timestamp(),
        ],
    ]
    boolean = [[True, False, True], [False, True, False], [True, False, True]]
    dates = [
        [pd.Timestamp("2023-06-01").date(), pd.Timestamp("2023-06-02").date()],
        [pd.Timestamp("2023-06-01").date(), pd.Timestamp("2023-06-03").date()],
        [pd.Timestamp("2023-06-01").date(), pd.Timestamp("2023-06-03").date()],
    ]

    # Create DataFrame
    df_array = pd.DataFrame(
        {
            "Strings": strings,
            "Int32": int32,
            "Int64": int64,
            "Float32": float32,
            "Float64": float64,
            "Timestamps": timestamps,
            "Boolean": boolean,
            "Dates": dates,
        }
    )

    one_d_array = np.array([1.1, 2.2, 3.3], dtype="float32")
    two_d_array = one_d_array * one_d_array[:, np.newaxis]
    numpy_array_df = pd.DataFrame({"A": [two_d_array]})

    def define_feature_group_resource_with_pandas_array_infer_schema(
        self, entity_id, feature_store_id
    ):
        feature_group_pandas_array = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_array"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_array)
        )
        return feature_group_pandas_array

    def define_feature_group_resource_with_pandas_array_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_pandas_array = (
            FeatureGroup()
            .with_description("feature group resource for pandas array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_pandas_array"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_pandas_array)
            .with_statistics_config(False)
        )
        return feature_group_pandas_array

    def define_feature_group_resource_with_numpy_array_infer_schema(
        self, entity_id, feature_store_id
    ):
        feature_group_numpy_array = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_numpy_array"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.numpy_array_df)
            .with_statistics_config(False)
        )
        return feature_group_numpy_array

    def define_feature_group_resource_with_numpy_array_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_numpy_array_schema = (
            FeatureGroup()
            .with_description("feature group resource for numpy array types")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("feature_group_numpy_array_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_numpy_array)
            .with_statistics_config(False)
        )
        return feature_group_numpy_array_schema

    def test_feature_group_pandas_array_infer_schema(self):
        """Tests  pandas array  types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_array_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()
        feature_group.materialise(self.df_array)

        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_array_with_schema(self):
        """Tests pandas array with schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_array_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )
        feature_group.create()
        feature_group.materialise(self.df_array)

        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_numpy_array_infer_schema(self):
        """Tests numpy array infer schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        try:
            self.define_feature_group_resource_with_numpy_array_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        except TypeError as e:
            assert e.__str__() == "Unable to infer the type of the field A."

        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_numpy_array_with_schema(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_numpy_array_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
        feature_group.create()
        feature_group.materialise(self.numpy_array_df)
        try:
            df = feature_group.select().read()
        except ads.feature_store.common.exceptions.NotMaterializedError as e:
            assert (
                e.__str__()
                == "featureGroup " + feature_group.name + " is not materialized."
            )

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

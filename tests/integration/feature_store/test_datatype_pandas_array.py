
from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
import pandas as pd
import numpy as np
import pytest


class TestDataTypePandasArray(FeatureStoreTestCase):
    input_feature_details_string_array = [
        FeatureDetail("A").with_feature_type(FeatureType.STRING).with_order_number(1),
        FeatureDetail("B").with_feature_type(FeatureType.STRING_ARRAY).with_order_number(2),
    ]

    input_feature_details_numpy_array = [
        FeatureDetail("A").with_feature_type(FeatureType.FLOAT_ARRAY).with_order_number(2),
    ]

    a = ["value1", "value2"]
    b = [['j@a.com', 'x@y.com'], ['x@a.com', 'x@y.com', 'j@k.com']]
    df_array = pd.DataFrame(
        {
            "A": a,
            "B": b
        }
    )

    one_d_array = np.array([1.1, 2.2, 3.3], dtype="float32")
    two_d_array = one_d_array * one_d_array[:, np.newaxis]
    numpy_array_df = pd.DataFrame(
        {
            "A": [two_d_array]
        })

    def define_feature_group_resource_with_pandas_array_infer_schema(
        self, entity_id, feature_store_id
    ):

        with pytest.raises(TypeError, match="object of type <class 'list'> not supported"):
            feature_group_pandas_array= (
                FeatureGroup()
                .with_description("feature group resource for pandas datatypes")
                .with_compartment_id(self.COMPARTMENT_ID)
                .with_name(self.get_name("feature_group_pandas_array"))
                .with_entity_id(entity_id)
                .with_feature_store_id(feature_store_id)
                .with_primary_keys([])
                .with_schema_details_from_dataframe(self.df_array)
                .with_statistics_config(False)
            )

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
            .with_input_feature_details(self.input_feature_details_string_array)
            .with_statistics_config(False)
        )
        return feature_group_pandas_array

    def define_feature_group_resource_with_numpy_array_infer_schema(
        self, entity_id, feature_store_id
    ):

        with pytest.raises(TypeError, match="object of type <class 'numpy.ndarray'> not supported"):
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
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        self.define_feature_group_resource_with_pandas_array_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_array_with_schema(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_pandas_array_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
        feature_group.create()
        feature_group.materialise(self.df_array)

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_numpy_array_infer_schema(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        self.define_feature_group_resource_with_numpy_array_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
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

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

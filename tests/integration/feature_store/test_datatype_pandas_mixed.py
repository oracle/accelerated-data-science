from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
from ads.feature_store.feature_group import FeatureGroup
import pandas as pd
import numpy as np
import pytest



class TestDataTypePandasMixed(FeatureStoreTestCase):
    flights_df = pd.read_csv("https://objectstorage.us-ashburn-1.oraclecloud.com/p/hh2NOgFJbVSg4amcLM3G3hkTuHyBD-8aE_iCsuZKEvIav1Wlld-3zfCawG4ycQGN/n/ociodscdev/b/oci-feature-store/o/beta/data/flights/flights.csv")
    flights_df_mixed = pd.DataFrame(flights_df["CANCELLATION_REASON"])

    input_feature_details_mixed = [
            FeatureDetail("CANCELLATION_REASON").with_feature_type(FeatureType.STRING).with_order_number(1)]
    def define_feature_group_resource_with_pandas_mixed_infer_schema(
        self, entity_id, feature_store_id
    ):

        with pytest.raises(TypeError, match="Input feature 'CANCELLATION_REASON' has mixed types, FeatureType.STRING and FeatureType.FLOAT. That is not allowed. "):
            feature_group_pandas_mixed = (
                FeatureGroup()
                .with_description("feature group resource for pandas datatypes")
                .with_compartment_id(self.COMPARTMENT_ID)
                .with_name(self.get_name("feature_group_pandas_mixed"))
                .with_entity_id(entity_id)
                .with_feature_store_id(feature_store_id)
                .with_primary_keys([])
                .with_schema_details_from_dataframe(self.flights_df_mixed)
                .with_statistics_config(False)
            )

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
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        self.define_feature_group_resource_with_pandas_mixed_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_mixed_with_schema(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_pandas_mixed_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.flights_df_mixed)
        df = feature_group.select().read()
        assert df

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)




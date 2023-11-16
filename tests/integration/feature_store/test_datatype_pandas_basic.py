from tests.integration.feature_store.test_base import FeatureStoreTestCase
from ads.feature_store.feature_group import FeatureGroup
from ads.feature_store.input_feature_detail import FeatureDetail, FeatureType
import pandas as pd
import numpy as np

"""   Test the following pandas datatypes
        int8
        int16
        int32
        int64
        uint8
        uint16
        uint32
        float16
        float32
        float64
        datetime64[ns, UTC]
        datetime64[ns]
        timedelta64[ns]
        bool
"""


class TestDataTypePandasBasic(FeatureStoreTestCase):
    df_pandas = pd.DataFrame(
        {
            "dt_init8": pd.Series(np.array(np.random.randn(8), dtype="int8")),
            "dt_init16": pd.Series(np.array(np.random.randn(8), dtype="int16")),
            "dt_init32": pd.Series(np.array(np.random.randn(8), dtype="int32")),
            "dt_init64": pd.Series(np.array(np.random.randn(8), dtype="int64")),
            "dt_uinit8": pd.Series(np.array(np.random.randn(8), dtype="uint8")),
            "dt_uinit16": pd.Series(np.array(np.random.randn(8), dtype="uint16")),
            "dt_uinit32": pd.Series(np.array(np.random.randn(8), dtype="uint32")),
            "dt_float16": pd.Series(np.array(np.random.randn(8), dtype="float16")),
            "dt_float32": pd.Series(np.array(np.random.randn(8), dtype="float32")),
            "dt_float64": pd.Series(np.array(np.random.randn(8), dtype="float64")),
            "dt_bool": pd.Series(np.array(np.zeros(8), dtype="bool")),
        }
    )

    df_pandas_datetime = pd.DataFrame(
        {
            "datetime64_ns_utc": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-13",
                    "2020-02-01",
                    "2020-02-23",
                    "2020-03-05",
                    "2020-03-05",
                    "2020-03-05",
                    "2020-03-05",
                ],
                utc=True,
            ),
            "datetime64_ns": np.array(
                [
                    "2020-01-02",
                    "2020-01-13",
                    "2020-02-01",
                    "2020-02-23",
                    "2020-03-05",
                    "2020-03-05",
                    "2020-03-05",
                    "2020-03-05",
                ],
                dtype="datetime64[ns]",
            ),
            "timedelta64_ns": pd.to_timedelta(np.arange(8), unit="s"),
        }
    )

    input_feature_details = [
        FeatureDetail("dt_init8")
        .with_feature_type(FeatureType.INTEGER)
        .with_order_number(1),
        FeatureDetail("dt_init16")
        .with_feature_type(FeatureType.INTEGER)
        .with_order_number(2),
        FeatureDetail("dt_init32")
        .with_feature_type(FeatureType.LONG)
        .with_order_number(3),
        FeatureDetail("dt_init64")
        .with_feature_type(FeatureType.LONG)
        .with_order_number(4),
        FeatureDetail("dt_uinit8")
        .with_feature_type(FeatureType.INTEGER)
        .with_order_number(5),
        FeatureDetail("dt_uinit16")
        .with_feature_type(FeatureType.INTEGER)
        .with_order_number(6),
        FeatureDetail("dt_uinit32")
        .with_feature_type(FeatureType.LONG)
        .with_order_number(7),
        FeatureDetail("dt_float16")
        .with_feature_type(FeatureType.FLOAT)
        .with_order_number(8),
        FeatureDetail("dt_float32")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(9),
        FeatureDetail("dt_float64")
        .with_feature_type(FeatureType.DOUBLE)
        .with_order_number(10),
        FeatureDetail("dt_bool")
        .with_feature_type(FeatureType.BOOLEAN)
        .with_order_number(12),
    ]

    input_feature_details_datetime = [
        FeatureDetail("datetime64_ns_utc")
        .with_feature_type(FeatureType.TIMESTAMP)
        .with_order_number(1),
        FeatureDetail("datetime64_ns")
        .with_feature_type(FeatureType.TIMESTAMP)
        .with_order_number(2),
        FeatureDetail("timedelta64_ns")
        .with_feature_type(FeatureType.LONG)
        .with_order_number(3),
    ]

    def define_feature_group_resource_with_pandas_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("fg_pandas_datatype_basic"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_pandas)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def define_feature_group_resource_with_pandas_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("fg_pandas_datatype_basic_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def define_feature_group_resource_with_pandas_datetime_infer_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("fg_pandas_datetime"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_schema_details_from_dataframe(self.df_pandas_datetime)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def define_feature_group_resource_with_pandas_datetime_with_schema(
        self, entity_id, feature_store_id
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group resource for pandas datatypes")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("fg_pandas_datetime_schema"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_input_feature_details(self.input_feature_details_datetime)
            .with_statistics_config(False)
        )
        return feature_group_resource

    def test_feature_group_pandas_datatypes_infer_schema(self):
        """Test supported pandas data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_pandas_infer_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_pandas)
        df = feature_group.select().read()
        assert df.count() == 8
        assert len(df.columns) == 11

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_datatypes_with_schema(self):
        """Test supported pandas data types
        with input_feature_details"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = self.define_feature_group_resource_with_pandas_with_schema(
            entity.oci_fs_entity.id, fs.oci_fs.id
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_pandas)
        df = feature_group.select().read()
        assert df.count() == 8
        assert len(df.columns) == 11

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_dt_datatime_infer_schema(self):
        """Tests  pandas date time data types"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_datetime_infer_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_pandas_datetime)
        df = feature_group.select().read()
        assert df.count() == 8
        assert len(df.columns) == 3

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_pandas_dt_datatime_with_schema(self):
        """Tests  pandas adtetime data types with schema"""
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        feature_group = (
            self.define_feature_group_resource_with_pandas_datetime_with_schema(
                entity.oci_fs_entity.id, fs.oci_fs.id
            )
        )

        feature_group.create()
        assert feature_group.oci_feature_group.id

        feature_group.materialise(self.df_pandas_datetime)
        df = feature_group.select().read()
        assert df.count() == 8
        assert len(df.columns) == 3

        self.clean_up_feature_group(feature_group)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

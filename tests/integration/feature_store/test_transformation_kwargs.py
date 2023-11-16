from ads.feature_store.feature_group import FeatureGroup
from tests.integration.feature_store.test_base import FeatureStoreTestCase


class TestFeatureGroupWithKwargsTransformation(FeatureStoreTestCase):
    """Contains integration tests for Feature Group Kwargs supported transformation."""

    def define_feature_group_resource_with_transformation(
        self, entity_id, feature_store_id, transformation_id, transformation_kwargs
    ) -> "FeatureGroup":
        feature_group_resource = (
            FeatureGroup()
            .with_description("feature group with statistics disabled")
            .with_compartment_id(self.COMPARTMENT_ID)
            .with_name(self.get_name("petals2"))
            .with_entity_id(entity_id)
            .with_feature_store_id(feature_store_id)
            .with_primary_keys([])
            .with_partition_keys([])
            .with_input_feature_details(self.INPUT_FEATURE_DETAILS)
            .with_statistics_config(False)
            .with_transformation_id(transformation_id)
            .with_transformation_kwargs(transformation_kwargs)
        )
        return feature_group_resource

    def test_feature_group_materialization_with_kwargs_supported_transformation(self):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource(fs)
        transformation_kwargs = {"is_area_enabled": True}

        fg = self.define_feature_group_resource_with_transformation(
            entity.oci_fs_entity.id,
            fs.oci_fs.id,
            transformation.oci_fs_transformation.id,
            transformation_kwargs,
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        df = fg.preview(row_count=1)

        assert "petal_area" in df.columns
        assert "sepal_area" in df.columns

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_materialization_with_kwargs_supported_transformation_with_passing_kwargs_as_empty_dict(
        self,
    ):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource(fs)

        fg = self.define_feature_group_resource_with_transformation(
            entity.oci_fs_entity.id,
            fs.oci_fs.id,
            transformation.oci_fs_transformation.id,
            {},
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        df = fg.preview(row_count=1)

        assert "petal_area" not in df.columns
        assert "sepal_area" not in df.columns

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

    def test_feature_group_materialization_with_kwargs_supported_transformation_with_passing_kwargs_as_None(
        self,
    ):
        fs = self.define_feature_store_resource().create()
        assert fs.oci_fs.id

        entity = self.create_entity_resource(fs)
        assert entity.oci_fs_entity.id

        transformation = self.create_transformation_resource(fs)

        fg = self.define_feature_group_resource_with_transformation(
            entity.oci_fs_entity.id,
            fs.oci_fs.id,
            transformation.oci_fs_transformation.id,
            None,
        ).create()
        assert fg.oci_feature_group.id

        fg.materialise(self.data)

        df = fg.preview(row_count=1)

        assert "petal_area" not in df.columns
        assert "sepal_area" not in df.columns

        self.clean_up_feature_group(fg)
        self.clean_up_transformation(transformation)
        self.clean_up_entity(entity)
        self.clean_up_feature_store(fs)

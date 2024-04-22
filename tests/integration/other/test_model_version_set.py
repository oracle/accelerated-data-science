#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from uuid import uuid4
import random
import pytest
import shutil
import tempfile

from ads.model.model_version_set import ModelVersionSet, experiment
from ads.model.generic_model import GenericModel
from ads.catalog.model import ModelCatalog
from ads.common.model_metadata import UseCaseType
from tests.integration.config import secrets

SAMPLE_PAYLOAD = dict(
    compartment_id=secrets.common.COMPARTMENT_ID,
    name=f"adstest-mvs-{str(uuid4())}",
    project_id=secrets.common.PROJECT_OCID,
    description="adstest-mvs",
    freeform_tags=dict(my_tag=f"adstest-mvs-{str(uuid4())}"),
)

ARTIFACT_DIR = tempfile.mkdtemp()


class TestModelVersionSet:
    """Contains integration tests for Model Version Set."""

    @pytest.fixture(scope="class")
    def model_id(cls):
        class Square:
            def predict(self, x):
                print("predicting....")

        X = random.sample(range(0, 100), 10)

        generic_model = GenericModel(estimator=Square(), artifact_dir=ARTIFACT_DIR)
        generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",  ##Chose the conda env you want to execute under
            training_conda_env="dataexpl_p37_cpu_v3",
            use_case_type=UseCaseType.MULTINOMIAL_CLASSIFICATION,
            X_sample=X,
            force_overwrite=True,
        )
        model_id = generic_model.save(display_name="Generic Model Experiment_1")
        yield model_id
        ModelCatalog(compartment_id=SAMPLE_PAYLOAD["compartment_id"]).delete_model(
            model_id
        )

    @pytest.fixture(scope="class")
    def mvs(self):
        """Creates model version set used for testing."""
        mvs = (
            ModelVersionSet()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_project_id(SAMPLE_PAYLOAD["project_id"])
            .with_name(SAMPLE_PAYLOAD["name"])
            .with_description(SAMPLE_PAYLOAD["description"])
            .with_freeform_tags(**SAMPLE_PAYLOAD["freeform_tags"])
        ).create()
        yield mvs
        if mvs.status in ["ACTIVE", "FAILED"]:
            mvs.delete()

    @pytest.fixture(scope="class")
    def yaml_file(self):
        """Creates well-defined yaml used for testing."""
        name = f"adstest-mvs-{str(uuid4())}"
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        # os.path.join(curr_dir, "modelversionset-test-files", "test_mvs.yaml")
        yaml_path = "/home/datascience/model_store_test/test_mvs.yaml"
        mvs = (
            ModelVersionSet()
            .with_compartment_id(SAMPLE_PAYLOAD["compartment_id"])
            .with_project_id(SAMPLE_PAYLOAD["project_id"])
            .with_name(name)
            .with_description(SAMPLE_PAYLOAD["description"])
        )
        mvs.to_yaml(yaml_path)
        yield [yaml_path, name]
        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)

    def test_create_with_constructor(self):
        """Tests creating model version set based on constructor."""
        name = f"adstest-mvs-{str(uuid4())}"
        mvs_c = ModelVersionSet(
            compartment_id=SAMPLE_PAYLOAD["compartment_id"],
            name=name,
            projectId=SAMPLE_PAYLOAD["project_id"],
            description=SAMPLE_PAYLOAD["description"],
        )
        mvs_c.create()
        assert mvs_c.dsc_model_version_set.id
        #         assert mvs.status == "ACTIVE"
        assert mvs_c.name == name
        mvs_c.delete()

    def test_create_with_builder_pattern(self, mvs):
        """Tests creating model version set with builder pattern."""
        assert mvs.dsc_model_version_set.id
        assert mvs.compartment_id == SAMPLE_PAYLOAD["compartment_id"]
        assert mvs.project_id == SAMPLE_PAYLOAD["project_id"]
        assert mvs.name == SAMPLE_PAYLOAD["name"]
        assert mvs.description == SAMPLE_PAYLOAD["description"]

    def test_create_from_yaml(self, yaml_file):
        """Test creating model version set from defined yaml."""
        mvs_y = ModelVersionSet.from_yaml(uri=yaml_file[0])
        mvs_y.create()
        assert mvs_y.dsc_model_version_set.id
        #         assert mvs_y.status == "ACTIVE"
        assert mvs_y.name == yaml_file[1]
        mvs_y.delete()

    def test_update_mvs(self, mvs):
        """Tests updating  model version set with new deescription and freeform tags."""
        mvs = mvs.with_description("Updated description.").update()
        assert mvs.description == "Updated description."

    def test_get_by_id(self, mvs):
        """Tests getting model version set from id."""
        mvs_id = mvs.id
        mvs = ModelVersionSet.from_id(mvs_id)
        assert mvs.dsc_model_version_set.id
        assert mvs.name == SAMPLE_PAYLOAD["name"]

    def test_get_by_name(self, mvs):
        """Tests getting model version set from name."""
        mvs_id = mvs.id
        mvs = ModelVersionSet.from_name(SAMPLE_PAYLOAD["name"])
        assert mvs.id == mvs_id

    def test_list_mvs(self, mvs):
        """Tests listing all model version set in given compartment."""
        mvs_list = ModelVersionSet.list(SAMPLE_PAYLOAD["compartment_id"])
        ids = [m.id for m in mvs_list]
        assert mvs.id in ids

    def test_list_model_with_mvs(self, mvs, model_id):
        """Tests listing all models with specific model version set."""
        mvs.model_add(model_id, version_label="ADSTEST_MVS")

        models = mvs.models()
        assert models[0].model_version_set_id == mvs.id

    def test_associated_with_existing_model(self, mvs):
        """Tests getting a list of models associated with given mvs through model catalog."""
        model_catalog = ModelCatalog(compartment_id=SAMPLE_PAYLOAD["compartment_id"])
        models = model_catalog.list_models(model_version_set_name=mvs.name)
        assert models[0].model_version_set_id == mvs.id

    def test_associated_with_new_model(self, mvs):
        """Tests adding model to a Model Version Set using a build in Context Manager."""

        class Toy:
            def predict(self, x):
                return x**2

        model = Toy()

        generic_model = GenericModel(estimator=model, artifact_dir=ARTIFACT_DIR)
        generic_model.prepare(
            inference_conda_env="dataexpl_p37_cpu_v3",
            model_file_name="toy_model.pkl",
            force_overwrite=True,
        )

        with experiment(name=mvs.name, create_if_not_exists=True):
            # experiment 1
            m_id_1 = generic_model.save(
                display_name="Generic Model Experiment_1", version_label="ADSTEST_MVS_1"
            )

            # experiment 2
            m_id_2 = generic_model.save(
                display_name="Generic Model Experiment_2", version_label="ADSTEST_MVS_2"
            )

            # experiment 3
            m_id_3 = generic_model.save(
                display_name="Generic Model Experiment_3", version_label="ADSTEST_MVS_3"
            )

        models = mvs.models()
        for model in models:
            assert model.model_version_set_id == mvs.id

        for i in [m_id_1, m_id_2, m_id_3]:
            ModelCatalog(compartment_id=SAMPLE_PAYLOAD["compartment_id"]).delete_model(
                i
            )

    def teardown_class(self):
        shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)

        model_catalog = ModelCatalog(compartment_id=SAMPLE_PAYLOAD["compartment_id"])
        models = model_catalog.list_models(
            project_id=SAMPLE_PAYLOAD["project_id"],
        )
        for model in models:
            model_catalog.delete_model(model)

        mvs_list = ModelVersionSet.list(
            SAMPLE_PAYLOAD["compartment_id"], project_id=SAMPLE_PAYLOAD["project_id"]
        )
        for mvs in mvs_list:
            if mvs.status in ["ACTIVE", "FAILED"]:
                mvs.delete()

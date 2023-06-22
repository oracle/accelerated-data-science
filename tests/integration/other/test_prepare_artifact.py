#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import cloudpickle
import lightgbm
import oci
import pandas as pd
import pytest
from ads.catalog.model import ModelCatalog
from ads.common.model import ADSModel
from ads.common.model_export_util import prepare_generic_model
from ads.model.model_metadata import Framework, UseCaseType
from ads.dataset.factory import DatasetFactory
from ads.dataset.label_encoder import DataFrameLabelEncoder
from ads.feature_engineering.schema import Expression
from lightgbm import LGBMClassifier
from tests.integration.config import secrets
from xgboost import XGBRegressor


class TestModelArtifactPrepare(TestCase):
    """Contains test cases for prepare method"""

    def setUp(self) -> None:
        """Sets up the test case."""
        super().setUp()
        # load data
        file_name = "vor_airline_sentiment.csv"
        self.file_path = os.path.join(
            os.path.dirname(__file__), "../../vor_datasets", file_name
        )
        self.df = pd.read_csv(self.file_path)
        le = DataFrameLabelEncoder()
        df_transformed = le.fit_transform(self.df)
        self.y = df_transformed["airline_sentiment"]
        self.X = df_transformed.drop(["airline_sentiment"], axis=1)

        self.clf = LGBMClassifier()
        self.clf.fit(self.X, self.y)
        self.model = ADSModel.from_estimator(self.clf)
        self.model_dir = tempfile.mkdtemp()
        self.download_dir = tempfile.mkdtemp()

    @staticmethod
    def get_test_dataset_path(file_name):
        return os.path.join(os.path.dirname(__file__), "../../vor_datasets", file_name)

    @pytest.mark.skip(reason="fails - Test fails because of connection timeout.")
    def test_prepare(self):
        model_artifact = self.model.prepare(
            self.model_dir,
            X_sample=self.X.head(),
            y_sample=self.y.head(),
            force_overwrite=True,
            data_science_env=True,
            ignore_deployment_error=False,
        )
        assert isinstance(model_artifact.metadata_taxonomy.to_dataframe(), pd.DataFrame)
        assert model_artifact.metadata_taxonomy.to_dataframe().shape == (6, 2)
        assert (
            model_artifact.metadata_taxonomy["Algorithm"].value
            == self.clf.__class__.__name__
        )
        assert (
            model_artifact.metadata_taxonomy["Framework"].value == Framework.LIGHT_GBM
        )
        assert (
            model_artifact.metadata_taxonomy["FrameworkVersion"].value
            == lightgbm.__version__
        )
        assert model_artifact.metadata_taxonomy["Hyperparameters"].value is not None

        assert isinstance(model_artifact.metadata_custom.to_dataframe(), pd.DataFrame)
        assert model_artifact.metadata_custom.to_dataframe().shape == (6, 4)
        assert model_artifact.metadata_custom["EnvironmentType"].value == "data_science"
        # assert model_artifact.metadata_custom["CondaEnvironment"].value is not None
        assert model_artifact.metadata_custom["SlugName"].value is not None

        assert set(model_artifact.schema_input.keys) == set(self.X.columns)
        assert model_artifact.schema_output[self.y.name].required
        mc_model = model_artifact.save(
            project_id=secrets.common.PROJECT_OCID,
            compartment_id=secrets.common.COMPARTMENT_ID,
            display_name="advanced-ds-test",
            description="A sample LightGBM classifier",
            ignore_pending_changes=True,
            timeout=600,
        )

        self.compare_metadata(mc_model, model_artifact)
        # modify the metadata of model_artifact
        self.update_metadata_schema(model_artifact)
        mc_model_updated = model_artifact.save(
            project_id=secrets.common.PROJECT_OCID,
            compartment_id=secrets.common.COMPARTMENT_ID,
            display_name="advanced-ds-test",
            description="A sample LightGBM classifier",
            ignore_pending_changes=True,
            timeout=600,
        )

        self.compare_metadata(mc_model_updated, model_artifact)
        self.compare_schema(mc_model_updated, model_artifact)
        assert mc_model.metadata_custom["SlugName"].value != "test"
        assert mc_model.metadata_taxonomy["Algorithm"].value != "test"
        assert mc_model.schema_input["negativereason"].domain.constraints == []

        # modify the metadata of mc_model
        self.update_metadata_schema(mc_model)
        self.compare_metadata(mc_model, mc_model_updated)
        self.compare_schema(mc_model_updated, model_artifact)

        # rollback the changes
        mc_model.rollback()
        assert mc_model.metadata_custom["SlugName"].value != "test"
        assert mc_model.metadata_taxonomy["Algorithm"].value != "test"
        assert mc_model.schema_input["negativereason"].domain.constraints == []

        # commit the changes
        self.update_metadata_schema(mc_model)
        self.compare_metadata(mc_model, mc_model_updated)
        self.compare_schema(mc_model_updated, model_artifact)
        mc_model.commit()
        self.compare_metadata(mc_model, mc_model_updated)

        # Note input and schema_output wont be updated.
        mc_model.rollback()
        self.compare_metadata(mc_model, mc_model_updated)

        # download model
        mc = ModelCatalog(compartment_id=secrets.common.COMPARTMENT_ID, timeout=600)
        dl_model_artifact = mc.download_model(
            mc_model.id, self.download_dir, force_overwrite=True
        )
        self.compare_metadata(mc_model, dl_model_artifact)

        assert dl_model_artifact.metadata_taxonomy["FrameworkVersion"].value == 2.1
        assert dl_model_artifact.metadata_taxonomy["Hyperparameters"].value == {
            "reg_lambda": 1.0
        }
        assert dl_model_artifact.metadata_custom["SlugName"].value == "test"
        # upload model
        uploaded_model = mc.upload_model(
            model_artifact, display_name="advanced-ds-test"
        )

        mc.delete_model(mc_model.id)
        mc.delete_model(uploaded_model.id)

    @staticmethod
    def compare_metadata(model1, model2):
        for key in model1.metadata_custom.keys:
            assert model1.metadata_custom[key] == model2.metadata_custom[key]
        for key in model1.metadata_taxonomy.keys:
            assert model1.metadata_taxonomy[key] == model2.metadata_taxonomy[key]

    @staticmethod
    def compare_schema(model1, model2):
        assert model1.schema_input == model2.schema_input
        assert model1.schema_output == model2.schema_output

    @staticmethod
    def update_metadata_schema(model):
        model.metadata_custom["SlugName"].value = "test"
        model.metadata_taxonomy["FrameworkVersion"].value = 2.1
        model.metadata_taxonomy["Hyperparameters"].value = {"reg_lambda": 1.0}
        model.schema_input["negativereason"].domain.constraints.append(
            Expression(
                """$x in ['Bad Flight', "Can't Tell", 'Late Flight', 'Customer Service Issue', 'Flight Booking Problems', 'Lost Luggage', 'Flight Attendant Complaints', 'Cancelled Flight', 'Damaged Luggage', 'longlines']"""
            )
        )
        model.schema_input["negativereason"].domain.constraints[-1].evaluate(
            x="""\"Can't Tell\""""
        )

    def test_model_schema(self):
        df = pd.read_csv(self.file_path)
        df.ads.feature_type = {
            "tweet_id": ["integer"],
            "airline_sentiment": ["category"],
            "airline_sentiment_confidence": ["continuous"],
            "negativereason": ["category"],
            "airline": ["category"],
            "airline_sentiment_gold": ["category"],
            "name": ["category"],
            "negativereason_gold": ["category"],
            "retweet_count": ["integer"],
            "text": ["text"],
            "tweet_coord": ["string"],
            "tweet_created": ["date_time"],
            "user_timezone": ["date_time"],
        }
        schema = df.ads.model_schema()
        assert schema["negativereason"].domain.constraints[0].evaluate(x="'Bad Flight'")
        assert (
            schema["negativereason"]
            .domain.constraints[0]
            .evaluate(x="'Invalid Option'")
            is False
        )
        assert schema["negativereason"].required is False
        assert schema["negativereason"].feature_type == "Category"

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        shutil.rmtree(self.download_dir)
        return super().tearDown()


class TestModelArtifactPrepareGenericArtifact(TestCase):
    """Contains test cases for prepare method"""

    def setUp(self) -> None:
        """Sets up the test case."""
        AUTH = "api_key"
        BUCKET_NAME = secrets.other.BUCKET_3
        NAMESPACE = secrets.common.NAMESPACE
        OS_PREFIX = "unit_test"
        profile = "DEFAULT"
        self.oci_path = f"oci://{BUCKET_NAME}@{NAMESPACE}/{OS_PREFIX}"
        config_path = os.path.expanduser(os.path.join("~/.oci", "config"))
        self.storage_options = {"config": oci.config.from_file(config_path)}

        file_name = "vor_house_price.csv"
        self.file_path = os.path.join(
            os.path.dirname(__file__), "../../vor_datasets", file_name
        )

        self.ds = DatasetFactory.open(self.file_path, target="SalePrice")

        self.train, self.test = self.ds.train_test_split(test_size=0.15)
        le = DataFrameLabelEncoder()
        self.train_X = le.fit_transform(self.train.X)
        self.train_y = self.train.y.values

        self.clf = XGBRegressor(n_estimators=10).fit(self.train_X, self.train_y)
        self.rf_model = ADSModel.from_estimator(self.clf)
        self.model_dir = tempfile.mkdtemp()
        super().setUp()

    def test_prepare_model_artifact_ads_model_raise_error(self):
        with pytest.raises(
            ValueError,
            match="Only generic model can be used to generate generic model artifact.",
        ):
            prepare_generic_model(self.model_dir, model=self.rf_model)

    def test_prepare_model_artifact_invalid_use_case_type(self):
        with pytest.raises(ValueError):
            model_artifact = prepare_generic_model(
                self.model_dir, model=self.clf, use_case_type="invalid"
            )

    def test_populate_metatdata_invalid_use_case_type(self):
        model_artifact = prepare_generic_model(
            self.model_dir, data_science_env=True, force_overwrite=True
        )
        with pytest.raises(ValueError):
            model_artifact.populate_metadata(
                model=self.rf_model, use_case_type="Invalid"
            )

    @patch("ads.common.model_artifact._TRAINING_RESOURCE_OCID", None)
    def test_prepare_model_artifact(self):
        model_artifact = prepare_generic_model(
            self.model_dir,
            use_case_type=UseCaseType.BINARY_CLASSIFICATION,
            data_science_env=True,
            force_overwrite=True,
        )
        assert not os.path.exists(os.path.join(self.model_dir, "schema_input.json"))
        assert not os.path.exists(os.path.join(self.model_dir, "schema_output.json"))
        assert (
            model_artifact.metadata_taxonomy["usecasetype"].value
            == UseCaseType.BINARY_CLASSIFICATION
        )
        assert model_artifact.metadata_custom["condaenvironment"] is not None
        assert model_artifact.metadata_custom["condaenvironmentpath"] is not None
        assert model_artifact.metadata_custom["environmenttype"] is not None
        assert model_artifact.metadata_custom["slugname"] is not None

        with open(os.path.join(self.model_dir, "model.pkl"), "wb") as outfile:
            cloudpickle.dump(self.clf, outfile)

        model_artifact.populate_schema(data_sample=self.test)
        assert os.path.exists(os.path.join(self.model_dir, "input_schema.json"))
        assert os.path.exists(os.path.join(self.model_dir, "output_schema.json"))

        model_artifact.populate_metadata(model=self.clf)
        assert (
            model_artifact.metadata_taxonomy["algorithm"].value
            == self.clf.__class__.__name__
        )
        assert model_artifact.metadata_taxonomy["framework"].value == Framework.XGBOOST
        assert (
            model_artifact.metadata_taxonomy["usecasetype"].value
            == UseCaseType.BINARY_CLASSIFICATION
        )

        metadata_taxonomy = model_artifact.metadata_taxonomy
        # reload
        model_artifact.reload(model_file_name="model.pkl")
        for key in metadata_taxonomy.keys:
            assert model_artifact.metadata_taxonomy[key] == metadata_taxonomy[key]
        assert (
            model_artifact.metadata_taxonomy["algorithm"].value
            == self.clf.__class__.__name__
        )
        assert model_artifact.metadata_taxonomy["framework"].value == Framework.XGBOOST
        assert (
            model_artifact.metadata_taxonomy["usecasetype"].value
            == UseCaseType.BINARY_CLASSIFICATION
        )
        assert "ArtifactTestResults" in model_artifact.metadata_taxonomy.keys
        assert model_artifact.metadata_custom["modelserializationformat"].value == "pkl"

        model_artifact.populate_metadata(
            use_case_type=UseCaseType.MULTINOMIAL_CLASSIFICATION
        )
        assert (
            model_artifact.metadata_taxonomy["algorithm"].value
            == self.clf.__class__.__name__
        )
        assert model_artifact.metadata_taxonomy["framework"].value == Framework.XGBOOST
        assert (
            model_artifact.metadata_taxonomy["usecasetype"].value
            == UseCaseType.MULTINOMIAL_CLASSIFICATION
        )

        # save data
        model_artifact._save_data_from_memory(
            self.oci_path,
            train_data=self.train.X,
            validation_data=self.test.X,
            train_data_name="house_price_train.csv",
            validation_data_name="house_price_validation.csv",
            storage_options=self.storage_options,
        )

        assert (
            model_artifact.metadata_custom["validationdatasetsize"].value == "(219, 80)"
        )
        assert (
            model_artifact.metadata_custom["trainingdatasetsize"].value == "(1241, 80)"
        )
        assert (
            model_artifact.metadata_custom["trainingdataset"].value
            == f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/unit_test/house_price_train.csv"
        )
        assert (
            model_artifact.metadata_custom["validationdataset"].value
            == f"oci://{secrets.other.BUCKET_3}@{secrets.common.NAMESPACE}/unit_test/house_price_validation.csv"
        )
        # save a file with a long name.
        model_artifact.metadata_custom["modelartifacts"].to_json_file(
            os.path.join(self.model_dir, "a" * 10 + ".json")
        )
        model_artifact.metadata_custom["modelartifacts"].to_json_file(
            os.path.join(self.model_dir, "b" * 10 + ".json")
        )
        model_artifact.metadata_custom["modelartifacts"].to_json_file(
            os.path.join(self.model_dir, "c" * 10 + ".json")
        )
        model_artifact.metadata_custom["modelartifacts"].to_json_file(
            os.path.join(self.model_dir, "d" * 10 + ".json")
        )
        model_artifact.metadata_custom.set_training_data(
            path="oci://bucket_name@namespace/train_data_filename",
            data_size="(200,100)",
        )
        model_artifact.metadata_custom.set_validation_data(
            path="oci://bucket_name@namespace/validation_data_filename",
            data_size="(100,100)",
        )
        assert (
            model_artifact.metadata_custom["trainingdataset"].value
            == "oci://bucket_name@namespace/train_data_filename"
        )
        assert (
            model_artifact.metadata_custom["trainingdatasetsize"].value == "(200,100)"
        )

        assert (
            model_artifact.metadata_custom["validationdataset"].value
            == "oci://bucket_name@namespace/validation_data_filename"
        )
        assert (
            model_artifact.metadata_custom["validationdatasetsize"].value == "(100,100)"
        )

        mc_model = model_artifact.save(
            display_name="advanced-ds-test", training_id=None
        )
        mc = ModelCatalog(compartment_id=secrets.common.COMPARTMENT_ID)
        assert mc.delete_model(mc_model) == True

    def tearDown(self) -> None:
        shutil.rmtree(self.model_dir)
        return super().tearDown()

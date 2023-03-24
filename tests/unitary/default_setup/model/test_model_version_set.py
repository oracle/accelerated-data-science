#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import os
from copy import copy
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from ads.common.oci_mixin import OCIModelMixin, OCIModelNotExists, OCIModelWithNameMixin
from ads.common.utils import batch_convert_case
from ads.model import DataScienceModel
from ads.model.model_version_set import (
    _MVS_COMPARTMENT_ENV_VAR,
    _MVS_ID_ENV_VAR,
    _MVS_NAME_ENV_VAR,
    _MVS_URL,
    ModelVersionSet,
    experiment,
)
from ads.model.service.oci_datascience_model_version_set import (
    LIFECYCLE_STOP_STATE,
    DataScienceModelVersionSet,
    ModelVersionSetNotExists,
    ModelVersionSetNotSaved,
)
from oci.data_science.models import ModelVersionSet as OCIModelVersionSet
from oci.data_science.models import UpdateModelDetails
from oci.response import Response

MVS_PAYLOAD = dict(
    compartment_id="ocid1.compartment.oc1..<unique_ocid>",
    project_id="ocid1.datascienceproject.oc1.iad.<unique_ocid>",
)
MVS_OCID = "ocid.xxx.datasciencemodelversionset.<unique_ocid>"


class TestDataScienceModelVersionSet:
    def setup_class(cls):

        cls.mock_date = datetime.datetime(2022, 7, 1)
        cls.mock_delete_mvs_response = Response(
            data=None, status=None, headers=None, request=None
        )

    @property
    def mock_create_mvs_response(self):
        payload = copy(MVS_PAYLOAD)
        payload["lifecycle_state"] = OCIModelVersionSet.LIFECYCLE_STATE_ACTIVE
        payload["id"] = MVS_OCID
        return Response(
            data=OCIModelVersionSet(**payload), status=None, headers=None, request=None
        )

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_model_version_set = Mock(
            return_value=self.mock_create_mvs_response
        )
        mock_client.delete_model_version_set = Mock(
            return_value=self.mock_delete_mvs_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model version set fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="`compartment_id` must be specified for the model version set.",
        ):
            DataScienceModelVersionSet().create()
        with pytest.raises(
            ValueError,
            match="`project_id` must be specified for the model version set.",
        ):
            DataScienceModelVersionSet(compartment_id="test").create()

    @pytest.fixture(scope="class")
    def mock_to_dict(self):
        return MagicMock(return_value=MVS_PAYLOAD)

    def test_create_success(self, mock_client, mock_to_dict):
        """Ensures creating model version set passes in case of valid input params."""

        datetime_mock = Mock(wraps=datetime.datetime)
        datetime_mock.now.return_value = self.mock_date
        with patch("datetime.datetime", new=datetime_mock):
            timestamp = self.mock_date.strftime("%Y%m%d-%H%M")
            dmvs = DataScienceModelVersionSet(**MVS_PAYLOAD)

            with patch.object(DataScienceModelVersionSet, "client", mock_client):
                with patch.object(DataScienceModelVersionSet, "to_dict", mock_to_dict):
                    dmvs.create()
                    mock_client.create_model_version_set.assert_called()
                    assert dmvs.name == f"model-version-set-{timestamp}"
                    assert dmvs.id == MVS_OCID
                    assert (
                        dmvs.lifecycle_state
                        == OCIModelVersionSet.LIFECYCLE_STATE_ACTIVE
                    )

    def test_delete_success(self, mock_client):
        """Ensures model version set can be deleted."""
        dmvs = DataScienceModelVersionSet(**MVS_PAYLOAD)
        with patch.object(DataScienceModelVersionSet, "client", mock_client):
            with patch.object(
                DataScienceModelVersionSet, "get_work_request_response"
            ) as mock_get_work_request_response:
                with patch.object(DataScienceModelVersionSet, "sync") as mock_sync:
                    dmvs.delete(delete_model=True)
                    mock_client.delete_model_version_set.assert_called_with(
                        None, is_delete_related_models=True
                    )
                    mock_get_work_request_response.assert_called_with(
                        self.mock_delete_mvs_response,
                        wait_for_state=LIFECYCLE_STOP_STATE,
                        success_state="SUCCEEDED",
                        wait_interval_seconds=1,
                    )
                    mock_sync.assert_called()

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_ocid(self, mock_from_ocid):
        """Tests getting a Model Version Set by OCID."""
        DataScienceModelVersionSet.from_ocid(MVS_OCID)
        mock_from_ocid.assert_called_with(MVS_OCID)

    @patch.object(OCIModelWithNameMixin, "from_name")
    def test_from_name_success(self, mock_from_name):
        """Ensures Model Version Set can be loaded by name."""
        DataScienceModelVersionSet.from_name(
            name="test_name", compartment_id="test_compartment_id"
        )
        mock_from_name.assert_called_with(
            name="test_name", compartment_id="test_compartment_id"
        )

    @patch.object(OCIModelWithNameMixin, "from_name")
    def test_from_name_fail(self, mock_from_name):
        """Ensures loading version set fails if it doesn"t exist."""
        mock_from_name.side_effect = OCIModelNotExists()
        with pytest.raises(
            ModelVersionSetNotExists,
            match=f"The model version set `test_name` not found.",
        ):
            DataScienceModelVersionSet.from_name(
                name="test_name", compartment_id="test_compartment_id"
            )


class TestModelVersionSet:
    def setup_class(cls):
        cls.mock_date = datetime.datetime(2022, 7, 1)

    def setup_method(self):
        self.mock_default_properties = {
            "compartment_id": MVS_PAYLOAD["compartment_id"],
            "project_id": MVS_PAYLOAD["project_id"],
            "name": f"model-version-set-{self.mock_date.strftime('%Y%m%d-%H%M')}",
        }
        self.payload = {
            **self.mock_default_properties,
            "defined_tags": {"key": {"key": "value"}},
            "freeform_tags": {"key1": "value1"},
            "name": "test_name",
            "description": "test_description",
        }

    @patch.object(ModelVersionSet, "_load_default_properties")
    def test__init(self, mock_load_default_properties):
        mock_load_default_properties.return_value = self.mock_default_properties
        mvs = ModelVersionSet()
        assert mvs.to_dict()["spec"] == batch_convert_case(
            self.mock_default_properties, to_fmt="camel"
        )

    @patch.object(ModelVersionSet, "_load_default_properties")
    def test__init_with_spec(self, mock_load_default_properties):
        mock_load_default_properties.return_value = self.mock_default_properties
        mvs = ModelVersionSet(spec=self.payload)
        assert mvs.to_dict()["spec"] == batch_convert_case(self.payload, to_fmt="camel")

    @patch.object(ModelVersionSet, "_load_default_properties")
    def test__init_with_kwargs(self, mock_load_default_properties):
        mock_load_default_properties.return_value = self.mock_default_properties
        mvs = ModelVersionSet(**self.payload)
        assert mvs.to_dict()["spec"] == batch_convert_case(self.payload, to_fmt="camel")

    def test_create_with_builder_pattern(self):
        mvs = (
            ModelVersionSet()
            .with_compartment_id(self.payload["compartment_id"])
            .with_project_id(self.payload["project_id"])
            .with_description(self.payload["description"])
            .with_name(self.payload["name"])
            .with_freeform_tags(**self.payload["freeform_tags"])
            .with_defined_tags(**self.payload["defined_tags"])
        )

        assert mvs.to_dict()["spec"] == batch_convert_case(self.payload, to_fmt="camel")

    @patch.object(DataScienceModelVersionSet, "update")
    def test_update(self, mock_update):
        test_dmvs = DataScienceModelVersionSet(**self.mock_default_properties)
        mock_update.return_value = test_dmvs
        mvs = ModelVersionSet()
        mvs.update()
        mock_update.assert_called()
        assert mvs.to_dict()["spec"] == batch_convert_case(
            self.mock_default_properties, to_fmt="camel"
        )

    @patch.object(DataScienceModelVersionSet, "delete")
    def test_delete(self, mock_delete):
        mvs = ModelVersionSet()
        mvs.delete()
        mock_delete.assert_called_with(False)
        mvs.delete(delete_model=True)
        mock_delete.assert_called_with(True)

    def test_model_add(self):
        mvs = ModelVersionSet()
        with pytest.raises(
            ModelVersionSetNotSaved, match="Model version set needs to be saved."
        ):
            mvs.model_add(model_id="test_model_id")

        mvs.dsc_model_version_set.id = MVS_OCID

        mock_client = MagicMock()
        mock_update_model = MagicMock()
        mock_client.update_model = mock_update_model

        with patch.object(DataScienceModelVersionSet, "client", mock_client):

            version_label = "test_version_label"

            expected_model_details = UpdateModelDetails(
                model_version_set_id=MVS_OCID, version_label=version_label
            )

            mvs.model_add(
                model_id="test_model_id",
                version_label=version_label,
                some_kward_param="test_kward",
            )

            mock_update_model.asssert_called_with(
                "test_model_id", expected_model_details, some_kward_param="test_kward"
            )

    @patch(
        "ads.model.model_version_set.COMPARTMENT_OCID", MVS_PAYLOAD["compartment_id"]
    )
    @patch("ads.model.model_version_set.PROJECT_OCID", MVS_PAYLOAD["project_id"])
    def test__load_default_properties(self):
        datetime_mock = Mock(wraps=datetime.datetime)
        datetime_mock.now.return_value = self.mock_date
        with patch("datetime.datetime", new=datetime_mock):
            test_result = ModelVersionSet._load_default_properties()
            assert test_result == batch_convert_case(
                self.mock_default_properties, to_fmt="camel"
            )

    @patch.object(
        DataScienceModelVersionSet,
        "status",
        new_callable=PropertyMock,
        return_value=OCIModelVersionSet.LIFECYCLE_STATE_ACTIVE,
    )
    def test_status(self, mock_status):

        mvs = ModelVersionSet()
        assert mvs.status == OCIModelVersionSet.LIFECYCLE_STATE_ACTIVE
        mock_status.assert_called()

    @patch(
        "ads.model.model_version_set.OCI_REGION_METADATA",
        '{"regionIdentifier": "test_region_1"}',
    )
    def test_details_link(self):
        mvs = ModelVersionSet()
        mvs.dsc_model_version_set.id = MVS_OCID
        with patch.object(
            DataScienceModelVersionSet,
            "auth",
            new_callable=PropertyMock,
            return_value={"config": {"region": "test_region"}},
        ):
            assert mvs.details_link == _MVS_URL.format(
                region="test_region", id=MVS_OCID
            )

        with patch.object(
            DataScienceModelVersionSet,
            "auth",
            new_callable=PropertyMock,
            return_value={"config": {}},
        ):
            assert mvs.details_link == _MVS_URL.format(
                region="test_region_1", id=MVS_OCID
            )

    def test_list(self):

        dmvs_list = [
            DataScienceModelVersionSet(**self.mock_default_properties),
            DataScienceModelVersionSet(**self.mock_default_properties),
            DataScienceModelVersionSet(**self.mock_default_properties),
        ]
        expected_result = [
            ModelVersionSet.from_dsc_model_version_set(dmvs) for dmvs in dmvs_list
        ]
        mvs = ModelVersionSet()
        with patch.object(
            DataScienceModelVersionSet, "list_resource"
        ) as mock_list_resource:
            mock_list_resource.return_value = dmvs_list
            test_result = mvs.list(compartment_id="test_compartment_id")
            for i, item in enumerate(test_result):
                assert item.to_dict() == expected_result[i].to_dict()
            mock_list_resource.assert_called_with("test_compartment_id")

    @patch.object(DataScienceModel, "list")
    def test_models(self, mock_list_models):
        mvs = ModelVersionSet(**self.mock_default_properties)
        with pytest.raises(
            ModelVersionSetNotSaved, match="Model version set needs to be saved."
        ):
            mvs.models()

        mvs.dsc_model_version_set.id = MVS_OCID

        with patch.object(DataScienceModelVersionSet, "client", "dsc_client"):

            mvs.models(
                project_id=self.mock_default_properties["project_id"],
                include_deleted=True,
            )

            mock_list_models.assert_called_with(
                compartment_id=self.mock_default_properties["compartment_id"],
                project_id=self.mock_default_properties["project_id"],
                include_deleted=True,
                model_version_set_name=self.mock_default_properties["name"],
            )

    def test_from_dsc_model_version_set(self):
        dmvs = DataScienceModelVersionSet(**self.mock_default_properties)
        mvs = ModelVersionSet.from_dsc_model_version_set(dmvs)
        assert mvs.to_dict()["spec"] == batch_convert_case(
            self.mock_default_properties, to_fmt="camel"
        )

    def test_from_id(self):
        dmvs = DataScienceModelVersionSet(**self.mock_default_properties)
        with patch.object(DataScienceModelVersionSet, "from_ocid", return_value=dmvs):
            mvs = ModelVersionSet.from_id(MVS_OCID)
            assert mvs.to_dict()["spec"] == batch_convert_case(
                self.mock_default_properties, to_fmt="camel"
            )

    @patch.object(ModelVersionSet, "from_id")
    def test_from_ocid(self, mock_from_id):
        ModelVersionSet.from_ocid(MVS_OCID)
        mock_from_id.assert_called_with(MVS_OCID)

    @patch(
        "ads.model.model_version_set.COMPARTMENT_OCID", MVS_PAYLOAD["compartment_id"]
    )
    def test_from_name(self):
        dmvs = DataScienceModelVersionSet(**self.mock_default_properties)
        with patch.object(
            DataScienceModelVersionSet, "from_name", return_value=dmvs
        ) as mock_from_name:
            mvs = ModelVersionSet.from_name("test_name")
            assert mvs.to_dict()["spec"] == batch_convert_case(
                self.mock_default_properties, to_fmt="camel"
            )
            mock_from_name.assert_called_with(
                name="test_name", compartment_id=MVS_PAYLOAD["compartment_id"]
            )

    def test_model_version_set_context_lib_fail(self):

        with patch.object(
            ModelVersionSet,
            "from_name",
        ) as mock_from_name:
            mock_from_name.side_effect = ModelVersionSetNotExists()
            with pytest.raises(ModelVersionSetNotExists):
                with experiment(
                    name="test_name",
                    compartment_id="test_compartment_id",
                    create_if_not_exists=False,
                ):
                    mock_from_name.assert_called_with(
                        name="from_name", compartment_id="test_compartment_id"
                    )

    def test_model_version_set_context_lib_create_if_not_exist(self):
        mvs = ModelVersionSet(**self.mock_default_properties)
        mvs.dsc_model_version_set.id = MVS_OCID
        with patch.object(
            ModelVersionSet,
            "from_name",
        ) as mock_from_name:
            with patch.object(ModelVersionSet, "create") as mock_create:
                mock_from_name.side_effect = ModelVersionSetNotExists()
                mock_create.return_value = mvs
                with experiment(
                    create_if_not_exists=True, **self.mock_default_properties
                ):
                    mock_from_name.assert_called_with(
                        name=self.mock_default_properties["name"],
                        compartment_id=self.mock_default_properties["compartment_id"],
                    )

                    assert os.environ[_MVS_ID_ENV_VAR] == mvs.id
                    assert os.environ[_MVS_NAME_ENV_VAR] == mvs.name
                    assert os.environ[_MVS_COMPARTMENT_ENV_VAR] == mvs.compartment_id

        assert _MVS_ID_ENV_VAR not in os.environ
        assert _MVS_NAME_ENV_VAR not in os.environ
        assert _MVS_COMPARTMENT_ENV_VAR not in os.environ

    def test_model_version_set_context_lib_success(self):
        mvs = ModelVersionSet(**self.mock_default_properties)
        mvs.dsc_model_version_set.id = MVS_OCID
        with patch.object(
            ModelVersionSet,
            "from_name",
        ) as mock_from_name:
            mock_from_name.return_value = mvs
            with experiment(create_if_not_exists=False, **self.mock_default_properties):
                mock_from_name.assert_called_with(
                    name=self.mock_default_properties["name"],
                    compartment_id=self.mock_default_properties["compartment_id"],
                )

                assert os.environ[_MVS_ID_ENV_VAR] == mvs.id
                assert os.environ[_MVS_NAME_ENV_VAR] == mvs.name
                assert os.environ[_MVS_COMPARTMENT_ENV_VAR] == mvs.compartment_id

        assert _MVS_ID_ENV_VAR not in os.environ
        assert _MVS_NAME_ENV_VAR not in os.environ
        assert _MVS_COMPARTMENT_ENV_VAR not in os.environ

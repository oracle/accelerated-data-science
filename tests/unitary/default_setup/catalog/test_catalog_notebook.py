#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import unittest
from collections import namedtuple
from datetime import datetime, timezone, timedelta
from importlib import reload
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import oci
import pandas as pd
import pytest
from oci.exceptions import ServiceError

import ads.config
import ads.catalog.notebook
from ads.catalog.notebook import NotebookCatalog, NotebookSummaryList
from ads.common import auth, oci_client
from ads.common.utils import random_valid_ocid
from ads.config import PROJECT_OCID


def generate_notebook_list(
    list_notebooks_len=1,
    date_time=datetime(2020, 7, 1, 18, 24, 42, 110000, tzinfo=timezone.utc),
    collision_value="",
    drop_response_value=None,
):
    random_list_notebooks = []

    for i in range(list_notebooks_len):
        formatted_datetime = date_time.isoformat()
        entity_item = {
            "compartment_id": random_valid_ocid(
                prefix="ocid1.compartment.oc1.<unique_ocid>"
            ),
            "created_by": "mock_user",
            "defined_tags": {},
            "display_name": "".join(["sample notebook", str(i)]),
            "freeform_tags": {},
            "id": "".join(
                [
                    random_valid_ocid(prefix="ocid1.notebookcatalog.oc1.<unique_ocid>"),
                    collision_value,
                ]
            ),
            "lifecycle_state": "ACTIVE",
            "notebook_session_configuration_details": "",
            "notebook_session_url": "".join(
                ["http://oci.notebook_session_url", str(i)]
            ),
            "project_id": PROJECT_OCID,
            "time_created": formatted_datetime,
        }

        if drop_response_value is not None:
            del entity_item[drop_response_value]

        random_list_notebooks.append(entity_item)
        date_time += timedelta(minutes=1)

    return random_list_notebooks


class NotebookCatalogTest(unittest.TestCase):
    """Contains test cases for catalog.notebook"""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ[
            "NB_SESSION_COMPARTMENT_OCID"
        ] = "ocid1.compartment.oc1.<unique_ocid>"
        reload(ads.config)
        ads.catalog.notebook.NB_SESSION_COMPARTMENT_OCID = (
            ads.config.NB_SESSION_COMPARTMENT_OCID
        )
        # Initialize class properties after reloading
        with patch.object(auth, "default_signer"):
            with patch.object(oci_client, "OCIClientFactory"):
                cls.notebook_id = "ocid1.notebookcatalog.oc1.iad.<unique_ocid>"
                cls.comp_id = os.environ.get(
                    "NB_SESSION_COMPARTMENT_OCID",
                    "ocid1.compartment.oc1.iad.<unique_ocid>",
                )
                cls.date_time = datetime(
                    2020, 7, 1, 18, 24, 42, 110000, tzinfo=timezone.utc
                )

                cls.notebook_catalog = NotebookCatalog(compartment_id=cls.comp_id)
                cls.notebook_catalog.ds_client = MagicMock()
                cls.notebook_catalog.identity_client = MagicMock()

                cls.nsl = NotebookSummaryList(generate_notebook_list())
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("NB_SESSION_COMPARTMENT_OCID", None)
        reload(ads.config)
        ads.catalog.notebook.NB_SESSION_COMPARTMENT_OCID = (
            ads.config.NB_SESSION_COMPARTMENT_OCID
        )
        return super().tearDownClass()

    @staticmethod
    def generate_notebook_response_data(compartment_id=None, notebook_id=None):
        entity_item = {
            "compartment_id": compartment_id,
            "created_by": "mock_user",
            "defined_tags": {},
            "display_name": "my new notebook catalog",
            "freeform_tags": {},
            "id": notebook_id,
            "lifecycle_state": "ACTIVE",
            "notebook_session_configuration_details": "",
            "notebook_session_url": "oci://notebook_session_url@test_namespace",
            "project_id": PROJECT_OCID,
            "time_created": NotebookCatalogTest.date_time.isoformat(),
        }
        notebook_response = oci.data_science.models.NotebookSession(**entity_item)
        return notebook_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_notebook_init_with_compartment_id(self, mock_client, mock_signer):
        """Test notebook catalog initiation with compartment_id."""
        test_notebook_catalog = NotebookCatalog(compartment_id="9898989898")
        assert test_notebook_catalog.compartment_id == "9898989898"

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_notebook_init_without_compartment_id(self, mock_client, mock_signer):
        """Test notebook catalog initiation without compartment_id."""
        test_notebook_catalog = NotebookCatalog()
        assert (
            test_notebook_catalog.compartment_id
            == ads.config.NB_SESSION_COMPARTMENT_OCID
        )

    def test_decorate_notebook_session_attributes(self):
        """Test NotebookCatalog._decorate_notebook_session method."""
        notebook = self.generate_notebook_response_data(
            compartment_id=self.comp_id, notebook_id=self.notebook_id
        )

        def generate_get_user_data(self, compartment_id=None):
            entity_item = {
                "compartment_id": compartment_id,
                "description": "fake user for testing",
                "email": "FakeUserEmail",
                "id": "ocid1.modeluser.oc1.iad.<unique_ocid>",
                "is_mfa_activated": False,
                "lifecycle_state": "ACTIVE",
                "name": "FakeUser",
                "time_created": self.date_time.isoformat(),
            }
            user_response = oci.identity.models.User(**entity_item)
            return user_response

        wrapper = namedtuple("wrapper", ["data"])
        client_get_user_response = wrapper(
            data=generate_get_user_data(self, compartment_id=self.comp_id)
        )
        self.notebook_catalog.identity_client.get_user = MagicMock(
            return_value=client_get_user_response
        )
        assert "to_dataframe" not in dir(notebook)
        assert "show_in_notebook" not in dir(notebook)
        assert "_repr_html_" not in dir(notebook)

        self.notebook_catalog._decorate_notebook_session(notebook)

        assert notebook.user_name == "FakeUser"
        assert notebook.user_email == "FakeUserEmail"
        assert "to_dataframe" in dir(notebook)
        assert "show_in_notebook" in dir(notebook)
        assert "_repr_html_" in dir(notebook)

        df = notebook.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.loc["id"][0] == self.notebook_id

    def test_get_notebook_session_raise_KeyError(self):
        """Test NotebookCatalog.get_notebook_session method with KeyError raise."""
        self.notebook_catalog.ds_client.get_notebook_session = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.notebook_catalog.get_notebook_session(notebook_id=self.notebook_id)

    def test_get_notebook_session_with_short_id(self):
        """Test NotebookCatalog.get_notebook_session with short_id."""
        # short_id must be 6 letters in lengths
        short_id = "sixlet"
        short_id_index = {short_id: "".join([self.comp_id, short_id])}
        self.notebook_catalog.short_id_index = short_id_index

        def mock_get_notebook_session(notebook_id=id):
            return Mock(
                data=self.generate_notebook_response_data(
                    compartment_id=self.comp_id,
                    notebook_id=short_id_index[short_id],
                )
            )

        self.notebook_catalog.ds_client.get_notebook_session = mock_get_notebook_session
        notebook = self.notebook_catalog.get_notebook_session(short_id)
        assert notebook.id == short_id_index[short_id]

    def test_list_notebook_raise_KeyError(self):
        """Test NotebookCatalog.list_notebook_session with KeyError raise."""
        self.notebook_catalog.ds_client.list_notebook_sessions = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.notebook_catalog.list_notebook_session()

    @mock.patch("builtins.print")
    def test_list_notebook_if_response_is_none(self, mock_print):
        """Test NotebookCatalog.list_notebook_session when response is none."""
        wrapper = namedtuple("wrapper", ["data"])
        client_list_nb_sessions_response_none = wrapper(data=None)
        self.notebook_catalog.ds_client.list_notebook_sessions = MagicMock(
            return_value=client_list_nb_sessions_response_none
        )
        self.notebook_catalog.list_notebook_session()
        mock_print.assert_called_with("No notebooks found.")

        client_list_nb_sessions_response_empty_list = wrapper(data=[])
        self.notebook_catalog.ds_client.list_notebook_sessions = MagicMock(
            return_value=client_list_nb_sessions_response_empty_list
        )
        self.notebook_catalog.list_notebook_session()
        mock_print.assert_called_with("No notebooks found.")

    def test_notebook_catalog_list_notebook_session_two_notebooks(self):
        """Test NotebookCatalog.list_notebook_session response when include_deleted=True and include_deleted=False."""
        wrapper = namedtuple("wrapper", ["data"])
        test_notebooks = [
            {
                "compartment_id": self.comp_id,
                "defined_tags": {},
                "display_name": "This is an active notebook.",
                "freeform_tags": {},
                "id": random_valid_ocid(
                    prefix="ocid1.notebookcatalog.oc1.<unique_ocid>"
                ),
                "lifecycle_state": "ACTIVE",
                "notebook_session_url": "oci://notebook_session_url_1@test_namespace",
                "time_created": "2021-03-01T20:59:39.875000+00:00",
            },
            {
                "compartment_id": self.comp_id,
                "defined_tags": {},
                "display_name": "Deleted notebook",
                "freeform_tags": {},
                "id": random_valid_ocid(
                    prefix="ocid1.notebookcatalog.oc1.<unique_ocid>"
                ),
                "lifecycle_state": "DELETED",
                "notebook_session_url": "oci://notebook_session_url_2@test_namespace",
                "time_created": "2021-03-02T20:59:39.875000+00:00",
            },
        ]

        client_list_nb_sessions_response = wrapper(
            data=[
                oci.data_science.models.NotebookSession(**notebook)
                for notebook in test_notebooks
            ]
        )
        self.notebook_catalog.ds_client.list_notebook_sessions = MagicMock(
            return_value=client_list_nb_sessions_response
        )
        # Test list all notebook session
        notebook_list_all = self.notebook_catalog.list_notebook_session(
            include_deleted=True
        )
        assert notebook_list_all[0] is not None
        assert len(notebook_list_all) == 2
        assert notebook_list_all[0].display_name == "This is an active notebook."

        # Test list only active notebook session
        notebook_list_active = self.notebook_catalog.list_notebook_session(
            include_deleted=False
        )
        assert notebook_list_active[0] is not None
        assert len(notebook_list_active) == 1
        assert notebook_list_active[0].display_name == "This is an active notebook."

    def test_notebook_catalog_create_notebook_session_failed(self):
        """Test NotebookCatalog.create_notebook_session when cannot get response from ds client."""
        notebook_config = {
            "subnet_id": "ocid1.notebookcatalog.oc1.iad.<unique_ocid>",
            "block_storage_size_in_gbs": 50,
            "display_name": "This is a sample notebook.",
            "project_id": "11111",
            "shape": "VM.Standard2.4",
        }
        self.notebook_catalog.ds_client.create_notebook_session = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.notebook_catalog.create_notebook_session(
                display_name=notebook_config["display_name"],
                project_id=notebook_config["project_id"],
                shape=notebook_config["shape"],
                block_storage_size_in_gbs=notebook_config["block_storage_size_in_gbs"],
                subnet_id=notebook_config["subnet_id"],
            )

    def test_update_notebook_session_with_short_id(self):
        """Test NotebookCatalog.update_notebook_session with short_id."""
        # short_id must be 6 letters in lengths
        short_id = "sixlet"
        short_id_index = {short_id: "".join([self.comp_id, short_id])}
        self.notebook_catalog.short_id_index = short_id_index
        wrapper = namedtuple("wrapper", ["data"])
        client_update_notebook_session_response = wrapper(
            data=self.generate_notebook_response_data(
                compartment_id=self.comp_id, notebook_id=short_id_index[short_id]
            )
        )
        self.notebook_catalog.ds_client.update_notebook_session = MagicMock(
            return_value=client_update_notebook_session_response
        )
        notebook = self.notebook_catalog.update_notebook_session(short_id)
        assert notebook.id == short_id_index[short_id]

    def test_delete_notebook_session_failed(self):
        """Test NotebookCatalog.delete_notebook_session when fail."""
        self.notebook_catalog.ds_client.delete_notebook_session = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        assert self.notebook_catalog.delete_notebook_session(self.notebook_id) is False

    def test_delete_notebook_session_succeeded(self):
        """Test NotebookCatalog.delete_notebook_session successfully."""
        self.notebook_catalog.ds_client.delete_notebook_session = Mock(data=None)
        assert self.notebook_catalog.delete_notebook_session(self.notebook_id) is True

    def test_notebook_summary_list_filter_with_lambda(self):
        """Test NotebookSummaryList.filter with selection param is a lambda funtion."""
        res_nsl = self.nsl.filter(
            lambda x: x["lifecycle_state"] == "ACTIVE", instance=None
        )
        assert isinstance(res_nsl, NotebookSummaryList)
        assert len(res_nsl.df.index) == 1

    @mock.patch("builtins.print")
    def test_notebook_summary_list_filter_no_return(self, mock_print):
        """Test NotebookSummaryList.filter with no notebook sessions found."""
        res_nsl_a = self.nsl.filter(
            lambda x: x["lifecycle_state"] == "DELETED", instance=None
        )
        mock_print.assert_called_with("No notebook sessions found")

        res_nsl_b = self.nsl.filter([], instance=None)
        mock_print.assert_called_with("No notebook sessions found")

    def test_notebook_summary_list_filter_invalid_param(self):
        """Test NotebookSummaryList.filter with invalid selection param."""
        # selection is a notebook session instance
        with pytest.raises(ValueError):
            self.nsl.filter(
                selection=self.generate_notebook_response_data(), instance=None
            )

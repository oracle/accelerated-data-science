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
import pytest
from oci.exceptions import ServiceError

import ads.config
import ads.catalog.project
from ads.catalog.project import ProjectCatalog, ProjectSummaryList
from ads.common import auth, oci_client
from ads.common.utils import random_valid_ocid


def generate_project_list(
    list_projects_len=1,
    date_time=datetime(2021, 3, 11, 18, 24, 42, 110000, tzinfo=timezone.utc),
    collision_value="",
    drop_response_value=None,
):
    random_list_projects = []

    for i in range(list_projects_len):
        formatted_datetime = date_time.isoformat()
        entity_item = {
            "compartment_id": random_valid_ocid(
                prefix="ocid1.compartment.oc1.<unique_ocid>"
            ),
            "created_by": "mock_user",
            "defined_tags": {},
            "description": "",
            "display_name": "".join(["sample notebook", str(i)]),
            "freeform_tags": {},
            "id": "".join(
                [
                    random_valid_ocid(prefix="ocid1.notebookcatalog.oc1.<unique_ocid>"),
                    collision_value,
                ]
            ),
            "lifecycle_state": "ACTIVE",
            "time_created": formatted_datetime,
        }

        if drop_response_value is not None:
            del entity_item[drop_response_value]

        random_list_projects.append(entity_item)
        date_time += timedelta(minutes=1)

    return random_list_projects


class ProjectCatalogTest(unittest.TestCase):
    """Contains test cases for catalog.project"""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ[
            "NB_SESSION_COMPARTMENT_OCID"
        ] = "ocid1.compartment.oc1.<unique_ocid>"
        reload(ads.config)
        ads.catalog.project.NB_SESSION_COMPARTMENT_OCID = ads.config.NB_SESSION_COMPARTMENT_OCID
        # Initialize class properties after reloading
        with patch.object(auth, "default_signer"):
            with patch.object(oci_client, "OCIClientFactory"):
                cls.project_id = "ocid1.projectcatalog.oc1.iad.<unique_ocid>"
                cls.comp_id = os.environ.get(
                    "NB_SESSION_COMPARTMENT_OCID",
                    "ocid1.compartment.oc1.iad.<unique_ocid>",
                )
                cls.date_time = datetime(
                    2020, 7, 1, 18, 24, 42, 110000, tzinfo=timezone.utc
                )

                cls.pc = ProjectCatalog(compartment_id=cls.comp_id)
                cls.pc.ds_client = MagicMock()
                cls.pc.identity_client = MagicMock()

                cls.psl = ProjectSummaryList(generate_project_list())
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("NB_SESSION_COMPARTMENT_OCID", None)
        reload(ads.config)
        ads.catalog.project.NB_SESSION_COMPARTMENT_OCID = ads.config.NB_SESSION_COMPARTMENT_OCID
        return super().tearDownClass()

    @staticmethod
    def generate_project_response_data(compartment_id=None, project_id=None):
        entity_item = {
            "compartment_id": compartment_id,
            "created_by": "mock_user",
            "defined_tags": {},
            "description": "",
            "display_name": "Active Project",
            "freeform_tags": {},
            "id": project_id,
            "lifecycle_state": "ACTIVE",
            "time_created": ProjectCatalogTest.date_time.isoformat(),
        }
        project_response = oci.data_science.models.Project(**entity_item)
        return project_response

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_project_init_with_compartment_id(self, mock_client, mock_signer):
        """Test project catalog initiation with compartment_id."""
        test_project_catalog = ProjectCatalog(compartment_id="9898989898")
        assert test_project_catalog.compartment_id == "9898989898"

    @patch("ads.common.auth.default_signer")
    @patch("ads.common.oci_client.OCIClientFactory")
    def test_project_init_without_compartment_id(self, mock_client, mock_signer):
        """Test project catalog initiation without compartment_id."""
        test_project_catalog = ProjectCatalog()
        assert (
            test_project_catalog.compartment_id
            == ads.config.NB_SESSION_COMPARTMENT_OCID
        )

    def test_decorate_project_session_attributes(self):
        """Test ProjectCatalog._decorate_project method."""
        project = self.generate_project_response_data(
            compartment_id=self.comp_id, project_id=self.project_id
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
        self.pc.identity_client.get_user = MagicMock(
            return_value=client_get_user_response
        )
        assert hasattr(project, "to_dataframe") is False
        assert hasattr(project, "show_in_notebook") is False
        assert hasattr(project, "_repr_html_") is False

        self.pc._decorate_project(project)

        assert project.user_name == "FakeUser"
        assert project.user_email == "FakeUserEmail"
        assert hasattr(project, "to_dataframe")
        assert hasattr(project, "show_in_notebook")
        assert hasattr(project, "_repr_html_")

    def test_get_project_raise_KeyError(self):
        """Test ProjectCatalog.get_project method with KeyError raise."""
        self.pc.ds_client.get_project = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.pc.get_project(project_id=self.project_id)

    def test_get_project_with_short_id(self):
        """Test ProjectCatalog.get_project with short_id."""
        # short_id must be 6 letters in lengths
        short_id = "sixlet"
        short_id_index = {short_id: "".join([self.comp_id, short_id])}
        self.pc.short_id_index = short_id_index

        def mock_get_notebook_session(project_id=id):
            return Mock(
                data=self.generate_project_response_data(
                    compartment_id=self.comp_id,
                    project_id=short_id_index[short_id],
                )
            )

        self.pc.ds_client.get_project = mock_get_notebook_session
        project = self.pc.get_project(short_id)
        assert project.id == short_id_index[short_id]

    def test_list_projects_raise_KeyError(self):
        """Test ProjectCatalog.list_projects with KeyError raise."""
        self.pc.ds_client.list_projects = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.pc.list_projects()

    @mock.patch("builtins.print")
    def test_list_notebook_if_response_is_none(self, mock_print):
        """Test ProjectCatalog.list_projects when response is none."""
        wrapper = namedtuple("wrapper", ["data"])
        client_list_projects_response_none = wrapper(data=None)
        self.pc.ds_client.list_notebook_sessions = MagicMock(
            return_value=client_list_projects_response_none
        )
        self.pc.list_projects()
        mock_print.assert_called_with("No project found.")

        client_list_projects_response_empty_list = wrapper(data=[])
        self.pc.ds_client.list_notebook_sessions = MagicMock(
            return_value=client_list_projects_response_empty_list
        )
        self.pc.list_projects()
        mock_print.assert_called_with("No project found.")

    @patch("builtins.print")
    def test_list_projects_two_projects(self, mock_print):
        """Test ProjectCatalog.list_projects response when include_deleted=True and include_deleted=False."""
        wrapper = namedtuple("wrapper", ["data"])
        test_projects = [
            {
                "compartment_id": self.comp_id,
                "created_by": "mock_user",
                "defined_tags": {},
                "description": "",
                "display_name": "Deleted notebook",
                "freeform_tags": {},
                "id": random_valid_ocid(
                    prefix="ocid1.projectcatalog.oc1.<unique_ocid>"
                ),
                "lifecycle_state": "DELETED",
                "time_created": "2021-03-01T20:59:39.875000+00:00",
            },
            {
                "compartment_id": self.comp_id,
                "created_by": "mock_user",
                "defined_tags": {},
                "description": "",
                "display_name": "Deleted notebook",
                "freeform_tags": {},
                "id": random_valid_ocid(
                    prefix="ocid1.projectcatalog.oc1.<unique_ocid>"
                ),
                "lifecycle_state": "DELETED",
                "time_created": "2021-03-02T20:59:39.875000+00:00",
            },
        ]

        client_list_project_response = wrapper(
            data=[
                oci.data_science.models.Project(**project) for project in test_projects
            ]
        )
        self.pc.ds_client.list_projects = MagicMock(
            return_value=client_list_project_response
        )
        # Test list all notebook session
        project_list_all = self.pc.list_projects(include_deleted=True)
        assert project_list_all[0] is not None
        assert len(project_list_all) == 2

        # Test list only active notebook session
        project_list_active = self.pc.list_projects(include_deleted=False)
        mock_print.assert_called_with("No project found.")
        assert len(project_list_active) == 0

    def test_update_project_with_short_id(self):
        """Test ProjectCatalog.update_project with short_id."""
        # short_id must be 6 letters in lengths
        short_id = "sixlet"
        short_id_index = {short_id: "".join([self.comp_id, short_id])}
        self.pc.short_id_index = short_id_index
        wrapper = namedtuple("wrapper", ["data"])
        client_update_project_response = wrapper(
            data=self.generate_project_response_data(
                compartment_id=self.comp_id, project_id=short_id_index[short_id]
            )
        )
        self.pc.ds_client.update_project = MagicMock(
            return_value=client_update_project_response
        )
        project = self.pc.update_project(short_id)
        assert project.id == short_id_index[short_id]

    def test_update_project_with_raise_KeyError(self):
        """Test ProjectCatalog.update_project with KeyError raise."""
        self.pc.ds_client.update_project = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        with pytest.raises(KeyError):
            self.pc.update_project(project_id=self.project_id)

    def test_delete_project_failed(self):
        """Test ProjectCatalog.delete_project when fail."""
        self.pc.ds_client.delete_project = Mock(
            side_effect=ServiceError(
                status=404, code=None, headers={}, message="error test msg"
            )
        )
        assert self.pc.delete_project(self.project_id) is False

    def test_delete_notebook_session_succeeded(self):
        """Test ProjectCatalog.delete_project successfully."""
        self.pc.ds_client.delete_notebook_session = Mock(data=None)
        assert self.pc.delete_project(self.project_id) is True

    def test_project_summary_list_filter_with_lambda(self):
        """Test ProjectSummaryList.filter with selection param is a lambda function."""
        res_psl = self.psl.filter(
            lambda x: x["lifecycle_state"] == "ACTIVE", instance=None
        )
        assert isinstance(res_psl, ProjectSummaryList)
        assert len(res_psl.df.index) == 1

    @mock.patch("builtins.print")
    def test_project_summary_list_filter_no_return(self, mock_print):
        """Test ProjectSummaryList.filter with no notebook sessions found."""
        res_psl_a = self.psl.filter(
            lambda x: x["lifecycle_state"] == "DELETED", instance=None
        )
        mock_print.assert_called_with("No project found")

        res_psl_b = self.psl.filter([], instance=None)
        mock_print.assert_called_with("No project found")

    def test_project_summary_list_filter_invalid_param(self):
        """Test ProjectSummaryList.filter with invalid selection param."""
        # selection is a notebook session instance
        with pytest.raises(ValueError):
            self.psl.filter(
                selection=self.generate_project_response_data(), instance=None
            )

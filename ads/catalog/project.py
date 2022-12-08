#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings

warnings.warn(
    (
        "The `ads.catalog.project` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`."
    ),
    DeprecationWarning,
    stacklevel=2,
)

from ads.catalog.summary import SummaryList
from ads.common import oci_client, auth
from ads.common import utils
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.config import (
    OCI_ODSC_SERVICE_ENDPOINT,
    OCI_IDENTITY_SERVICE_ENDPOINT,
    NB_SESSION_COMPARTMENT_OCID,
)
from collections.abc import Mapping
from oci.config import from_file
from oci.data_science.models import (
    Project,
    ProjectSummary,
    CreateProjectDetails,
    UpdateProjectDetails,
)
from oci.exceptions import ServiceError
from pandas import DataFrame
from types import MethodType


create_project_details_attributes = CreateProjectDetails().swagger_types.keys()
update_project_details_attributes = UpdateProjectDetails().swagger_types.keys()
project_attributes = list(Project().swagger_types.keys())
project_attributes.append("user_name")


class ProjectSummaryList(SummaryList):
    """
    A class used to represent Project Summary List.

    ...

    Attributes
    ----------
    df : data frame
        Summary information for a project.
    datetime_format: str
        Format used to describe time.
    response : oci.response.Response
        A response object with data of type list of ProjectSummaryList.
    short_id_index: (dict of str: str)
        Mapping of short id and its value.

    Methods
    -------
    sort_by(self, columns, reverse=False):
        Sort ProjectSummaryList by columns.

    filter(self, selection, instance=None):
        Filter the project list according to a lambda filter function, or list comprehension.
    """

    def __init__(self, project_list, response=None, datetime_format=utils.date_format):
        super(ProjectSummaryList, self).__init__(
            project_list, datetime_format=datetime_format
        )
        self.response = response

    def __add__(self, rhs):
        return ProjectSummaryList(
            list.__add__(self, rhs), datetime_format=self.datetime_format
        )

    def sort_by(self, columns, reverse=False):
        """Sort ProjectSummaryList by columns.

        Performs a multi-key sort on a particular set of columns and returns the sorted ProjectSummaryList
        Results are listed in a descending order by default.

        Parameters
        ----------
        columns: List of string
          A list of columns which are provided to sort on
        reverse: Boolean (defaults to false)
          If you'd like to reverse the results (for example, to get ascending instead of descending results)

        Returns
        -------
        ProjectSummaryList: A sorted ProjectSummaryList
        """
        return ProjectSummaryList(
            self._sort_by(columns, reverse=reverse),
            datetime_format=self.datetime_format,
        )

    def filter(self, selection, instance=None):
        """Filter the project list according to a lambda filter function, or list comprehension.

        Parameters
        ----------
        selection: lambda function filtering Project instances, or a list-comprehension
            function of list filtering projects
        instance: list, optional
            list to filter, optional, defaults to self

        Returns
        -------
        ProjectSummaryList: A filtered ProjectSummaryList

        Raises
        -------
        ValueError: If selection passed is not correct.
        """
        instance = instance if instance is not None else self

        if callable(selection):
            res = list(filter(selection, instance))
            # lambda filtering
            if len(res) == 0:
                print("No project found")
                return
            return ProjectSummaryList(res)
        elif isinstance(selection, list):
            # list comprehension
            if len(selection) == 0:
                print("No project found")
                return
            return ProjectSummaryList(selection, datetime_format=self.datetime_format)
        else:
            raise ValueError(
                "Filter selection must be a function or a ProjectSummaryList"
            )


class ProjectCatalog(Mapping):
    def __init__(
        self, compartment_id=None, ds_client_auth=None, identity_client_auth=None
    ):
        self.compartment_id = (
            NB_SESSION_COMPARTMENT_OCID if compartment_id is None else compartment_id
        )
        if self.compartment_id is None:
            raise ValueError("compartment_id is required")

        self.ds_client_auth = ds_client_auth or auth.default_signer(
            {"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT}
            if OCI_ODSC_SERVICE_ENDPOINT
            else None
        )
        self.identity_client_auth = identity_client_auth or auth.default_signer(
            {"service_endpoint": OCI_IDENTITY_SERVICE_ENDPOINT}
            if OCI_IDENTITY_SERVICE_ENDPOINT
            else None
        )

        self.ds_client = oci_client.OCIClientFactory(**self.ds_client_auth).data_science
        self.identity_client = oci_client.OCIClientFactory(
            **self.identity_client_auth
        ).identity
        self.short_id_index = {}

    def __getitem__(self, key):
        return self.get_project(key)

    def __iter__(self):
        return self.list_projects().__iter__()

    def __len__(self):
        return len(self.list_projects())

    def _decorate_project(self, project, response=None):
        project.catalog = self
        project.response = response
        project.user_name = ""
        project.swagger_types["user_name"] = "str"
        project.swagger_types["user_email"] = "str"
        project.swagger_types["short_id"] = "str"
        project.ocid = project.id
        try:
            user = self.identity_client.get_user(project.created_by)
            project.user = user.data
            project.user_name = user.data.name
            project.user_email = user.data.email
        except:
            pass

        def commit(project_self, **kwargs):
            update_project_details = UpdateProjectDetails(
                **{
                    key: getattr(project, key)
                    for key in update_project_details_attributes
                }
            )
            return project_self.catalog.update_project(
                project_self.id, update_project_details, **kwargs
            )

        def rollback(project_self):
            """
            Get back the project to remote state

            Returns
            -------
            The project from remote state
            """
            project_self.__dict__.update(self.get_project(project_self.id).__dict__)

        def to_dataframe(project_self):
            df = DataFrame.from_dict(
                {key: getattr(project_self, key) for key in project_attributes},
                orient="index",
                columns=[""],
            )
            return df

        @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
        def show_in_notebook(project_self):
            """
            Describe the project by showing it's properties
            """
            from IPython.core.display import display

            display(project_self)

        def _repr_html_(project_self):
            return (
                project_self.to_dataframe()
                .style.set_properties(**{"margin-left": "0px"})
                .render()
            )

        project.commit = MethodType(commit, project)
        project.rollback = MethodType(rollback, project)
        project.to_dataframe = MethodType(to_dataframe, project)
        project.show_in_notebook = MethodType(show_in_notebook, project)
        project._repr_html_ = MethodType(_repr_html_, project)

        return project

    def list_projects(
        self, include_deleted=False, datetime_format=utils.date_format, **kwargs
    ):
        """
        List all projects in a given compartment, or in the current notebook session's compartment

        Parameters
        ----------
        include_deleted: bool, optional, default=False
            Whether to include deleted projects in the returned list
        datetime_format: str, optional, default: '%Y-%m-%d %H:%M:%S'
            Change format for date time fields

        Returns
        -------
        ProjectSummaryList: List of Projects.

        Raises
        -------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        try:
            list_projects_response = self.ds_client.list_projects(
                self.compartment_id, **kwargs
            )
            if (
                list_projects_response.data is None
                or len(list_projects_response.data) == 0
            ):
                print("No project found.")
                return
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise

        project_list_filtered = [
            self._decorate_project(project)
            for project in list_projects_response.data
            if include_deleted
            or project.lifecycle_state != ProjectSummary.LIFECYCLE_STATE_DELETED
        ]

        # handle empty list
        if project_list_filtered is None or len(project_list_filtered) == 0:
            print("No project found.")
            return []

        psl = ProjectSummaryList(
            project_list_filtered,
            list_projects_response,
            datetime_format=datetime_format,
        )
        self.short_id_index.update(psl.short_id_index)
        return psl

    def create_project(self, create_project_details=None, **kwargs):
        """
        Create a new project with the supplied details.
        create_project_details contains parameters needed to create a new project, according to oci.data_science.models.CreateProjectDetails.

        Parameters
        ----------
        display_name: str
            The value to assign to the display_name property of this CreateProjectDetails.
        description: str
            The value to assign to the description property of this CreateProjectDetails.
        compartment_id: str
            The value to assign to the compartment_id property of this CreateProjectDetails.
        freeform_tags: dict(str, str)
            The value to assign to the freeform_tags property of this CreateProjectDetails.
        defined_tags: dict(str, dict(str, object))
            The value to assign to the defined_tags property of this CreateProjectDetails.
        kwargs:
            New project details can be supplied instead as kwargs

        Returns
        -------
        oci.data_science.models.Project: A new Project record.
        """
        if create_project_details is None:
            create_project_details = CreateProjectDetails(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in create_project_details_attributes
                }
            )
            create_project_details.compartment_id = self.compartment_id
            # filter kwargs removing used keys
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in create_project_details_attributes
            }
        try:
            create_project_response = self.ds_client.create_project(
                create_project_details, **kwargs
            )
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_project(
            create_project_response.data, response=create_project_response
        )

    def get_project(self, project_id):
        """
        Get the Project based on project_id

        Parameters
        ----------
        project_id: str, required
            The OCID of the project to get.

        Returns
        -------
        The oci.data_science.models.Project with the matching ID.

        Raises
        -------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        if not project_id.startswith("ocid"):
            project_id = self.short_id_index[project_id]

        try:
            get_project_response = self.ds_client.get_project(project_id)
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_project(get_project_response.data, get_project_response)

    def update_project(self, project_id, update_project_details=None, **kwargs):
        """
        Updates a project with given project_id, using the provided update data
        update_project_details contains the update project details data to apply, according to oci.data_science.models.UpdateProjectDetails

        Parameters
        ----------
        project_id: str
            project_id OCID to update
        display_name: str
            The value to assign to the display_name property of this UpdateProjectDetails.
        description: str
            The value to assign to the description property of this UpdateProjectDetails.
        freeform_tags: dict(str, str)
            The value to assign to the freeform_tags property of this UpdateProjectDetails.
        defined_tags: dict(str, dict(str, object))
            The value to assign to the defined_tags property of this UpdateProjectDetails.
        kwargs: dict, optional
            Update project details can be supplied instead as kwargs

        Returns
        -------
        oci.data_science.models.Project: The updated Project record
        """
        if not project_id.startswith("ocid"):
            project_id = self.short_id_index[project_id]
        if update_project_details is None:
            update_project_details = UpdateProjectDetails(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in update_project_details_attributes
                }
            )
            update_project_details.compartment_id = self.compartment_id
            # filter kwargs removing used keys
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in update_project_details_attributes
            }
        try:
            update_project_response = self.ds_client.update_project(
                project_id, update_project_details, **kwargs
            )
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_project(
            update_project_response.data, update_project_response
        )

    def delete_project(self, project, **kwargs):
        """
        Deletes the project based on project_id.

        Parameters
        ----------
        project: str ID or oci.data_science.models.Project,required
            The OCID of the project to delete as a string, or a Project instance

        Returns
        -------
        Bool: True if delete was succesful
        """

        try:
            project_id = (
                project.id
                if isinstance(project, Project)
                else self.short_id_index[project]
                if not project.startswith("ocid")
                else project
            )
            self.ds_client.delete_project(project_id, **kwargs)
            return True
        except:
            return False

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import warnings

warnings.warn(
    (
        "The `ads.catalog.notebook` is deprecated in `oracle-ads 2.6.9` and will be removed in `oracle-ads 3.0`."
    ),
    DeprecationWarning,
    stacklevel=2,
)

from pandas import DataFrame
import oci
from oci.data_science.models import (
    NotebookSessionSummary,
    UpdateNotebookSessionDetails,
    CreateNotebookSessionDetails,
    NotebookSession,
    NotebookSessionConfigurationDetails,
)
from oci.exceptions import ServiceError
from types import MethodType

from ads.catalog.summary import SummaryList
from ads.common import utils
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common import auth as authutil
from ads.common import oci_client as oc
from ads.config import (
    OCI_IDENTITY_SERVICE_ENDPOINT,
    NB_SESSION_COMPARTMENT_OCID,
    PROJECT_OCID,
    OCI_ODSC_SERVICE_ENDPOINT,
)

create_notebook_details_attributes = CreateNotebookSessionDetails().swagger_types.keys()
notebook_attributes = list(NotebookSession().swagger_types.keys())
notebook_attributes.append("user_name")
update_notebook_details_attributes = UpdateNotebookSessionDetails().swagger_types.keys()


class NotebookSummaryList(SummaryList):
    def __init__(self, notebook_list, response=None, datetime_format=utils.date_format):
        super(NotebookSummaryList, self).__init__(
            notebook_list, datetime_format=datetime_format
        )
        self.response = response
        self.df["notebook_session_url"] = self.df[
            self.df["lifecycle_state"] == "ACTIVE"
        ]["notebook_session_url"].apply(
            lambda x: "<a href='%s'>%s</a>"
            % (x if x.startswith("http") else "http://%s" % x, "open")
        )

    def __add__(self, rhs):
        return NotebookSummaryList(
            list.__add__(self, rhs), datetime_format=self.datetime_format
        )

    def sort_by(self, columns, reverse=False):
        """
        Performs a multi-key sort on a particular set of columns and returns the sorted NotebookSummaryList
        Results are listed in a descending order by default.

        Parameters
        ----------
        columns: List of string
          A list of columns which are provided to sort on
        reverse: Boolean (defaults to false)
          If you'd like to reverse the results (for example, to get ascending instead of descending results)

        Returns
        -------
        NotebookSummaryList: A sorted NotebookSummaryList
        """
        return NotebookSummaryList(
            self._sort_by(columns, reverse=reverse),
            datetime_format=self.datetime_format,
        )

    def filter(self, selection, instance=None):
        """
        Filter the notebook list according to a lambda filter function, or list comprehension.

        Parameters
        ----------
        selection: lambda function filtering notebook instances, or a list-comprehension
            function of list filtering notebooks
        instance: list, optional
            list to filter, optional, defaults to self

        Raises
        ------
        ValueError: If selection passed is not correct. For example: selection=oci.data_science.models.NotebookSession.
        """
        instance = instance if instance is not None else self

        if callable(selection):
            res = list(filter(selection, instance))
            # lambda filtering
            if len(res) == 0:
                print("No notebook sessions found")
                return
            return NotebookSummaryList(res)
        elif isinstance(selection, list):
            # list comprehension
            if len(selection) == 0:
                print("No notebook sessions found")
                return
            return NotebookSummaryList(selection, datetime_format=self.datetime_format)
        else:
            raise ValueError(
                "Filter selection must be a function or a NotebookSummaryList"
            )


class NotebookCatalog:
    def __init__(self, compartment_id=None):
        self.compartment_id = (
            NB_SESSION_COMPARTMENT_OCID if compartment_id is None else compartment_id
        )
        if self.compartment_id is None:
            raise ValueError("compartment_id needs to be specified.")

        if OCI_ODSC_SERVICE_ENDPOINT:
            odsc_auth = authutil.default_signer(
                client_kwargs={"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT}
            )
        else:
            odsc_auth = authutil.default_signer()
        self.ds_client = oc.OCIClientFactory(**odsc_auth).data_science

        if OCI_IDENTITY_SERVICE_ENDPOINT:
            identity_auth = authutil.default_signer(
                client_kwargs={"service_endpoint": OCI_IDENTITY_SERVICE_ENDPOINT}
            )
        else:
            identity_auth = authutil.default_signer()
        self.identity_client = oc.OCIClientFactory(**identity_auth).identity

        self.short_id_index = {}

    def __getitem__(self, key):
        return self.get_notebook_session(key)

    def __iter__(self):
        return self.list_notebook_session().__iter__()

    def __len__(self):
        return len(self.list_notebook_session())

    def _decorate_notebook_session(self, notebook, response=None):
        notebook.catalog = self
        notebook.response = response
        notebook.user_name = ""
        notebook.swagger_types["user_name"] = "str"
        notebook.swagger_types["user_email"] = "str"
        try:
            user = self.identity_client.get_user(notebook.created_by)
            notebook.user = user.data
            notebook.user_name = user.data.name
            notebook.user_email = user.data.email
        except:
            pass

        def commit(notebook_self, **kwargs):
            # exclude 'notebook_session_configuration_details' key because it's can't be committed/changed
            keys_to_commit = ["display_name", "freeform_tags", "defined_tags"]
            update_notebook_details = UpdateNotebookSessionDetails(
                **{key: getattr(notebook, key) for key in keys_to_commit}
            )
            return self.update_notebook_session(
                notebook_self.id, update_notebook_details, **kwargs
            )

        def rollback(notebook_self):
            """
            Rollback the project to a remote state

            Returns
            -------
            None
            """
            notebook_self.__dict__.update(
                self.get_notebook_session(notebook_self.id).__dict__
            )

        def to_dataframe(notebook_self):
            df = DataFrame.from_dict(
                {
                    key: getattr(notebook_self, key)
                    for key in notebook_attributes
                    if hasattr(notebook_self, key)
                },
                orient="index",
                columns=[""],
            )
            return df

        @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
        def show_in_notebook(notebook_self):
            """
            Describe the project by showing it's properties
            """
            from IPython.core.display import display

            display(notebook_self)

        def _repr_html_(notebook_self):
            df = notebook_self.to_dataframe()
            df.loc["notebook_session_url"] = df.loc["notebook_session_url"].apply(
                lambda x: "<a href='%s'>%s</a>"
                % (x if x.startswith("http") else "http://%s" % x, "open")
            )
            return df.style.set_properties(**{"margin-left": "0px"}).to_html()

        notebook.commit = MethodType(commit, notebook)
        notebook.rollback = MethodType(rollback, notebook)
        notebook.to_dataframe = MethodType(to_dataframe, notebook)
        notebook.show_in_notebook = MethodType(show_in_notebook, notebook)
        notebook._repr_html_ = MethodType(_repr_html_, notebook)

        return notebook

    def list_notebook_session(
        self, include_deleted=False, datetime_format=utils.date_format, **kwargs
    ):
        """
        List all notebooks in a given compartment

        Parameters
        ----------
        include_deleted: bool, optional, default=False
            Whether to include deleted notebooks in the returned list
        datetime_format: str, optional, default: '%Y-%m-%d %H:%M:%S'
            Change format for date time fields

        Returns
        -------
        NotebookSummaryList: A List of notebooks.

        Raises
        ------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        try:
            list_notebook_response = self.ds_client.list_notebook_sessions(
                self.compartment_id, **kwargs
            )
            if (
                list_notebook_response.data is None
                or len(list_notebook_response.data) == 0
            ):
                print("No notebooks found.")
                return
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise

        notebook_list_filtered = [
            self._decorate_notebook_session(notebook)
            for notebook in list_notebook_response.data
            if include_deleted
            or notebook.lifecycle_state
            != NotebookSessionSummary.LIFECYCLE_STATE_DELETED
        ]

        # handle empty list
        if notebook_list_filtered is None or len(notebook_list_filtered) == 0:
            print("No notebooks found.")
            return

        nsl = NotebookSummaryList(
            notebook_list_filtered,
            list_notebook_response,
            datetime_format=datetime_format,
        )
        self.short_id_index.update(nsl.short_id_index)
        return nsl

    def create_notebook_session(
        self,
        display_name=None,
        project_id=None,
        shape=None,
        block_storage_size_in_gbs=None,
        subnet_id=None,
        **kwargs,
    ):
        """
        Create a new notebook session with the supplied details.

        Parameters
        ----------
        display_name: str, required
            The value to assign to the display_name property of this CreateNotebookSessionDetails.
        project_id: str, required
            The value to assign to the project_id property of this CreateNotebookSessionDetails.
        shape: str, required
            The value to assign to the shape property of this NotebookSessionConfigurationDetails.
            Allowed values for this property are: "VM.Standard.E2.2", "VM.Standard.E2.4",
            "VM.Standard.E2.8", "VM.Standard2.1", "VM.Standard2.2", "VM.Standard2.4", "VM.Standard2.8",
            "VM.Standard2.16","VM.Standard2.24".
        block_storage_size_in_gbs: int, required
            Size of the block storage drive. Limited to values between 50 (GB) and 1024 (1024GB = 1TB)
        subnet_id: str, required
            The OCID of the subnet resource where the notebook is to be created.
        kwargs: dict, optional
            Additional kwargs passed to `DataScienceClient.create_notebook_session()`

        Returns
        -------
        oci.data_science.models.NotebookSession: A new notebook record.

        Raises
        ------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        notebook_session_configuration_details = NotebookSessionConfigurationDetails(
            shape=shape,
            block_storage_size_in_gbs=block_storage_size_in_gbs,
            subnet_id=subnet_id,
        )
        project_id = PROJECT_OCID if project_id is None else project_id
        create_notebook_details = CreateNotebookSessionDetails(
            display_name=display_name,
            project_id=project_id,
            compartment_id=self.compartment_id,
            notebook_session_configuration_details=notebook_session_configuration_details,
        )
        try:
            create_notebook_response = self.ds_client.create_notebook_session(
                create_notebook_details, **kwargs
            )
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise

        return self.get_notebook_session(create_notebook_response.data.id)

    def get_notebook_session(self, notebook_id):
        """
        Get the notebook based on notebook_id

        Parameters
        ----------
        notebook_id: str, required
            The OCID of the notebook to get.

        Returns
        -------
        oci.data_science.models.NotebookSession: The oci.data_science.models.NotebookSession with the matching ID.

        Raises
        ------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        if not notebook_id.startswith("ocid"):
            notebook_id = self.short_id_index[notebook_id]

        try:
            get_notebook_response = self.ds_client.get_notebook_session(notebook_id)
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_notebook_session(
            get_notebook_response.data, get_notebook_response
        )

    def update_notebook_session(
        self, notebook_id, update_notebook_details=None, **kwargs
    ):
        """
        Updates a notebook with given notebook_id, using the provided update data

        Parameters
        ----------
        notebook_id: str
            notebook_id OCID to update
        update_notebook_details: oci.data_science.models.UpdateNotebookSessionDetails
            contains the new notebook details data to apply
        kwargs: dict, optional
            Update notebook session details can be supplied instead as kwargs

        Returns
        -------
        oci.data_science.models.NotebookSession: The updated Notebook record

        Raises
        ------
        KeyError: If the resource was not found or do not have authorization to access that resource.
        """
        if not notebook_id.startswith("ocid"):
            notebook_id = self.short_id_index[notebook_id]
        if update_notebook_details is None:
            update_notebook_details = UpdateNotebookSessionDetails(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in update_notebook_details_attributes
                }
            )
            update_notebook_details.compartment_id = self.compartment_id
            # filter kwargs removing used keys
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in update_notebook_details_attributes
            }
        try:
            update_notebook_response = self.ds_client.update_notebook_session(
                notebook_id, update_notebook_details, **kwargs
            )
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_notebook_session(
            update_notebook_response.data, update_notebook_response
        )

    def delete_notebook_session(self, notebook, **kwargs):
        """
        Deletes the notebook based on notebook_id.

        Parameters
        ----------
        notebook: str ID or oci.data_science.models.NotebookSession,required
            The OCID of the notebook to delete as a string, or a Notebook Session instance

        Returns
        -------
        Bool: True if delete was successful, false otherwise
        """
        try:
            notebook_id = (
                notebook.id
                if isinstance(notebook, NotebookSession)
                else self.short_id_index[notebook]
                if not notebook.startswith("ocid")
                else notebook
            )
            self.ds_client.delete_notebook_session(notebook_id, **kwargs)
            return True
        except:
            return False

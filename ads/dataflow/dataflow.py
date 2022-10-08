#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2020, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from datetime import datetime
from types import MethodType
import time
import pathlib
import json
import re
import uuid
from urllib.parse import urlparse
from pathlib import Path
from pandas import DataFrame
from ast import literal_eval

from jinja2 import Environment, PackageLoader

from ads.common import oci_client, auth, logger
from ads.common.utils import FileOverwriteError
from ads.common.decorator.runtime_dependency import (
    runtime_dependency,
    OptionalDependency,
)
from ads.common.decorator.deprecate import deprecated

from oci.data_flow.models import (
    CreateApplicationDetails,
    Application,
    ApplicationParameter,
    ApplicationSummary,
    UpdateApplicationDetails,
    CreateRunDetails,
    Run,
    UpdateRunDetails,
    RunSummary,
)
from oci.exceptions import ServiceError

from ads.common import utils
from ads.dataflow.dataflowsummary import SummaryList
from ads.config import NB_SESSION_COMPARTMENT_OCID, TENANCY_OCID, OCI_REGION_METADATA

env = Environment(loader=PackageLoader("ads", "templates"))

create_application_details_attributes = CreateApplicationDetails().swagger_types.keys()
update_application_details_attributes = UpdateApplicationDetails().swagger_types.keys()
application_attributes = list(Application().swagger_types.keys())

create_run_details_attributes = CreateRunDetails().swagger_types.keys()
update_run_details_attributes = UpdateRunDetails().swagger_types.keys()
run_attributes = list(Run().swagger_types.keys())


class SPARK_VERSION(str):
    v2_4_4 = "2.4.4"
    v3_0_2 = "3.0.2"


class DataFlow:
    @deprecated(
        "2.6.3",
        details="Use ads.jobs.DataFlow class for creating DataFlow applications and runs. Check https://accelerated-data-science.readthedocs.io/en/latest/user_guide/apachespark/dataflow.html#create-run-data-flow-application-using-ads-python-sdk",
    )
    def __init__(
        self,
        compartment_id=None,
        dataflow_base_folder="/home/datascience/dataflow",
        os_auth=None,
        df_auth=None,
    ):
        # create iff not found dataflow_base_folder
        self.dataflow_base_folder = dataflow_base_folder
        self.os_auth = os_auth if os_auth else auth.default_signer()
        self.df_auth = df_auth if df_auth else auth.default_signer()
        self.compartment_id = (
            NB_SESSION_COMPARTMENT_OCID if compartment_id is None else compartment_id
        )
        if self.compartment_id is None:
            raise ValueError("compartment_id needs to be specified.")
        self.display_name = None
        self.driver_shape = None
        self.executor_shape = None
        self.file_uri = None
        self.archive_uri = None
        self.language = None
        self.logs_bucket_uri = None
        self.num_executors = None
        self.spark_version = None
        self.warehouse_bucket_uri = None

        self.object_storage_client = oci_client.OCIClientFactory(
            **self.os_auth
        ).object_storage
        self.df_client = oci_client.OCIClientFactory(**self.df_auth).dataflow

        self.region = (
            self.df_auth["config"]["region"]
            if "config" in self.df_auth and "region" in self.df_auth["config"]
            else (
                literal_eval(OCI_REGION_METADATA)["regionIdentifier"]
                if OCI_REGION_METADATA
                else None
            )
        )
        if not self.region:
            logger.warning(
                "Region information not found from oci config file. Set region in the OCI config file"
            )
        try:
            self.namespace = self.object_storage_client.get_namespace().data
        except ServiceError as se:
            if se.status == 404:
                raise ValueError(
                    f'The compartment_id "{self.compartment_id}" have to be '
                    f"in same tenancy as current user "
                ) from se
            else:
                raise

        self.short_id_index = {}
        # Currently Data Flow only supports VM.Standard.2 series
        # VM_shapes dict needs to be updated if any change from Data Flow
        self.VM_shapes = {
            "VM.Standard2.1",
            "VM.Standard2.2",
            "VM.Standard2.4",
            "VM.Standard2.8",
            "VM.Standard2.16",
            "VM.Standard2.24",  # not available in some tenancy
        }

    def __iter__(self):
        return self.list_apps().__iter__()

    def __len__(self):
        return len(self.list_apps())

    def _decorate_app(self, app):
        app.swagger_types["short_id"] = "str"
        app.ocid = app.id

        def to_dataframe(app_self):
            if "arguments" in application_attributes:
                application_attributes.remove("arguments")
            df = DataFrame.from_dict(
                {
                    key: getattr(app_self, key)
                    for key in application_attributes
                    if hasattr(app_self, key)
                },
                orient="index",
                columns=[""],
            )
            return df

        @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
        def show_in_notebook(app_self):
            """
            Describe the project by showing its properties
            """
            from IPython.display import display

            display(app_self)

        def _repr_html_(app_self):
            return (
                app_self.to_dataframe()
                .style.set_properties(**{"margin-left": "0px"})
                .render()
            )

        app.to_dataframe = MethodType(to_dataframe, app)
        app.show_in_notebook = MethodType(show_in_notebook, app)
        app._repr_html_ = MethodType(_repr_html_, app)

        return app

    def prepare_app(
        self,
        display_name: str,
        script_bucket: str,
        pyspark_file_path: str,
        spark_version: str = SPARK_VERSION.v2_4_4,
        compartment_id: str = None,
        archive_path: str = None,
        archive_bucket: str = None,
        logs_bucket: str = "dataflow-logs",
        driver_shape: str = "VM.Standard2.4",
        executor_shape: str = "VM.Standard2.4",
        num_executors: int = 1,
        arguments: list = [],
        script_parameters: dict = [],
    ) -> dict:
        """
        Check if the parameters provided by users to create an application are
        valid and then prepare app_configuration for creating an app or saving
        for future reuse.

        Parameters
        ----------
        display_name: str, required
            A user-friendly name. This name is not necessarily unique.
        script_bucket: str, required
            bucket in object storage to upload the pyspark file
        pyspark_file_path: str, required
            path to the pyspark file
        spark_version: str
            Allowed values are "2.4.4", "3.0.2".
        compartment_id: str
            OCID of the compartment to create a dataflow app. If not
            provided, compartment_id will use the same as the notebook session.
        archive_path: str, optional
            path to the archive file
        archive_bucket: str, optional
            bucket in object storage to upload the archive file
        logs_bucket: str, default is 'dataflow-logs'
            bucket in object storage to put run logs
        driver_shape: str
            The value to assign to the driver_shape property of this
            CreateApplicationDetails.
            Allowed values for this property are: "VM.Standard2.1",
            "VM.Standard2.2", "VM.Standard2.4", "VM.Standard2.8",
            "VM.Standard2.16", "VM.Standard2.24".
        executor_shape: str
            The value to assign to the executor_shape property of this
            CreateApplicationDetails.
            Allowed values for this property are: "VM.Standard2.1",
            "VM.Standard2.2", "VM.Standard2.4", "VM.Standard2.8",
            "VM.Standard2.16", "VM.Standard2.24".
        num_executors: int
            The number of executor VMs requested.
        arguments: list of str
            The values passed into the command line string to run the application
        script_parameters: dict
            The value of the parameters passed to the running application as
            command line arguments for the pyspark script.

        Returns
        -------
        app_configuration: dictionary containing all the validated params for CreateApplicationDetails.
        """
        if not self._check_bucket_exist(script_bucket):
            raise ValueError(
                "The bucket {} does not exist in object storage".format(script_bucket)
            )
        else:
            self.script_bucket = script_bucket

        if not self._check_bucket_exist(logs_bucket):
            raise ValueError(
                "The log bucket {} does not exist in object storage".format(logs_bucket)
            )
        else:
            self.logs_bucket_uri = (
                self.warehouse_bucket_uri
            ) = f"oci://{logs_bucket}@{self.namespace}"

        # check if local path of script file is valid
        self._check_valid_path(pyspark_file_path)

        if archive_path:
            # check if local path of archive file is valid
            self._check_valid_path(archive_path)

            if archive_bucket is None:
                # use script bucket by default if archive_bucket not provided
                archive_bucket = script_bucket
            else:
                if not self._check_bucket_exist(archive_bucket):
                    raise ValueError(
                        "The bucket {} does not exist in object storage".format(
                            archive_bucket
                        )
                    )

        # check whether the params have valid input type and value
        self._check_valid_param(
            display_name, driver_shape, executor_shape, num_executors
        )

        # when user try to specify a non-python application, we throw warnings
        if self.language is not None and self.language != "PYTHON":
            logger.warning("ADS only supports Python.")

        app_compartment_id = (
            self.compartment_id if compartment_id is None else compartment_id
        )

        app_configuration = {
            "compartment_id": app_compartment_id,
            "language": "PYTHON",
            "pyspark_file_path": pyspark_file_path,
            "script_bucket": self.script_bucket,
            "archive_path": archive_path,
            "archive_bucket": archive_bucket,
            "logs_bucket": logs_bucket,
            "display_name": self.display_name,
            "driver_shape": self.driver_shape,
            "executor_shape": self.executor_shape,
            "num_executors": self.num_executors,
            "spark_version": spark_version,
        }

        # here we handle the case where users specify arguments
        if arguments:
            # check if the arguments are valid
            for arg in arguments:
                if not isinstance(arg, str):
                    raise TypeError("Arguments must be a list of str.")

                if re.match("\$\{([^}]+)\}", arg):
                    arg_name = arg.strip("${}")
                    if " " in arg_name:
                        raise ValueError(
                            f"With {arg} in the format of "
                            "${var}, space is not allowed in "
                            f"{arg_name}"
                        )

                    if arg_name not in script_parameters:
                        logger.warning(
                            f"With `{arg}` in the format of "
                            "`${var}`, "
                            f"the argument `{arg_name}` will be replaced by the value provided in script parameters when passed in. "
                            f"While arguments not in this format are passed to the PySpark script verbatim."
                            f"Therefore, `{arg_name}` must be a valid key in script parameters."
                        )
                        raise KeyError(
                            f"{arg_name} doesn't exist in script parameters, thus {arg} is not valid."
                        )

            # convert script parameters to be a list of tuples
            app_configuration["script_parameters"] = [
                (k, script_parameters[k]) for k in script_parameters
            ]
            app_configuration["arguments"] = arguments

        return app_configuration

    def template(
        self,
        job_type: str = "standard_pyspark",
        script_str: str = "",
        file_dir: str = None,
        file_name: str = None,
    ) -> str:
        """
        Populate a prewritten pyspark or sparksql python script with
        user's choice to write additional lines and save in local directory.

        Parameters
        ----------
        job_type: str, default is 'standard_pyspark'
            Currently supports two types, 'standard_pyspark' or 'sparksql'
        script_str: str, optional, default is ''
            code provided by user to write in the python script
        file_dir: str, optional
            Directory to save the python script in local directory
        file_name: str, optional
            name of the python script to save to the local directory

        Returns
        -------
        script_path: str
            Path to the template generated python file in local directory
        """
        if file_dir is None:
            file_dir = self.dataflow_base_folder
            if not os.path.isdir(file_dir):
                os.mkdir(file_dir)

        if file_name is None:
            creation_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            file_name = f"{job_type}_{creation_time}.py"

        script_path = os.path.join(file_dir, file_name)

        if os.path.exists(script_path):
            logger.info(f"Overwriting {script_path}.")

        if job_type == "standard_pyspark":
            self._get_pyspark_template(script_path, script_str)
        elif job_type == "sparksql":
            self._get_sparksql_template(script_path, script_str)
        else:
            raise ValueError(
                "Currently only supports template for two job types, 'standard_pyspark' or 'sparksql'"
            )
        relative_path = os.path.relpath(script_path)
        # FileLink has to be tested with router to check compatibility. Till then let us comment it (ODSC-8310)
        # return display(FileLink(relative_path))
        logger.info(f"Code generated: {script_path}.")
        return script_path

    def _get_pyspark_template(self, script_path, script_str: str = ""):
        """
        Create a prewriiten pyspark script

        Parameters
        ----------
        script_path: str
            Path to the template generated python file in local directory
        script_str: str, optional, default is ''
            code provided by user to write in the python script

        Returns
        -------
        None
        """
        pyspark_template = env.get_template("dataflow_pyspark.jinja2")

        with open(script_path, "w") as fp:
            fp.write(pyspark_template.render(script_str=script_str))

    def _get_sparksql_template(self, script_path, script_str):
        """
        Create a prewriiten sparksql python script

        Parameters
        ----------
        script_path: str
            Path to the template generated python file in local directory
        script_str: str, optional, default is ''
            code provided by user to write in the python script

        Returns
        -------
        None
        """
        pyspark_template = env.get_template("dataflow_sparksql.jinja2")

        with open(script_path, "w") as fp:
            fp.write(pyspark_template.render(script_str=script_str))

    def _check_valid_path(self, file_path):
        """
        Returns
        -------
        valid_path: bool
            whether the provided file path is valid
        """
        file_dir = os.path.dirname(file_path)
        if len(file_dir) > 0 and not os.path.exists(file_dir):
            raise ValueError("The directoy of file {} does not exist".format(file_path))
        elif not os.path.exists(file_path):
            raise ValueError(
                " The directoy of the file {} is valid but the file does not exist".format(
                    file_path
                )
            )
        return True

    def _check_valid_param(
        self, display_name, driver_shape, executor_shape, num_executors
    ):
        """
        Returns
        -------
        valid_param: bool
            whether the params have valid input type and value
        """
        if not isinstance(display_name, str):
            raise TypeError("param 'display_name' must be string")
        else:
            self.display_name = display_name

        if not isinstance(driver_shape, str):
            raise TypeError("param 'driver_shape' must be string")
        elif driver_shape not in self.VM_shapes:
            raise ValueError("param 'driver_shape' is not a valid VM shape")
        else:
            self.driver_shape = driver_shape

        if not isinstance(executor_shape, str):
            raise TypeError("param 'executor_shape' must be string")
        elif executor_shape not in self.VM_shapes:
            raise ValueError("param 'executor_shape' is not a valid VM shape")
        else:
            self.executor_shape = executor_shape

        if not isinstance(num_executors, int):
            raise TypeError("param 'num_executors' must be an integer")
        elif num_executors < 1:
            raise ValueError("param 'num_executors' has a minimum value of 1")
        else:
            self.num_executors = num_executors
        return True

    def _check_bucket_exist(self, bucket_name: str) -> bool:
        """
        Returns
        -------
        bucket_exist: bool
            whether the bucket already exists in the object storage
        """
        try:
            bucket_response = self.object_storage_client.head_bucket(
                self.namespace, bucket_name
            )
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(
                    f"The bucket {bucket_name} does not exist in object storage"
                ) from se
            else:
                raise
        return True

    def _download(self, bucket_name, script_uri, target_folder):

        remote_pyspark_file_name = urlparse(script_uri).path[1:]

        local_pyspark_file_name = remote_pyspark_file_name
        if "/" in remote_pyspark_file_name:
            local_pyspark_file_name = remote_pyspark_file_name.replace("/", "_")
            if local_pyspark_file_name.endswith(".py"):
                local_pyspark_file_name = (
                    local_pyspark_file_name.rstrip(".py")
                    + "_"
                    + str(uuid.uuid4())[-6:]
                    + ".py"
                )
            else:
                local_pyspark_file_name += "_" + str(uuid.uuid4())[-6:]

        destination_pyspark_file = f"{target_folder}/{local_pyspark_file_name}"

        if not os.path.exists(destination_pyspark_file):

            with open(destination_pyspark_file, "wb") as f:

                f.write(
                    self.object_storage_client.get_object(
                        self.namespace, bucket_name, remote_pyspark_file_name
                    ).data.content
                )

            return destination_pyspark_file

        else:
            raise ValueError(
                f"The app script file ({remote_pyspark_file_name}) already exists in {target_folder}"
            )

    def _create_or_load_app(
        self,
        app_config: dict,
        file_uri: str,
        archive_uri: str = None,
        app_dir: str = None,
    ) -> object:

        with utils.get_progress_bar(2) as progress:
            progress.update()

            #
            # common to load & create
            #

            self.display_name = app_config["display_name"]
            self.driver_shape = app_config["driver_shape"]
            self.executor_shape = app_config["executor_shape"]
            self.num_executors = app_config["num_executors"]
            self.logs_bucket_uri = (
                self.warehouse_bucket_uri
            ) = f"oci://{app_config['logs_bucket']}@{self.namespace}"

            app_details = CreateApplicationDetails(
                compartment_id=app_config["compartment_id"],
                language="PYTHON",
                display_name=self.display_name,
                driver_shape=self.driver_shape,
                executor_shape=self.executor_shape,
                file_uri=file_uri,
                archive_uri=archive_uri,
                logs_bucket_uri=self.logs_bucket_uri,
                num_executors=self.num_executors,
                spark_version=app_config.get("spark_version", SPARK_VERSION.v2_4_4),
                warehouse_bucket_uri=self.warehouse_bucket_uri,
                arguments=app_config.get("arguments", []),
                parameters=[
                    ApplicationParameter(name=k, value=str(v))
                    for (k, v) in app_config.get("script_parameters", [])
                ],
            )

            new_app = self.df_client.create_application(app_details)
            progress.update("Done")
            # # add app_config to app obj as an attribute
            # new_app.data.configuration = app_config

            # make app dir
            if app_dir is None:
                app_dir = f"{self.dataflow_base_folder}/{self.display_name}_{str(uuid.uuid4())[-6:]}"

            pathlib.Path(app_dir).mkdir(parents=True, exist_ok=True)

            return DataFlowApp(
                app_config,
                new_app,
                app_dir,
                f"https://console."
                f"{self.region}.oraclecloud.com/data-flow/apps/details/{new_app.data.id}",
                os_auth=self.os_auth,
                df_auth=self.df_auth,
            )

    def create_app(
        self, app_config: dict, overwrite_script=False, overwrite_archive=False
    ) -> object:
        """
        Create a new dataflow application with the supplied app config.
        app_config contains parameters needed to create a new application,
        according to oci.data_flow.models.CreateApplicationDetails.

        Parameters
        ----------
        app_config: dict
            the config file that contains all necessary parameters used to create a dataflow app
        overwrite_script: bool
            whether to overwrite the existing pyscript script on Object Storage
        overwrite_archive: bool
            whether to overwrite the existing archive file on Object Storage

        Returns
        -------
        df_app: oci.dataflow.models.Application
            New dataflow application.
        """
        #
        # upload pyspark_file_path to OCI object storage
        #
        try:
            self._upload(
                app_config["pyspark_file_path"],
                app_config["script_bucket"],
                overwrite=overwrite_script,
            )
        except FileOverwriteError:
            raise ValueError(
                "You have a file with the same key in your bucket on object storage. Rename your file or set overwrite_script option to True."
            )
        script_name = os.path.basename(app_config["pyspark_file_path"])
        file_uri = f'oci://{app_config["script_bucket"]}@{self.namespace}/{script_name}'

        # upload archive file to object storage if specified
        if app_config["archive_path"] is None:
            return self._create_or_load_app(app_config, file_uri)
        else:
            try:
                self._upload(
                    app_config["archive_path"],
                    app_config["archive_bucket"],
                    overwrite=overwrite_archive,
                )
            except FileOverwriteError:
                raise ValueError(
                    "You have a file with the same key in your bucket on object storage. Rename your file or set overwrite_archive option to True."
                )
            archive_name = os.path.basename(app_config["archive_path"])
            archive_uri = (
                f'oci://{app_config["archive_bucket"]}@{self.namespace}/{archive_name}'
            )
            return self._create_or_load_app(
                app_config, file_uri, archive_uri=archive_uri
            )

    def _upload(self, local_path, bucket_name, overwrite=False):
        """
        upload local files to object storage

        Parameters
        ----------
        local_path: str
            the file path
        bucket_name: str
            bucket name on object storage to upload the file
        overwrite: bool
            whether to overwrite the existing file on Object Storage

        Returns
        -------
        None
        """
        object_name = os.path.basename(local_path)
        if self._check_object_exist(object_name, bucket_name):
            if not overwrite:
                raise FileOverwriteError()
            else:
                logger.warning(
                    "You have a file with the same key in your bucket on object storage. It will be overwritten per your request."
                )

        with open(local_path, "rb") as in_file:
            self.object_storage_client.put_object(
                self.namespace, bucket_name, object_name, in_file
            )
            logger.info(f"Finished uploading `{object_name}`.")

    def _check_object_exist(self, object_name: str, bucket_name: str) -> bool:
        """

        Parameters
        ----------
        object_name: str
            the file name on object storage
        bucket_name: str
            bucket name on object storage

        Returns
        -------
        bool
            whether the file already exists in the bucket in object storage
        """
        object_exist = self._check_object_exist_helper(
            object_name, bucket_name, start=None
        )

        if object_exist:
            logger.info(
                f"The file object `{object_name}` "
                f"already exists in bucket `{bucket_name}`."
            )
        else:
            logger.info(
                f"The file object `{object_name}` "
                f"does not exist in bucket `{bucket_name}` and will be uploaded."
            )
        return object_exist

    def _check_object_exist_helper(
        self, object_name: str, bucket_name: str, start: str = None
    ) -> bool:
        """

        Parameters
        ----------
        object_name: str
            the file name on object storage
        bucket_name: str
            bucket name on object storage
        start: str
            Object names returned by a list query must be greater or equal to this parameter.

        Returns
        -------
        bool
            whether the file already exists in the bucket in object storage
        """
        object_exist = False
        list_objects_response = self.object_storage_client.list_objects(
            self.namespace, bucket_name, start=start
        )

        objects_list = list_objects_response.data.objects
        for object_item in objects_list:
            if object_item.name == object_name:
                object_exist = True

        next_start_with = list_objects_response.data.next_start_with

        if object_exist or not next_start_with:
            return object_exist
        else:
            return self._check_object_exist_helper(
                object_name, bucket_name, start=next_start_with
            )

    def list_apps(
        self,
        include_deleted: bool = False,
        compartment_id: str = None,
        datetime_format: str = utils.date_format,
        **kwargs,
    ) -> object:
        """
        List all apps in a given compartment, or in the current notebook session's compartment.

        Parameters
        ----------
        include_deleted: bool, optional, default=False
            Whether to include deleted apps in the returned list.
        compartment_id: str, optional, default: NB_SESSION_COMPARTMENT_OCID
            The compartment specified to list apps.
        datetime_format: str, optional, default: '%Y-%m-%d %H:%M:%S'
            Change format for date time fields.

        Returns
        -------
        dsl: List
            List of Dataflow applications.
        """
        app_compartment_id = (
            self.compartment_id if compartment_id is None else compartment_id
        )
        list_applications_response = self.df_client.list_applications(
            app_compartment_id, **kwargs
        ).data

        # handle empty list
        if list_applications_response is None:
            logger.warning("No applications found.")
            return

        application_list_filtered = [
            self._decorate_app(app)
            for app in list_applications_response
            if include_deleted
            or Application.lifecycle_state != ApplicationSummary.LIFECYCLE_STATE_DELETED
        ]

        dsl = SummaryList(
            entity_list=application_list_filtered,
            datetime_format=datetime_format,
        )
        self.short_id_index.update(dsl.short_id_index)
        return dsl

    def get_app(self, app_id: str):
        """
        Get the Project based on app_id.

        Parameters
        ----------
        app_id: str, required
            The OCID of the dataflow app to get.

        Returns
        -------
        app: oci.dataflow.models.Application
            The oci.dataflow.models.Application with the matching ID.
        """
        if not app_id.startswith("ocid"):
            app_id = self.short_id_index[app_id]

        try:
            get_app_response = self.df_client.get_application(app_id)
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_app(get_app_response.data)

    def load_app(
        self,
        app_id: str,
        target_folder: str = None,
    ) -> object:
        """
        Load an existing dataflow application based on application id.
        The existing dataflow application can be created either from dataflow
        service or the dataflow integration of ADS.

        Parameters
        ----------
        app_id: str, required
            The OCID of the dataflow app to load.

        target_folder: str, optional,
            the folder to store the local artifacts of this application.
            If not specified, the target_folder will use the
            dataflow_base_folder by default.

        Returns
        -------
        dfa: ads.dataflow.dataflow.DataFlowApp
            A dataflow application of type ads.dataflow.dataflow.DataFlowApp
        """

        # support short id when loading an application by getting ocid based on
        # provided short id
        if not app_id.startswith("ocid"):
            app_id = self.short_id_index[app_id]

        # get app response that fetched using df client
        try:
            get_app_response = self.df_client.get_application(app_id)
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise

        # if users try to load a non-python application, we throw a warning
        if get_app_response.data.language != "PYTHON":
            logger.warning("ADS only supports Python.")
            return

        # for apps created with default logs bucket, logs_bucket_uri attribute in app response will be empty string
        # set default value manually for logs_bucket
        if get_app_response.data.logs_bucket_uri == "":
            logger.info(
                "Using the default logs bucket 'dataflow-logs'. Set the parameter `logs_bucket_uri` to use a different bucket."
            )
            logs_bucket = "dataflow-logs"
        else:
            logs_bucket = re.split(r"[@/]", get_app_response.data.logs_bucket_uri)[2]

        # reform app config from app response
        app_config = {
            "compartment_id": get_app_response.data.compartment_id,
            "language": get_app_response.data.language,
            "script_bucket": re.split(r"[@/]", get_app_response.data.file_uri)[2],
            "logs_bucket": logs_bucket,
            "archive_path": None,
            "archive_bucket": None,
            "display_name": get_app_response.data.display_name,
            "driver_shape": get_app_response.data.driver_shape,
            "executor_shape": get_app_response.data.executor_shape,
            "num_executors": get_app_response.data.num_executors,
            "spark_version": get_app_response.data.spark_version,
            "arguments": get_app_response.data.arguments,
            "script_parameters": [
                (param.name, param.value) for param in get_app_response.data.parameters
            ],
        }

        # set the default value to target_folder to dataflow_base_folder
        if target_folder is None:
            target_folder = self.dataflow_base_folder

        app_dir = (
            f"{target_folder}/{app_config['display_name']}_{str(uuid.uuid4())[-6:]}"
        )
        pathlib.Path(app_dir).mkdir(parents=True, exist_ok=True)

        app_config["pyspark_file_path"] = self._download(
            app_config["script_bucket"], get_app_response.data.file_uri, app_dir
        )

        if get_app_response.data.archive_uri != "":
            app_config["archive_bucket"] = re.split(
                r"[@/]", get_app_response.data.archive_uri
            )[2]
            app_config["archive_path"] = self._download(
                app_config["archive_bucket"], get_app_response.data.archive_uri, app_dir
            )

        return DataFlowApp(
            app_config,
            get_app_response,
            app_dir,
            f"https://console.{self.region}.oraclecloud.com/data-flow/apps/details/{get_app_response.data.id}",
            os_auth=self.os_auth,
            df_auth=self.df_auth,
        )


class DataFlowApp(DataFlow):
    @deprecated("2.6.3")
    def __init__(self, app_config, app_response, app_dir, oci_link, **kwargs):
        super().__init__(compartment_id=app_config["compartment_id"], **kwargs)
        self._config = app_config
        self.app_response = app_response
        self.app_dir = app_dir
        self._oci_link = oci_link

    def __iter__(self):
        return self.list_runs().__iter__()

    def __len__(self):
        return len(self.list_runs())

    def __repr__(self):
        return self._config["display_name"]

    def _decorate_run(self, run):
        run.swagger_types["short_id"] = "str"
        run.ocid = run.id

        def to_dataframe(run_self):
            if "arguments" in run_attributes:
                run_attributes.remove("arguments")
            df = DataFrame.from_dict(
                {
                    key: getattr(run_self, key)
                    for key in run_attributes
                    if hasattr(run_self, key)
                },
                orient="index",
                columns=[""],
            )
            return df

        @runtime_dependency(module="IPython", install_from=OptionalDependency.NOTEBOOK)
        def show_in_notebook(run_self):
            """
            Describe the project by showing it's properties
            """
            from IPython.display import display

            display(run_self)

        def _repr_html_(run_self):
            return (
                run_self.to_dataframe()
                .style.set_properties(**{"margin-left": "0px"})
                .render()
            )

        run.to_dataframe = MethodType(to_dataframe, run)
        run.show_in_notebook = MethodType(show_in_notebook, run)
        run._repr_html_ = MethodType(_repr_html_, run)

        return run

    @property
    def config(self) -> dict:
        """
        Retrieve the app_config file used to create the data flow app

        Returns
        -------
        app_config: Dict
            dictionary containing all the validated params for this DataFlowApp
        """
        return self._config

    @property
    def oci_link(self) -> object:
        """
        Retrieve the oci link of the data flow app

        Returns
        -------
        oci_link: str
            a link to the app page in an oci console.
        """
        return self._oci_link

    def prepare_run(
        self,
        run_display_name: str,
        compartment_id: str = None,
        logs_bucket: str = "",
        driver_shape: str = "VM.Standard2.4",
        executor_shape: str = "VM.Standard2.4",
        num_executors: int = 1,
        **kwargs,
    ) -> dict:
        """
        Check if the parameters provided by users to create a run are
        valid and then prepare run_config for creating run details.

        Parameters
        ----------
        run_display_name: str
            A user-friendly name. This name is not necessarily unique.
        compartment_id: str
            OCID of the compartment to create a dataflow run. If not
            provided, compartment_id will use the same as the dataflow app.
        logs_bucket: str
            bucket in object storage to put run logs, if not provided,
            will use the same logs_bucket as defined in app_config
        driver_shape: str
            The value to assign to the driver_shape property of this
            CreateApplicationDetails.
            Allowed values for this property are: "VM.Standard2.1",
            "VM.Standard2.2", "VM.Standard2.4", "VM.Standard2.8",
            "VM.Standard2.16", "VM.Standard2.24".
        executor_shape: str
            The value to assign to the executor_shape property of this
            CreateApplicationDetails.
            Allowed values for this property are: "VM.Standard2.1",
            "VM.Standard2.2", "VM.Standard2.4", "VM.Standard2.8",
            "VM.Standard2.16", "VM.Standard2.24".
        num_executors: int
            The number of executor VMs requested.

        Returns
        -------
        run_config: Dict
            Dictionary containing all the validated params for CreateRunDetails.
        """
        # if logs_bucket not provided to prepare run_config, will use the
        # same logs_bucket as defined in app_config
        if logs_bucket == "":
            logs_bucket = self.config["logs_bucket"]
        if not self._check_bucket_exist(logs_bucket):
            raise ValueError(
                "The log bucket {} does not exist in object storage".format(logs_bucket)
            )
        else:
            logs_bucket_uri = f"oci://{logs_bucket}@{self.namespace}"

        if not isinstance(run_display_name, str):
            raise TypeError("param 'run_display_name' must be string")

        if not isinstance(driver_shape, str):
            raise TypeError("param 'driver_shape' must be string")
        elif driver_shape not in self.VM_shapes:
            raise ValueError("param 'driver_shape' is not a valid VM shape")

        if not isinstance(executor_shape, str):
            raise TypeError("param 'executor_shape' must be string")
        elif executor_shape not in self.VM_shapes:
            raise ValueError("param 'executor_shape' is not a valid VM shape")

        if not isinstance(num_executors, int):
            raise TypeError("param 'num_executors' must be integer")
        elif num_executors < 1:
            raise ValueError("param 'num_executors' has a minimum value of 1")

        run_compartment_id = (
            self.compartment_id if compartment_id is None else compartment_id
        )

        run_configuration = {
            "compartment_id": run_compartment_id,
            "script_bucket": self._config["script_bucket"],
            "pyspark_file_path": self._config["pyspark_file_path"],
            "archive_path": self._config["archive_path"],
            "archive_bucket": self._config["archive_bucket"],
            "run_display_name": run_display_name,
            "logs_bucket": logs_bucket,
            "logs_bucket_uri": logs_bucket_uri,
            "driver_shape": driver_shape,
            "executor_shape": executor_shape,
            "num_executors": num_executors,
        }

        # get arguments from app config
        if "arguments" in self._config:
            run_configuration["arguments"] = list(self._config["arguments"])

        # get script parameters in app config
        if "script_parameters" in self._config:
            run_configuration["script_parameters"] = list(
                self._config["script_parameters"]
            )

        # update the new value of the argument in the run config when user provides new name value pairs
        if kwargs:
            # convert script parameters into dict
            param_dict = dict(run_configuration["script_parameters"])
            for param, value in kwargs.items():
                # overwrite the value
                if param in param_dict:
                    param_dict[param] = value
                else:
                    raise KeyError(f"{param} is not a valid key in script parameters")

            # convert param_dict back to list of tuples
            run_configuration["script_parameters"] = [
                (k, param_dict[k]) for k in param_dict
            ]

        return run_configuration

    def run(
        self,
        run_config: dict,
        save_log_to_local: bool = False,
        copy_script_to_object_storage: bool = True,
        copy_archive_to_object_storage: bool = True,
        pyspark_file_path: str = None,
        archive_path: str = None,
        wait: bool = True,
    ) -> object:
        """
        Create a new dataflow run with the supplied run config.
        run_config contains parameters needed to create a new run, according to oci.data_flow.models.CreateRunDetails.

        Parameters
        ----------
        run_config: dict, required
            The config file that contains all necessary parameters used to create a dataflow run
        save_log_to_local: bool, optional
            A boolean value that defaults to false. If set to true, it saves the log files to local dir
        copy_script_to_object_storage: bool, optional
            A boolean value that defaults to true. Local script will be copied to object storage
        copy_archive_to_object_storage: bool, optional
            A boolean value that defaults to true. Local archive file will be copied to object storage
        pyspark_file_path: str, optional
            The pyspark file path used for creating the dataflow app.
            if pyspark_file_path isn't specified then reuse the path that the app was created with.
        archive_path: str, optional
            The archive file path used for creating the dataflow app.
            if archive_path isn't specified then reuse the path that the app was created with.
        wait: bool, optional
            A boolean value that defaults to true.
            When True, the return will be ads.dataflow.dataflow.DataFlowRun in terminal state.
            When False, the return will be a ads.dataflow.dataflow.RunObserver.

        Returns
        -------
        df_run: Variable
            Either a new Data Flow run or a run observer.
        """

        if copy_script_to_object_storage:
            self._sync(pyspark_file_path)

        if run_config["archive_path"] is not None:
            if copy_archive_to_object_storage:
                self._sync(archive_path, type="archive")

        run_observer = RunObserver(self, run_config, save_log_to_local)

        if wait:
            return run_observer.wait()  # blocks, returns DataFlowRun
        else:
            return run_observer  # unblocks, returns RunObserver

    def _sync(self, file_path: str = None, type: str = "script") -> object:
        """
        Push to create a new app if the script has been modified.

        Parameters
        ----------
        file_path: str
            The pyspark file path used for creating the dataflow app.
            if pyspark_file_path isn't specified then reuse the path that the app was created with.
        type: str, only two types supported here, 'script' or 'archive'

        Returns
        -------
        Self, a Data Flow app object.
        """

        # local pyspark file, which may or may not be modified
        if type == "script":
            file_path = (
                self.config["pyspark_file_path"] if file_path is None else file_path
            )

            # compare byte content of two files
            # script no diff, return the original app obj
            if not self._modified(file_path, type="script"):
                return self

            os_bucket = self.config["script_bucket"]
            os_objectname = self.config["pyspark_file_path"].rsplit("/")[-1]

        elif type == "archive":
            file_path = self.config["archive_path"] if file_path is None else file_path
            if not self._modified(file_path, type="archive"):
                return self

            os_bucket = self.config["archive_bucket"]
            os_objectname = self.config["archive_path"].rsplit("/")[-1]

        # push file to object storage
        with open(file_path, "rb") as in_file:
            try:
                self.object_storage_client.put_object(
                    self.namespace,
                    os_bucket,
                    os_objectname,
                    in_file,
                )
                logger.info(
                    f"The existing file `{os_objectname}` in bucket "
                    f"`{os_bucket}` on object storage has been overwritten by your latest changes of "
                    f"`{os_objectname}`."
                )
            except ServiceError as se:
                if se.status == 404:
                    raise KeyError(se.message) from se
                else:
                    raise
        return self

    def _modified(
        self, file_path: str = None, type: str = "script", encoding="utf8"
    ) -> bool:
        """
        Check if any modification in the pyspark script

        Returns
        -------
        True or False
        """
        # read local python file into bytes
        if type == "script":
            bucket_name = self.config["script_bucket"]
            # remote pyspark file, which was pushed originally while creating this dataflow app
            remote_file_name = Path(self.config["pyspark_file_path"]).name.lstrip("/")
        elif type == "archive":
            bucket_name = self.config["archive_bucket"]
            remote_file_name = Path(self.config["archive_path"]).name.lstrip("/")
        local_file = open(file_path, "rb").read()

        # remote_file_name = Path(self.config["pyspark_file_path"]).name.lstrip('/')

        try:
            remote_file = self.object_storage_client.get_object(
                self.namespace, bucket_name, remote_file_name
            ).data.content
        except ServiceError as se:
            if se.code == "ObjectNotFound":
                local_filename = os.path.basename(file_path)
                logger.info(
                    f"The `{local_filename}` is "
                    f"not found in your bucket. "
                    f"The `{local_filename}` will be uploaded"
                )

                with open(file_path, "rb") as in_file:
                    self.object_storage_client.put_object(
                        self.namespace, bucket_name, local_filename, in_file
                    )

                return False
            else:
                raise se

        if type == "archive":
            return local_file != remote_file

        elif type == "script":
            if isinstance(local_file, bytes):
                local_file = local_file.decode(encoding)
            if isinstance(remote_file, bytes):
                remote_file = remote_file.decode(encoding)
            return local_file != remote_file

    def list_runs(
        self,
        include_failed: bool = False,
        datetime_format: str = utils.date_format,
        **kwargs,
    ) -> object:
        """
        List all run of a dataflow app

        Parameters
        ----------
        include_failed: bool, optional, default=False
            Whether to include failed runs in the returned list
        datetime_format: str, optional, default: '%Y-%m-%d %H:%M:%S'
            Change format for date time fields

        Returns
        -------
        df_runs: List
            List of Data flow runs.
        """

        list_runs_response = self.df_client.list_runs(
            self.compartment_id, **kwargs
        ).data

        # handle empty list
        if list_runs_response is None:
            logger.warning("No runs found.")
            return

        run_list_filtered = [
            self._decorate_run(run)
            for run in list_runs_response
            if include_failed
            or Run.lifecycle_state != RunSummary.LIFECYCLE_STATE_FAILED
        ]

        rsl = SummaryList(
            entity_list=run_list_filtered, datetime_format=datetime_format
        )
        self.short_id_index.update(rsl.short_id_index)
        return rsl

    def get_run(self, run_id: str):
        """
        Get the Run based on run_id

        Parameters
        ----------
        run_id: str, required
            The OCID of the dataflow run to get.

        Returns
        -------
        df_run: oci.dataflow.models.Run
            The oci.dataflow.models.Run with the matching ID.
        """
        if not run_id.startswith("ocid"):
            run_id = self.short_id_index[run_id]

        try:
            get_run_response = self.df_client.get_run(run_id)
        except ServiceError as se:
            if se.status == 404:
                raise KeyError(se.message) from se
            else:
                raise
        return self._decorate_run(get_run_response.data)


class RunObserver:
    @deprecated("2.6.3")
    def __init__(self, app, run_config, save_log_to_local):
        self.app = app
        self._config = run_config
        self.save_log_to_local = save_log_to_local

        self._local_dir = self._create_run_dir()
        self._run_details = self._get_run_details()
        self._new_run = self.app.df_client.create_run(self._run_details)
        self._oci_link = f"https://console.{self.app.region}.oraclecloud.com/data-flow/runs/details/{self._new_run.data.id}"
        self._save_log_to_local = save_log_to_local

    def __repr__(self):
        return self._config["run_display_name"] + " " + self.status

    def wait(self):
        """
        Wait and monitor the run creation process.

        Parameters
        ----------
        None

        Returns
        -------
        df_run: oci.dataflow.models.Run
            The oci.dataflow.models.Run after monitoring is done.
        """
        # monitor the run creation process
        self._monitor_run()
        self._create_run_dir()

        return DataFlowRun(
            self._config,
            self._new_run,
            self.save_log_to_local,
            self._local_dir,
            os_auth=self.app.os_auth,
            df_auth=self.app.df_auth,
        )

    def _terminal_state(self, status):
        return status in ["SUCCEEDED", "FAILED", "CANCELED"]

    # this is a blocking function, it will only complete when dataflow run reaches terminal state
    def _monitor_run(self):
        curr_status = self.status
        if not self._terminal_state(curr_status):

            # when wait is called after the run being submitted, progress bar does not start from the beginning
            if curr_status == "ACCEPTED":
                progress_bar_num = 3
            elif curr_status == "IN_PROGRESS":
                progress_bar_num = 2
            elif curr_status == "SUCCEEDED":
                progress_bar_num = 1
            else:
                progress_bar_num = 4

            with utils.get_progress_bar(progress_bar_num) as progress:
                progress.update()
                while not self._terminal_state(curr_status):
                    time.sleep(2)
                    new_status = self.status
                    if new_status != curr_status:
                        progress.update(f"{new_status}")
                        curr_status = new_status
                progress.update("Done")

    def _create_run_dir(self):

        creation_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        local_dir = (
            f"{self.app.app_dir}/{self._config['run_display_name']}-{creation_time}"
        )
        pathlib.Path(local_dir).mkdir(parents=True, exist_ok=True)

        # add pyspark script for the run in the run dir
        user_pyspark_file_path = self._config["pyspark_file_path"]
        filename = os.path.basename(user_pyspark_file_path)
        run_pyspark_file_path = os.path.join(local_dir, filename)

        # copy content
        from shutil import copyfile

        copyfile(user_pyspark_file_path, run_pyspark_file_path)

        # add run metadata for the run in the run dir
        run_config_path = os.path.join(local_dir, "run_metadata.json")
        with open(run_config_path, "w") as fp:
            json.dump(self._config, fp)

        return local_dir

    def _get_run_details(self):
        self.run_display_name = self._config["run_display_name"]
        self.logs_bucket_uri = self._config["logs_bucket_uri"]
        self.driver_shape = self._config["driver_shape"]
        self.executor_shape = self._config["executor_shape"]
        self.num_executors = self._config["num_executors"]

        run_details = CreateRunDetails(
            application_id=self.app.app_response.data.id,
            compartment_id=self._config["compartment_id"],
            display_name=self.run_display_name,
            logs_bucket_uri=self.logs_bucket_uri,
            driver_shape=self.driver_shape,
            executor_shape=self.executor_shape,
            num_executors=self.num_executors,
            arguments=self._config.get("arguments", []),
            parameters=[
                ApplicationParameter(name=k, value=str(v))
                for (k, v) in self._config.get("script_parameters", [])
            ],
        )
        return run_details

    @property
    def status(self) -> str:
        """
        Returns the lifecycle state of the Data Flow run
        """
        return self.app.df_client.get_run(self._new_run.data.id).data.lifecycle_state

    @property
    def config(self) -> dict:
        """
        Retrieve the run_config file used to create the data flow run

        Returns
        -------
        run_config: Dict
            Dictionary containing all the validated parameters for this Data Flow run
        """
        return self._config

    def update_config(self, param_dict) -> None:
        """
        Modify the run_config file used to create the data flow run

        Parameters
        ----------
        param_dict: Dict
            dictionary containing the key value pairs of the run_config parameters and
            the updated values.

        Returns
        -------
        None
        """
        for key, val in param_dict.items():
            if key in self._config:
                self._config[key] = val
            else:
                raise ValueError(
                    "The key '{}' does not exist in run_config parameters".format(key)
                )

    @property
    def local_dir(self) -> str:
        """
        Retrieve the local directory of the data flow run

        Returns
        -------
        local_dir: str
            the local path to the Data Flow run
        """
        return self._local_dir

    @property
    def oci_link(self) -> object:
        """
        Retrieve the oci link of the data flow run

        Returns
        -------
        oci_link: str
            link to the run page in an oci console
        """
        return self._oci_link


class DataFlowRun(DataFlow):
    LOG_OUTPUTS = ["stdout", "stderr"]

    @deprecated("2.6.3")
    def __init__(
        self, run_config, run_response, save_log_to_local, local_dir, **kwargs
    ):
        super().__init__(compartment_id=run_config["compartment_id"], **kwargs)
        self._config = run_config
        self.run_response = run_response
        self.run_id = self.run_response.data.id
        self._log_stdout = None
        self._log_stderr = None
        self._local_dir = local_dir
        self._status = self.df_client.get_run(self.run_id).data.lifecycle_state
        self._oci_link = None
        if save_log_to_local:
            for log_type in DataFlowRun.LOG_OUTPUTS:
                self.fetch_log(log_type).save()

    def __repr__(self):
        return self._config["run_display_name"]

    @property
    def config(self) -> dict:
        """
        Retrieve the run_config file used to create the Data Flow run

        Returns
        -------
        run_config: Dict
            dictionary containing all the validated params for this DataFlowRun
        """
        return self._config

    def update_config(self, param_dict) -> None:
        """
        Modify the run_config file used to create the data flow run

        Parameters
        ----------
        param_dict: Dict
            Dictionary containing the key value pairs of the run_config parameters and
            the updated values.

        Returns
        -------
        None
        """
        for key, val in param_dict.items():
            if key in self._config:
                self._config[key] = val
            else:
                raise ValueError(
                    "The key '{}' does not exist in run_config parameters".format(key)
                )

    @property
    def status(self) -> str:
        """
        Retrieve the status of the data flow run

        Returns
        -------
        status: str
            String that describes the status of the run
        """
        return self._status

    @property
    def log_stdout(self) -> object:
        """
        Retrieve the stdout of the data flow run

        Returns
        -------
        log_out: ads.dataflow.dataflow.DataFlowLog
            a clickable link that opens the stdout log in another tab in a JupyterLab notebook environment
        """
        if self._log_stdout is None:
            self._log_stdout = self.fetch_log("stdout")
        return self._log_stdout

    @property
    def log_stderr(self) -> object:
        """
        Retrieve the stderr of the data flow run

        Returns
        -------
        log_error: ads.dataflow.dataflow.DataFlowLog
            a clickable link that opens the stderror log in another tab in jupyter notebook environment
        """
        if self._log_stderr is None:
            self._log_stderr = self.fetch_log("stderr")
        return self._log_stderr

    @property
    def local_dir(self) -> str:
        """
        Retrieve the local directory of the data flow run

        Returns
        -------
        local_dir: str
            the local path to the Data Flow run
        """
        return self._local_dir

    @property
    def oci_link(self) -> object:
        """
        Retrieve the oci link of the data flow run

        Returns
        -------
        oci_link: str
            link to the run page in an oci console
        """
        return self._oci_link

    def fetch_log(self, log_type: str) -> object:
        """
        Fetch the log information of a run

        Parameters
        ----------
        log_type: str, have two values, 'stdout' or 'stderr'

        Returns
        -------
        dfl: DataFlowLog
            a Data Flow log object
        """
        if log_type not in DataFlowRun.LOG_OUTPUTS:
            raise ValueError(
                f"Invalid log type ({log_type}), valid types: {', '.join(DataFlowRun.LOG_OUTPUTS)}"
            )

        tmp = self.df_client.get_run_log(
            self.run_id, f"spark_application_{log_type}.log.gz"
        )
        text_str = str(tmp.data.text.lstrip("\x00").rstrip("\n"))

        opc_request_id = self.run_response.data.opc_request_id

        log_filename = os.path.join(
            opc_request_id, f"spark_application_{log_type}.log.gz"
        )
        object_storage_log_path = os.path.join(
            self._config["logs_bucket_uri"], log_filename
        )
        log_local_dir = os.path.join(self._local_dir, "logs")

        log_obj = DataFlowLog(text_str, object_storage_log_path, log_local_dir)
        if log_type == "stdout":
            self._log_stdout = log_obj
        elif log_type == "stderr":
            self._log_stderr = log_obj

        return log_obj

    def _repr_html_(self):
        if self.status == "FAILED":
            logger.warning("Run failed. See the logs for details.")
            logger.info("Printing tail of the stdout log...")
            self.log_stdout.tail()
        else:
            return


class DataFlowLog:
    @deprecated("2.6.3")
    def __init__(self, text, oci_path, log_local_dir):
        self.text = str(text)
        self._oci_path = oci_path
        self._local_dir = log_local_dir
        self._local_path = None
        self.line_list = self.text.split("\n")

    def head(self, n: int = 10):
        """
        Show the first n lines of the log as the output of the notebook cell

        Parameters
        ----------
        n: int, default is 10
            the number of lines from head of the log file

        Returns
        -------
        None
        """
        for _, v in enumerate(self.line_list[:n]):
            print(v)

    def tail(self, n: int = 10):
        """
        Show the last n lines of the log as the output of the notebook cell

        Parameters
        ----------
        n: int, default is 10
            the number of lines from tail of the log file

        Returns
        -------
        None
        """
        for _, v in enumerate(self.line_list[-n:]):
            print(v)

    def show_all(self):
        """
        Show all content of the log as the output of the notebook cell

        Returns
        -------
        None
        """
        for _, v in enumerate(self.line_list):
            print(v)

    @property
    def oci_path(self):
        """
        Get the path of the log file in object storage

        Returns
        -------
        oci_path: str
            Path of the log file in object storage
        """
        return self._oci_path

    @property
    def local_path(self):
        """
        Get the path of the log file in local directory

        Returns
        -------
        local_path: str
            Path of the log file in local directory
        """
        if self._local_path is None:
            logger.warning(
                "The log file is not stored in local directory. "
                "Call the save() method to save the log file to local first.",
            )

        return self._local_path

    @property
    def local_dir(self):
        """
        Get the local directory where the log file is saved.

        Returns
        -------
        local_dir: str
            Path to the local directory where the log file is saved.
        """
        return self._local_dir

    def save(self, log_dir=None):
        """
        save the log file to a local directory.

        Parameters
        ----------
        log_dir: str,
            The path to the local directory to save log file, if not
        set, log will be saved to the _local_dir by default.

        Returns
        -------
        None
        """
        if self._local_path is not None:
            logger.warning(f"The log file is already exists in `{self._local_path}`.")
            return
        if log_dir is not None:
            self._local_dir = log_dir

        if not os.path.isdir(self._local_dir):
            os.makedirs(self._local_dir)

        filename = os.path.basename(self._oci_path)
        self._local_path = os.path.join(self._local_dir, filename)

        with open(self._local_path, "w+") as in_file:
            in_file.write(self.text)

        logger.info(f"The log file saved to `{self._local_path}`.")

    def __str__(self):
        return self.text

    def _repr_html_(self):
        """
        Display the link of the log file to open in another tab of the jupyter
        environment.

        Returns
        -------
        link of the log file
        """
        if self._local_path is None:
            return

        relative_path = os.path.relpath(self._local_path)
        # FileLink has to be tested with router to check compatibility. Till then let us comment it (ODSC-8310)
        # return display(FileLink(relative_path))
        return logger.info(f"Log file saved to {self._local_path}.")

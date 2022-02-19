#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import logging
import time
from typing import Union

import oci.logging
import oci.loggingsearch
from ads.common.oci_mixin import OCIModelMixin, OCIWorkRequestMixin
from ads.common.oci_resource import OCIResource, ResourceNotFoundError

logger = logging.getLogger(__name__)


class OCILoggingModelMixin(OCIModelMixin, OCIWorkRequestMixin):
    """Base model for representing OCI logging resources managed through oci.logging.LoggingManagementClient.
    This class should not be initialized directly. Use a sub-class (OCILogGroup or OCILog) instead.
    """

    @classmethod
    def from_name(cls, display_name: str) -> Union["OCILogGroup", "OCILog"]:
        """Obtain an existing OCI logging resource by using its display name.
        OCI log group or log resource requires display name to be unique.

        Parameters
        ----------
        display_name : str
            Display name of the logging resource (e.g. log group)

        Returns
        -------
        An instance of logging resource, e.g. OCILogGroup, or OCILog.

        """
        items = cls.list_resource()
        for item in items:
            if item.display_name == display_name:
                return item
        return None

    @classmethod
    def init_client(cls, **kwargs) -> oci.logging.LoggingManagementClient:
        """Initialize OCI client"""
        return cls._init_client(client=oci.logging.LoggingManagementClient, **kwargs)

    @property
    def client(self) -> oci.logging.LoggingManagementClient:
        """OCI logging management client"""
        return super().client

    def create_async(self):
        """Creates the OCI logging resource asynchronously.
        Sub-class should implement this method with OCI Python SDK and return the response from the OCI PythonSDK.

        """
        raise NotImplementedError()

    def create(self):
        """Creates a new resource with OCI service synchronously.
        This method will wait for the resource creation to be succeeded or failed.

        Each sub-class should implement the create_async() method with the corresponding method in OCI SDK
            to create the resource.

        Raises
        ------
        NotImplementedError
            when user called create but the create_async() method is not implemented.

        oci.exceptions.RequestException
            when there is an error creating the resource with OCI request.

        """
        wait_for_states = ("SUCCEEDED", "FAILED")
        success_state = "SUCCEEDED"

        res = self.create_async()
        # Wait for the work request to be completed.
        if "opc-work-request-id" in res.headers:
            opc_work_request_id = res.headers.get("opc-work-request-id")
            res_work_request = self.wait_for_work_request(
                opc_work_request_id,
                wait_for_state=wait_for_states,
                wait_interval_seconds=1,
            )
            # Raise an error if the failed to create the resource.
            if res_work_request.data.status != success_state:
                raise oci.exceptions.RequestException(
                    f"Failed to create {self.__class__.__name__}.\n" + str(res.data)
                )
            self.id = res_work_request.data.resources[0].identifier
            logger.debug("Created %s: %s", self.__class__.__name__, self.id)

        else:
            # This will likely never happen as OCI SDK will raise an error if the HTTP request is not successful.
            raise oci.exceptions.RequestException(
                f"opc-work-request-id not found in response headers: {res.headers}"
            )
        self.sync()
        return self


class OCILogGroup(OCILoggingModelMixin, oci.logging.models.LogGroup):
    """Represents the OCI Log Group resource.

    Using ``OCILogGroup`` to create a new log group.
    OCI requires display_name to be unique and it cannot contain space.
    >>> log_group = OCILogGroup(display_name="My_Log_Group").create()
    Once created, access the OCID and other properties
    >>> log_group.id # The OCID is None before the log is created.
    >>> None
    Create a log resource within the log group
    >>> log_group.id # OCID will be available once the log group is created
    Access the property
    >>> log_group.display_name
    Create logs within the log group
    >>> log = log_group.create_log("My custom access log")
    >>> log_group.create_log("My custom prediction log")
    List the logs in a log group. The following line will return a list of OCILog objects.
    >>> logs = log_group.list_logs()
    Delete the resource
    >>> log_group.delete()
    """

    def create_async(self):
        """Creates a new LogGroup asynchronously with OCI logging service"""
        self.load_properties_from_env()
        return self.client.create_log_group(
            self.to_oci_model(oci.logging.models.CreateLogGroupDetails)
        )

    def create_log(self, display_name: str, **kwargs):
        """Create a log (OCI resource) within the log group.

        Parameters
        ----------
        display_name : str
            The display name of the log
        **kwargs :
            Keyword arguments to be passed into the OCI API for log properties.

        Returns
        -------
        OCILog
            An instance of OCILog

        """
        return OCILog(
            display_name=display_name,
            log_group_id=self.id,
            compartment_id=self.compartment_id,
            **kwargs,
        ).create()

    def list_logs(self, **kwargs) -> list:
        """Lists all logs within the log group.

        Parameters
        ----------
        **kwargs :
            keyword arguments for filtering the results.
            They are passed into oci.logging.LoggingManagementClient.list_logs()

        Returns
        -------
        list
            A list of OCILog

        """
        items = oci.pagination.list_call_get_all_results(
            self.client.list_logs, self.id, **kwargs
        ).data
        return [OCILog.from_oci_model(item) for item in items]

    def delete(self):
        """Deletes the log group and the logs in the log group."""
        logs = self.list_logs()
        for log in logs:
            logger.debug("Deleting OCI Log: %s", log.id)
            log.delete()
        logger.debug("Deleting OCI log group: %s", self.id)
        self.client.delete_log_group(self.id)
        return self


class OCILog(OCILoggingModelMixin, oci.logging.models.Log):
    """Represents the OCI Log resource.

    Usage: (OCI requires display_name to be unique and it cannot contain space)
    >>> log = OCILog.create(display_name="My_Log", log_group_id=LOG_GROUP_ID)
    Usually it is better to create a log using the create_log() method in OCILogGroup.
    >>> log.delete() # Delete the resource
    Get a log object from OCID
    >>> oci_log = OCILog.from_ocid("LOG_OCID_HERE")
    Stream the logs from an OCI Data Science Job run to stdout
    >>> oci_log.stream(source="JOB_RUN_OCID_HERE")
    Gets the most recent 10 logs
    >>> oci_log.tail(10)
    """

    def __init__(self, log_type: str = "CUSTOM", **kwargs) -> None:
        """Initializes an OCI log model locally.
        The resource is not created in OCI until the create() or create_async() method is called.
        """
        if "logType" not in kwargs:
            kwargs["log_type"] = log_type
        super().__init__(**kwargs)
        self._search_client = None

    @staticmethod
    def _get_log_group_id(log_ocid, compartment_ocid: str = None) -> str:
        if not compartment_ocid:
            compartment_ocid = OCIResource.get_compartment_id(log_ocid)
        log_groups = OCILogGroup.list_resource(compartment_id=compartment_ocid)
        for log_group in log_groups:
            oci_logs = log_group.list_logs()
            for oci_log in oci_logs:
                if oci_log.id == log_ocid:
                    return oci_log.log_group_id
        raise ResourceNotFoundError(f"Log not found. OCID={log_ocid}")

    @classmethod
    def _get(cls, ocid) -> oci.logging.models.Log:
        """Gets the OCI Log data (properties)

        OCI API requires both Log OCID and Log Group OCID to get the log properties.
        This method will lookup the log group ID in the same compartment as the log.
        The look up process involves multiple API calls.
        To avoid the lookup, initialize the OCILog with both Log OCID and Log Group OCID,
        for example: log = OCILog(id=LOG_OCID, log_group_id=LOG_GROUP_OCID)

        Parameters
        ----------
        ocid : str
            OCID of the OCI log resource

        Returns
        -------
        oci.logging.models.Log
            An oci.logging.models.Log object containing the properties of the log resource
        """
        log_group_id = cls._get_log_group_id(ocid)
        return cls().client.get_log(log_group_id, ocid).data

    def create_async(self):
        """Creates a new Log with OCI logging service"""
        self.load_properties_from_env()
        return self.client.create_log(
            self.log_group_id, self.to_oci_model(oci.logging.models.CreateLogDetails)
        )

    def sync(self) -> None:
        """Refreshes the properties of the Log model
        OCI requires both Log OCID and Log group OCID to get the Log model.

        This method override the sync() method from OCIMixin to improve performance.
        """
        if not self.log_group_id:
            self.log_group_id = self._get_log_group_id(self.id, self.compartment_id)
        self.update_from_oci_model(self.client.get_log(self.log_group_id, self.id).data)
        return self

    def delete(self):
        """Delete the log"""
        self.client.delete_log(self.log_group_id, self.id)
        return self

    @property
    def search_client(self):
        """OCI logging search client."""
        if not self._search_client:
            self._search_client = self._init_client(
                client=oci.loggingsearch.LogSearchClient
            )
        return self._search_client

    @staticmethod
    def format_datetime(dt: datetime.datetime) -> str:
        """Converts datetime object to RFC3339 date time format in string

        Parameters
        ----------
        dt: datetime.datetime
            Datetime object to be formated.


        Returns
        -------
        str:
            A timestamp in RFC3339 format.

        """
        return dt.isoformat()[:-3] + "Z"

    def search(
        self,
        source: str = None,
        time_start: Union[datetime.datetime, str] = None,
        time_end: Union[datetime.datetime, str] = None,
        limit: int = 100,
        sort_by: str = "datetime",
        sort_order: str = "DESC",
    ):
        """Search logs

        Parameters
        ----------
        source : str
            Expression to match the "source" field of the OCI log record.
        time_start : datetime.datetime or str
            Starting time for the query.
             (Default value = None)
        time_end : datetime.datetime or str
            Ending time for the query.
             (Default value = None)
        limit : int
            Maximum number of records to be returned.
             (Default value = 100)
        sort_by : str
            The field for sorting the logs.
             (Default value = "datetime")
        sort_order : str
            Specify how the records should be sorted. Must be ASC or DESC.
             (Default value = "DESC")

        Returns
        -------

        """
        # Default time_start and time_end
        if time_start is None:
            time_start = datetime.datetime.utcnow() - datetime.timedelta(days=14)
        if time_end is None:
            time_end = datetime.datetime.utcnow()

        # Save original datetime object before conversion
        orig_time_start = time_start

        search_query = f'SEARCH "{self.compartment_id}/{self.log_group_id}/{self.id}"'
        if source:
            search_query += f" | WHERE source = '*{source}'"

        if sort_by:
            if not sort_order:
                sort_order = "DESC"
            search_query += f" | SORT BY {sort_by} {sort_order}"
        results = []
        count = 0
        # From our testing, number larger than total 4 fortnights
        num_fortnights = 3
        while len(results) < limit and count < num_fortnights:
            # Converts datetime objects to RFC3339 format
            if isinstance(time_start, datetime.datetime):
                time_start = self.format_datetime(time_start)
            if isinstance(time_end, datetime.datetime):
                time_end = self.format_datetime(time_end)

            search_details = oci.loggingsearch.models.SearchLogsDetails(
                # time_start cannot be more than 14 days older
                time_start=time_start,
                time_end=time_end,
                # https://docs.oracle.com/en-us/iaas/Content/Logging/Reference/query_language_specification.htm
                # Double quotes must be used for "<log_stream>" after search
                # Single quotes must be used for the string in <where_expression>
                # source = <OCID> is not allowed but source = *<OCID> works
                search_query=search_query,
                # is_return_field_info=True
            )
            results += self.search_client.search_logs(
                search_details, limit=limit
            ).data.results
            time_end = orig_time_start
            time_start = orig_time_start - datetime.timedelta(days=14)
            count += 1
            orig_time_start = time_start
        return results

    def tail(self, source: str = None, limit=100) -> list:
        """Returns the most recent log records.

        Parameters
        ----------
        source : str
            Filter the records by the source field.

        limit : int
            Maximum number of records to be returned.
             (Default value = 100)

        Returns
        -------
        list:
            A list of log records. Each log record is a dictionary with the following keys: id, time, message.

        """
        logs = self.search(source, limit=limit, sort_order="DESC")
        logs = [log.data for log in logs]
        logs = sorted(logs, key=lambda x: x.get("datetime"))
        logs = [log.get("logContent", {}) for log in logs]
        return [
            {
                "id": log.get("id"),
                "message": log.get("data").get("message"),
                "time": log.get("time"),
            }
            for log in logs
        ]

    def stream(
        self,
        source: str = None,
        interval: int = 3,
        stop_condition: callable = None,
        batch_size: int = 100,
    ):
        """Streams logs to console/terminal until stop_condition() returns true.

        Parameters
        ----------
        source : str
        interval : int
            The time interval between sending each request to pull logs from OCI logging service (Default value = 3)
        stop_condition : callable
            A function to determine if the streaming should stop. (Default value = None)
            The log streaming will stop if the function returns true.
        batch_size : int
            (Default value = 100)
            The number of logs to be returned by OCI in each request
            This basically limits the number logs streamed for each interval
            This number should be large enough to cover the messages generated during the interval
            However, Setting this to a large number will decrease the performance
            This method calls the the tail

        """
        # Use a set to store the IDs of the printed messages
        printed = set()
        exception_count = 0
        while True:
            try:
                logs = self.tail(source, batch_size)
            except Exception:
                exception_count += 1
                if exception_count > 20:
                    raise
                else:
                    time.sleep(interval)
                    continue
            for log in logs:
                if log.get("id") not in printed:
                    timestamp = log.get("time", "")
                    if timestamp:
                        timestamp = timestamp.split(".")[0].replace("T", " ")
                    else:
                        timestamp = ""
                    print(f"{timestamp} - {log.get('message')}")
                    printed.add(log.get("id"))
            if stop_condition and stop_condition():
                return
            time.sleep(interval)

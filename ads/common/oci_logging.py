#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import logging
import time
from typing import Dict, Union, List

import oci.logging
import oci.loggingsearch
import oci.exceptions
from ads.common.decorator.utils import class_or_instance_method
from ads.common.oci_mixin import OCIModelMixin, OCIWorkRequestMixin
from ads.common.oci_resource import OCIResource, ResourceNotFoundError


logger = logging.getLogger(__name__)

# Maximum number of log records to be returned by default.
LOG_RECORDS_LIMIT = 100
# The time interval in seconds between sending each request to pull logs
LOG_INTERVAL = 3
MAXIMUM_RETRY_COUNT = 20


class SortOrder:
    ASC = "ASC"
    DESC = "DESC"


class OCILoggingModelMixin(OCIModelMixin, OCIWorkRequestMixin):
    """Base model for representing OCI logging resources managed through oci.logging.LoggingManagementClient.
    This class should not be initialized directly. Use a sub-class (OCILogGroup or OCILog) instead.
    """

    @class_or_instance_method
    def from_name(cls, display_name: str, **kwargs) -> Union["OCILogGroup", "OCILog"]:
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
        items = cls.list_resource(**kwargs)
        for item in items:
            if item.display_name == display_name:
                return item
        return None

    @class_or_instance_method
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
    >>> log = OCILog(display_name="My_Log", log_group_id=LOG_GROUP_ID).create()
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
        self.source = kwargs.get("source", None)
        self.annotation = kwargs.get("annotation", None)

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

    @class_or_instance_method
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

    def sync(self, **kwargs) -> None:
        """Refreshes the properties of the Log model
        OCI requires both Log OCID and Log group OCID to get the Log model.

        This method override the sync() method from OCIMixin to improve performance.
        """
        if not self.log_group_id:
            self.log_group_id = self._get_log_group_id(self.id, self.compartment_id)
        self.update_from_oci_model(
            self.client.get_log(self.log_group_id, self.id).data, **kwargs
        )
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

    @staticmethod
    def _to_utc_native(dt: datetime.datetime) -> str:
        """Converts offset-aware datetime to offset-native UTC time.

        Parameters
        ----------
        dt : datetime.datetime
            A datetime object to be converted.
            If dt is offset native, it will be returned as is.

        Returns
        -------
        datetime.datetime
            Offset-native datetime
        """
        if dt.tzinfo:
            return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return dt

    def _search_logs(
        self,
        time_start: datetime.datetime,
        time_end: datetime.datetime,
        search_query: str,
    ) -> List[oci.loggingsearch.models.SearchResult]:
        """Search logs from OCI logging service

        Due to the limitation on the Log search API.
        This method cannot return more than 1000 logs within 1000 microseconds.

        Parameters
        ----------
        time_start : datetime.datetime
            Starting UTC time for the log search
        time_end : datetime.datetime
            Ending UTC time for the log search
        search_query : str
            Search query following the Logging Query Language Specification.
            https://docs.oracle.com/en-us/iaas/Content/Logging/Reference/query_language_specification.htm

        Returns
        -------
        List[oci.loggingsearch.models.SearchResult]
            A list of SearchResult objects
        """
        # The log search API has a default limit of 100 results per request
        # This limit can be set to 1,000
        # https://docs.oracle.com/en-us/iaas/api/#/en/logging-search/20190909/SearchResult/SearchLogs

        LIMIT_PER_REQUEST = 1000
        logger.debug("Requesting logs between %s and %s ...", time_start, time_end)
        search_details = oci.loggingsearch.models.SearchLogsDetails(
            time_start=self.format_datetime(time_start),
            time_end=self.format_datetime(time_end),
            search_query=search_query,
        )
        result_too_large = False
        try:
            response = self.search_client.search_logs(
                search_details, limit=LIMIT_PER_REQUEST
            )
            records = response.data.results
            logger.debug("%d logs received.", len(records))
        except oci.exceptions.ServiceError as ex:
            if ex.status == 400 and "search result is too large" in ex.message:
                logger.debug(ex.message)
                records = []
                result_too_large = True
            else:
                raise oci.exceptions.ServiceError from ex
        if result_too_large or len(records) >= LIMIT_PER_REQUEST:
            mid = time_start + (time_end - time_start) / 2
            # The log search API used RFC3339 time format.
            # The minimum time unit of RFC3339 format is 1000 microseconds.
            # In the extreme case, if there are over 1000 logs with a 1k-microsecond interval,
            # only 1000 logs can be returned.
            if time_start + datetime.timedelta(microseconds=1000) > mid:
                return records
            return self._search_logs(time_start, mid, search_query) + self._search_logs(
                mid, time_end, search_query
            )
        return records

    def search(
        self,
        source: str = None,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = None,
        sort_by: str = "datetime",
        sort_order: str = "DESC",
        log_filter: str = None,
    ) -> List[oci.loggingsearch.models.SearchResult]:
        """Search logs

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None. No filtering will be performed.
        time_start : datetime.datetime, optional
            Starting UTC time for the query.
            Defaults to None. Logs from the past 24 hours will be returned.
        time_end : datetime.datetime, optional
            Ending UTC time for the query.
            Defaults to None. The current time will be used.
        limit : int, optional
            Maximum number of records to be returned.
            Defaults to None. All logs will be returned.
        sort_by : str, optional
            The field for sorting the logs.
            Defaults to "datetime"
        sort_order : str, optional
            Specify how the records should be sorted. Must be "ASC" or "DESC".
            Defaults to "DESC".
        log_filter : str, optional
            Expression for filtering the logs.
            This will be the WHERE clause of the query.
            Defaults to None.

        Returns
        -------
        List[oci.loggingsearch.models.SearchResult]
            A list of SearchResult objects

        """
        # Default time_start and time_end
        if time_start is None:
            time_start = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        if time_end is None:
            time_end = datetime.datetime.utcnow()

        time_start = self._to_utc_native(time_start)
        time_end = self._to_utc_native(time_end)

        logger.debug("Searching logs between %s and %s ...", time_start, time_end)

        if time_start >= time_end:
            return []

        # Build search query
        search_query = f'SEARCH "{self.compartment_id}/{self.log_group_id}/{self.id}"'
        filters = []

        filter_source = source or self.source
        if filter_source:
            # source = <OCID> is not allowed but source = *<OCID> works
            if filter_source.startswith("ocid1."):
                filter_source = "*" + filter_source
            filters = [f"source = '{filter_source}'"] + filters

        if log_filter:
            filters.append(log_filter)

        if filters:
            search_query += " | WHERE " + " AND ".join(filters)

        if sort_by:
            if not sort_order:
                sort_order = "DESC"
            search_query += f" | SORT BY {sort_by} {sort_order}"
        logger.debug("Log search query: %s", search_query)

        # Save original datetime object before conversion
        orig_time_start = time_start
        # For each search request, time_start cannot be more than 14 days older
        if time_start < time_end - datetime.timedelta(days=14):
            time_start = time_end - datetime.timedelta(days=14)

        results = []

        while not limit or len(results) < limit:
            if time_start + datetime.timedelta(microseconds=1000) > time_end:
                # The log search API used RFC3339 time format.
                # The minimum time unit of RFC3339 format is 1000 microseconds.
                # If the search interval is less than this,
                # there will be InvalidParameter error from OCI.
                records = []
            elif limit:
                logger.debug(
                    "Requesting logs between %s and %s ...", time_start, time_end
                )
                search_details = oci.loggingsearch.models.SearchLogsDetails(
                    # time_start cannot be more than 14 days older
                    time_start=self.format_datetime(time_start),
                    time_end=self.format_datetime(time_end),
                    # https://docs.oracle.com/en-us/iaas/Content/Logging/Reference/query_language_specification.htm
                    # Double quotes must be used for "<log_stream>" after search
                    # Single quotes must be used for the string in <where_expression>
                    search_query=search_query,
                    # is_return_field_info=True
                )
                response = self.search_client.search_logs(
                    search_details, limit=(limit - len(results))
                )
                records = response.data.results
                logger.debug("%d logs received.", len(records))
            else:
                records = self._search_logs(time_start, time_end, search_query)

            results.extend(records)

            if time_start > orig_time_start:
                time_end = time_start
                time_start = time_end - datetime.timedelta(days=14)
                time_start = max(time_start, orig_time_start)
            else:
                break

        return results

    def _search_and_format(
        self,
        source: str = None,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = LOG_RECORDS_LIMIT,
        sort_by: str = "datetime",
        sort_order: str = SortOrder.DESC,
        log_filter: str = None,
    ):
        """Returns the formatted log records.

        Parameters
        ----------
        source: (str, optional). Defaults to None.
            Filter the records by the source field.
        time_start: (datetime.datetime, optional). Defaults to None.
            Starting time for the query.
        time_end: (datetime.datetime, optional). Defaults to None.
            Ending time for the query.
        limit: (int, optional). Defaults to 100.
            Maximum number of records to be returned.
        sort_by: (str, optional). Defaults to "datetime"
            The field for sorting the logs.
        sort_order: (str, optional). Defaults to "DESC".
            The sort order for the log records. Can be "ASC" or "DESC".
        log_filter : (str, optional). Defaults to None.
            Expression for filtering the logs.
            This will be the WHERE clause of the query.

        Returns
        -------
        list
            A list of log records.
            Each log record is a dictionary with the following keys: `id`, `time`, `message`.
        """
        if not time_start:
            time_start = datetime.datetime.utcnow() - datetime.timedelta(days=14)

        logs = self.search(
            source=source,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            log_filter=log_filter,
        )
        logs = sorted((log.data for log in logs), key=lambda x: x.get("datetime"))
        logs = [log.get("logContent", {}) for log in logs]
        return [
            {
                "id": log.get("id"),
                "message": log.get("data", {}).get("message"),
                "time": log.get("time"),
            }
            for log in logs
        ]

    def tail(
        self,
        source: str = None,
        limit=LOG_RECORDS_LIMIT,
        time_start: datetime.datetime = None,
        log_filter: str = None,
    ) -> List[dict]:
        """Returns the most recent log records.

        Parameters
        ----------
        source: (str, optional). Defaults to None.
            The source field to filter the log records.
        limit: (int, optional). Defaults to 100.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
        time_start: (datetime.datetime, optional)
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.
        log_filter : (str, optional). Defaults to None.
            Expression for filtering the logs.
            This will be the WHERE clause of the query.

        Returns
        -------
        list
            A list of log records.
            Each log record is a dictionary with the following keys: `id`, `time`, `message`.
        """
        return self._search_and_format(
            source=source,
            limit=limit,
            sort_order=SortOrder.DESC,
            time_start=time_start,
            log_filter=log_filter,
        )

    def head(
        self,
        source: str = None,
        limit=LOG_RECORDS_LIMIT,
        time_start: datetime.datetime = None,
    ) -> List[dict]:
        """Returns the preceding log records.

        Parameters
        ----------
        source: (str, optional). Defaults to None.
            The source field to filter the log records.
        limit: (int, optional). Defaults to 100.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
        time_start: (datetime.datetime, optional)
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.

        Returns
        -------
        list
            A list of log records.
            Each log record is a dictionary with the following keys: `id`, `time`, `message`.
        """
        return self._search_and_format(
            source=source, limit=limit, sort_order=SortOrder.ASC, time_start=time_start
        )

    def stream(
        self,
        source: str = None,
        interval: int = LOG_INTERVAL,
        stop_condition: callable = None,
        time_start: datetime.datetime = None,
        log_filter: str = None,
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
        time_start: datetime.datetime
            Starting time for the log query.
            Defaults to None. Logs up to 14 days from now will be returned.
        log_filter : str, optional
            Expression for filtering the logs.
            This will be the WHERE clause of the query.
            Defaults to None.

        Returns
        -------
        int:
            The number of logs printed.
        """
        # Use a set to store the IDs of the printed messages
        printed = set()
        exception_count = 0
        while True:
            try:
                # Tail the logs from time_start until now
                # Logs may come in after the timestamp.
                # For the next tail() call,
                # use the time that's 3 minutes before now so that we can have some overlaps.
                next_time_start = datetime.datetime.utcnow() - datetime.timedelta(
                    seconds=180
                )
                logs = self.tail(
                    source, limit=None, time_start=time_start, log_filter=log_filter
                )
                # Update time_start if the tail() is successful
                time_start = next_time_start
            except Exception:
                exception_count += 1
                if exception_count > MAXIMUM_RETRY_COUNT:
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
                return len(printed)
            time.sleep(interval)


class ConsolidatedLog:
    """Represents the Consolidated OCI Log resource.

    Usage:
    Initialize consolidated log instance for oci_log_one and oci_log_two
    >>> oci_log_one = OCILog(
    >>>     compartment_id=<compartment_id_one>,
    >>>     id=<id_one>,
    >>>     log_group_id=<log_group_id_one>,
    >>>     annotation=<annotation_one>
    >>> )
    >>> oci_log_two = OCILog(
    >>>     compartment_id=<compartment_id_two>,
    >>>     id=<id_two>,
    >>>     log_group_id=<log_group_id_two>,
    >>>     annotation=<annotation_two>
    >>> )
    >>> consolidated_log = ConsolidatedLog(oci_log_one, oci_log_two)
    Stream, sort and annotate the logs from oci_log_one and oci_log_two
    >>> consolidated_log.stream()
    Get the most recent 20 consolidated logs from oci_log_one and oci_log_two
    >>> consolidated_log.tail(limit=20)
    Get the most recent 20 raw logs from oci_log_one and oci_log_two
    >>> consolidated_log.search(limit=20)
    """

    def __init__(self, *args) -> None:
        """Initializes a consolidate log model instance.

        Parameters
        ----------
        args:
            A list of OCILog instance.
        """
        self.logging_instance = []
        for arg in args:
            if isinstance(arg, OCILog):
                self.logging_instance.append(arg)
            else:
                raise ValueError(
                    "Unrecognized type. ConsolidatedLog constructor requires OCILog instances as parameters."
                )

    def stream(
        self,
        source: str = None,
        interval: int = LOG_INTERVAL,
        stop_condition: callable = None,
        time_start: datetime.datetime = None,
        log_filter: str = None,
    ):
        """Streams consolidated logs to console/terminal until stop_condition() returns true.

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        interval : int, optional
            The time interval between sending each request to pull logs from OCI logging service.
            Defaults to 3.
        stop_condition : callable, optional
            A function to determine if the streaming should stop.
            The log streaming will stop if the function returns true.
            Defaults to None.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.
        """
        self._print_log_annotation_message()
        # Use a set to store the IDs of the printed messages
        printed = set()
        exception_count = 0
        while True:
            try:
                # Tail the logs from time_start until now
                # Logs may come in after the timestamp.
                # For the next tail() call,
                # use the time that's 3 minutes before now so that we can have some overlaps.
                next_time_start = datetime.datetime.utcnow() - datetime.timedelta(
                    seconds=180
                )
                logs = self._search_and_format(
                    source=source,
                    limit=None,
                    sort_order=SortOrder.DESC,
                    time_start=time_start,
                    log_filter=log_filter,
                )
                # Update time_start if the tail() is successful
                time_start = next_time_start
            except Exception:
                exception_count += 1
                if exception_count > MAXIMUM_RETRY_COUNT:
                    raise
                else:
                    time.sleep(interval)
                    continue
            for log in logs:
                if log.get("id") not in printed:
                    self._print_log_details(log)
                    printed.add(log.get("id"))
            if stop_condition and stop_condition():
                return len(printed)
            time.sleep(interval)

    def tail(
        self,
        source: str = None,
        limit: int = LOG_RECORDS_LIMIT,
        time_start: datetime.datetime = None,
        log_filter: str = None,
    ) -> None:
        """Returns the most recent consolidated log records.

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        limit : int, optional.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
            Defaults to 100.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.
        """
        self._print(
            self._search_and_format(
                source=source,
                limit=limit,
                sort_order=SortOrder.DESC,
                time_start=time_start,
                log_filter=log_filter,
            )
        )

    def head(
        self,
        source: str = None,
        limit: int = LOG_RECORDS_LIMIT,
        time_start: datetime.datetime = None,
    ) -> None:
        """Returns the preceding consolidated log records.

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        limit : int, optional.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
            Defaults to 100.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        """
        self._print(
            self._search_and_format(
                source=source,
                limit=limit,
                sort_order=SortOrder.ASC,
                time_start=time_start,
            )
        )

    def _print_log_annotation_message(self) -> None:
        message = ""
        for instance in self.logging_instance:
            if instance.annotation:
                message = (
                    f"[{instance.annotation[0].upper()}] - {instance.annotation} log, "
                    + message
                )
        if message:
            print(message[:-2])

    def _print(self, logs: List[Dict]) -> None:
        self._print_log_annotation_message()
        for log in logs:
            self._print_log_details(log)

    @staticmethod
    def _print_log_details(log) -> None:
        annotation = log.get("annotation", "")
        if annotation:
            annotation = annotation[0].upper()
        timestamp = log.get("time", "")
        if timestamp:
            timestamp = timestamp.split(".")[0].replace("T", " ")
        print(f"[{annotation}] - {timestamp} - {log.get('message', '')}")

    def search(
        self,
        source: str = None,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = None,
        sort_by: str = "datetime",
        sort_order: str = SortOrder.DESC,
        log_filter: str = None,
    ) -> List[oci.loggingsearch.models.SearchResult]:
        """Searches raw logs.

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        time_end : datetime.datetime, optional.
            Ending time for the query.
            Defaults to None.
        limit : int, optional.
            Maximum number of records to be returned. All logs will be returned.
            Defaults to None.
        sort_by : str, optional.
            The field for sorting the logs.
            Defaults to "datetime"
        sort_order : str, optional.
            The sort order for the log records. Can be "ASC" or "DESC".
            Defaults to "DESC".
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.

        Returns
        -------
        list
            A list of oci.loggingsearch.models.SearchResult objects
        """
        return self._search_and_format(
            source=source,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            log_filter=log_filter,
            need_format=False,
        )

    def _search_and_format(
        self,
        source: str = None,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = LOG_RECORDS_LIMIT,
        sort_by: str = "datetime",
        sort_order: str = SortOrder.DESC,
        log_filter: str = None,
        need_format: bool = True,
    ) -> List[Union[oci.loggingsearch.models.SearchResult, dict]]:
        """Returns the formatted consolidated log records.

        Parameters
        ----------
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        time_end : datetime.datetime, optional.
            Ending time for the query.
            Defaults to None.
        limit : int, optional.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
            Defaults to 100.
        sort_by : str, optional.
            The field for sorting the logs.
            Defaults to "datetime"
        sort_order : str, optional.
            The sort order for the log records. Can be "ASC" or "DESC".
            Defaults to "DESC".
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.
        need_format : bool, optional
            To decide whether to format the logs or not.
            Defaults to True.

        Returns
        -------
        list
            A list of oci.loggingsearch.models.SearchResult objects or log records.
            Each log record is a dictionary with the following keys: `annotation`, `id`, `time`,
            `message` and `datetime`.
        """
        if not time_start:
            time_start = datetime.datetime.utcnow() - datetime.timedelta(days=14)

        batch_logs = []
        for instance in self.logging_instance:
            batch_logs.extend(
                self._collect_logs(
                    instance,
                    source=source,
                    time_start=time_start,
                    time_end=time_end,
                    limit=limit,
                    sort_by=sort_by,
                    sort_order=sort_order,
                    log_filter=log_filter,
                    need_format=need_format,
                )
            )

        # _collect_logs returns a list of either dict or oci.loggingsearch.models.SearchResult
        # objects based on `need_format` parameter, so below there are two cases for log sorting.
        if need_format:
            batch_logs.sort(key=lambda x: x.get("datetime"))
        else:
            batch_logs.sort(key=lambda x: x.data.get("datetime"))
        if limit and len(batch_logs) > limit:
            batch_logs = batch_logs[:limit]
        return batch_logs

    def _collect_logs(
        self,
        logging_instance: OCILog,
        source: str = None,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = LOG_RECORDS_LIMIT,
        sort_by: str = "datetime",
        sort_order: str = SortOrder.DESC,
        log_filter: str = None,
        need_format: bool = True,
    ) -> List[Union[oci.loggingsearch.models.SearchResult, dict]]:
        """Collects logs and formats log records.

        Parameters
        ----------
        logging_instance : OCILog
            OCILog instance.
        source : str, optional
            Expression or OCID to filter the "source" field of the OCI log record.
            Defaults to None.
        time_start : datetime.datetime, optional
            Starting time for the log query.
            Defaults to None.
        time_end : datetime.datetime, optional.
            Ending time for the query.
            Defaults to None.
        limit : int, optional.
            Maximum number of records to be returned.
            If limit is set to None, all logs from time_start to now will be returned.
            Defaults to 100.
        sort_by : str, optional.
            The field for sorting the logs.
            Defaults to "datetime"
        sort_order : str, optional.
            The sort order for the log records. Can be "ASC" or "DESC".
            Defaults to "DESC".
        log_filter : str, optional
            Expression for filtering the logs. This will be the WHERE clause of the query.
            Defaults to None.
        need_format : bool, optional
            To decide whether to format the logs or not.
            Defaults to True.

        Returns
        -------
        list
            A list of oci.loggingsearch.models.SearchResult objects or log records.
            Each log record is a dictionary with the following keys: `annotation`, `id`, `time`,
            `message` and `datetime`.
        """
        logs = logging_instance.search(
            source=source,
            time_start=time_start,
            time_end=time_end,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order,
            log_filter=log_filter,
        )
        if logs and need_format:
            return [
                self._format_and_add_annotation(logging_instance.annotation, log.data)
                for log in logs
            ]

        return logs

    @staticmethod
    def _format_and_add_annotation(annotation: str, data) -> Dict:
        """Formats log and adds annotation.

        Parameters
        ----------
        annotation : str
            The logging annotation of OCILog instance.
        data : SearchResult.data
            The data from oci.loggingsearch.models.SearchResult

        Returns
        -------
        dict:
            Log record dictionary with the following keys: `annotation`, `id`, `time`,
            `message` and `datetime`.
        """
        log_content = data.get("logContent", {})
        return {
            "annotation": annotation,
            "id": log_content.get("id", ""),
            "message": log_content.get("data", {}).get("message", ""),
            "time": log_content.get("time", ""),
            "datetime": data.get("datetime", ""),
        }

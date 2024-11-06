#!/usr/bin/env python

# Copyright (c) 2022, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import collections
import datetime
import json
import logging
import os
from typing import Any, Dict, List

import oci.functions
import pandas as pd

from ads.common.oci_function import OCIFunctionsInvoke, OCIFunctionsManagementMixin
from ads.common.oci_logging import LOG_INTERVAL, LOG_RECORDS_LIMIT, OCILog

logger = logging.getLogger(__name__)

LOG_GROUP_OCID = os.environ.get("TOOL_LOG_GROUP_OCID")
APPLICATION_ID = os.environ.get("APPLICATION_ID")
TERMINAL_STATE = ["FAILED", "CANCELED", "DELETED"]


class OCIFunction(OCIFunctionsManagementMixin, oci.functions.models.Function):
    """Represents an OCI Oracle Function."""

    @classmethod
    def from_id(cls, ocid: str) -> "OCIFunction":
        """Gets function by OCID.

        Parameters
        ----------
        ocid: str
            The OCID of the function.

        Returns
        -------
        OCIFunction
            An instance of `OCIFunction`.
        """
        if not ocid:
            raise ValueError("Function OCID not provided.")
        return super().from_ocid(ocid)

    def list(self, application_id: str = None) -> List["OCIFunction"]:
        return self.client.list_functions(
            application_id=application_id or APPLICATION_ID
        ).data

    def logs(self) -> OCILog:
        """Builds the log for function."""

        # search service logs
        service_logs = (
            OCILog(
                compartment_id=self.compartment_id,
            )
            .client.list_logs(
                log_group_id=LOG_GROUP_OCID,
                source_resource=self.application_id,
                source_service="functions",
                log_type="SERVICE",
            )
            .data
        )

        if not service_logs:
            raise ValueError(
                f"Service log is not configured for the application: {self.application_id}."
            )

        return OCILog(
            id=service_logs[0].id,
            log_group_id=LOG_GROUP_OCID,
            compartment_id=self.compartment_id,
            annotation="service",
        )

    def watch(
        self,
        time_start: datetime = None,
        interval: float = LOG_INTERVAL,
    ) -> "OCIFunction":
        """Watches the logs."""

        time_start = time_start or self.time_created
        try:
            count = self.logs().stream(
                interval=interval,
                stop_condition=self.status in TERMINAL_STATE,
                time_start=time_start,
                log_filter=f"subject = '{self.display_name}'",
            )

            if not count:
                print(
                    "No logs in the last 14 days. Please reset time_start to see older logs."
                )

            return self.sync()
        except KeyboardInterrupt:
            print("Stop watching logs.")
            pass

    def show_logs(
        self,
        time_start: datetime.datetime = None,
        time_end: datetime.datetime = None,
        limit: int = LOG_RECORDS_LIMIT,
    ):
        """Shows logs as a pandas dataframe"""
        logging = self.logs()

        def prepare_log_record(log):
            """Converts a log record to ordered dict"""
            log_content = log.get("logContent", {})
            return collections.OrderedDict(
                [
                    ("type", log_content.get("type").split(".")[-1]),
                    ("id", log_content.get("id")),
                    ("message", log_content.get("data", {}).get("message")),
                    ("time", log_content.get("time")),
                ]
            )

        logs = logging.search(
            time_start=time_start,
            time_end=time_end,
            limit=limit,
            log_filter=f"subject = '{self.display_name}'",
        )
        return pd.DataFrame([prepare_log_record(log.data) for log in logs])

    def invoke(self, payload: Dict[str, Any]) -> Dict:
        """Invokes the function with given payload."""

        return (
            OCIFunctionsInvoke(client_kwargs={"service_endpoint": self.invoke_endpoint})
            .client.invoke_function(
                function_id=self.id, invoke_function_body=json.dumps(payload)
            )
            .data.text
        )

#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import oci
from unittest.mock import patch

from ads.common.oci_logging import ConsolidatedLog, OCILog

FORMATTED_LOGS = [
    {
        "annotation": "access",
        "id": "bc7aed18-889b-446a-9f1c-20a414dc0503",
        "message": "POST /predict 1.1",
        "time": "2022-10-14T22:32:06.861Z",
        "datetime": 1665786726861,
    },
    {
        "annotation": "predict",
        "id": "e3c2e84a-e856-4f51-99b8-d69998244511",
        "message": "--- Logging error ---",
        "time": "2022-10-14T22:32:30.817Z",
        "datetime": 1665786750817,
    },
]

RAW_LOGS = [
    {
        "data": {
            "datetime": 1665786726861,
            "logContent": {
                "data": {
                    "logEmissionTime": "2022-10-14T22:32:06.860Z",
                    "message": "POST /predict 1.1",
                    "modelLatency": 1.77,
                    "opcRequestId": "/0F18708DEC8A1ABA9EB7071C4E67F6FF/365F8ACE27615C0F2888DB85E829F4F1",
                    "status": 500,
                },
                "id": "bc7aed18-889b-446a-9f1c-20a414dc0503",
                "oracle": {
                    "compartmentid": "ocid1.compartment.oc1..<unique_ocid>",
                    "ingestedtime": "2022-10-14T22:32:38.455Z",
                    "loggroupid": "ocid1.loggroup.oc1.iad.<unique_ocid>",
                    "logid": "ocid1.log.oc1.iad.<unique_ocid>",
                    "tenantid": "ocid1.tenancy.oc1..<unique_ocid>",
                },
                "source": "ocid1.datasciencemodeldeployment.oc1.iad.<unique_ocid>",
                "specversion": "1.0",
                "time": "2022-10-14T22:32:06.861Z",
                "type": "com.oraclecloud.datascience.modeldeployment.access",
            },
        }
    },
    {
        "data": {
            "datetime": 1665786750817,
            "logContent": {
                "data": {
                    "logEmissionTime": "2022-10-14T22:32:30.816Z",
                    "message": "--- Logging error ---",
                },
                "id": "e3c2e84a-e856-4f51-99b8-d69998244511",
                "oracle": {
                    "compartmentid": "ocid1.compartment.oc1..<unique_ocid>",
                    "ingestedtime": "2022-10-14T22:33:01.133Z",
                    "loggroupid": "ocid1.loggroup.oc1.iad.<unique_ocid>",
                    "logid": "ocid1.log.oc1.iad.<unique_ocid>",
                    "tenantid": "ocid1.tenancy.oc1..<unique_ocid>",
                },
                "source": "ocid1.datasciencemodeldeployment.oc1.iad.<unique_ocid>",
                "specversion": "1.0",
                "time": "2022-10-14T22:32:30.817Z",
                "type": "com.oraclecloud.datascience.modeldeployment.predict",
            },
        }
    },
]

UNFORMATTED_LOGS = [
    oci.loggingsearch.models.SearchResult(**RAW_LOGS[0]),
    oci.loggingsearch.models.SearchResult(**RAW_LOGS[1]),
]


class TestConsolidatedLog:
    @patch.object(ConsolidatedLog, "_collect_logs", return_value=FORMATTED_LOGS)
    def test_search_and_format(self, mock_collect_log):
        oci_log = OCILog(source="test")
        consolidated_log = ConsolidatedLog(oci_log)
        time_start = datetime.datetime.utcnow()

        logs = consolidated_log._search_and_format(
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=True,
        )

        mock_collect_log.assert_called_with(
            oci_log,
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=True,
        )

        assert len(logs) == 2
        assert logs[0] == FORMATTED_LOGS[0]
        assert logs[1] == FORMATTED_LOGS[1]

    @patch.object(ConsolidatedLog, "_collect_logs", return_value=UNFORMATTED_LOGS)
    def test_search_and_no_format(self, mock_collect_log):
        oci_log = OCILog(source="test")
        consolidated_log = ConsolidatedLog(oci_log)
        time_start = datetime.datetime.utcnow()

        logs = consolidated_log._search_and_format(
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=False,
        )

        mock_collect_log.assert_called_with(
            oci_log,
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=False,
        )

        assert len(logs) == 2
        assert logs[0] == UNFORMATTED_LOGS[0]
        assert logs[1] == UNFORMATTED_LOGS[1]

    @patch.object(OCILog, "search", return_value=[UNFORMATTED_LOGS[0]])
    def test_collect_formatted_logs(self, mock_search):
        oci_log = OCILog(source="test", annotation="access")
        consolidated_log = ConsolidatedLog(oci_log)
        time_start = datetime.datetime.utcnow()

        logs = consolidated_log._collect_logs(
            oci_log,
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=True,
        )

        mock_search.assert_called_with(
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
        )

        assert logs == [FORMATTED_LOGS[0]]

    @patch.object(OCILog, "search", return_value=[UNFORMATTED_LOGS[0]])
    def test_collect_unformatted_logs(self, mock_search):
        oci_log = OCILog(source="test", annotation="access")
        consolidated_log = ConsolidatedLog(oci_log)
        time_start = datetime.datetime.utcnow()

        logs = consolidated_log._collect_logs(
            oci_log,
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
            need_format=False,
        )

        mock_search.assert_called_with(
            source="test",
            time_start=time_start,
            time_end=None,
            limit=20,
            sort_by="datetime",
            sort_order="DESC",
            log_filter=None,
        )

        assert logs == [UNFORMATTED_LOGS[0]]

    def test_format_and_add_annotation(self):
        oci_log = OCILog(source="test", annotation="access")
        consolidated_log = ConsolidatedLog(oci_log)

        logs = consolidated_log._format_and_add_annotation(
            oci_log.annotation, UNFORMATTED_LOGS[0].data
        )

        assert logs == FORMATTED_LOGS[0]

    def test_print_log_annotation_message(self, capsys):
        access_log = OCILog(source="test", annotation="access")
        predict_log = OCILog(source="test", annotation="predict")
        consolidated_log = ConsolidatedLog(access_log, predict_log)
        consolidated_log._print_log_annotation_message()
        captured = capsys.readouterr()
        assert captured.out == "[P] - predict log, [A] - access log\n"

        consolidated_log = ConsolidatedLog(access_log)
        consolidated_log._print_log_annotation_message()
        captured = capsys.readouterr()
        assert captured.out == "[A] - access log\n"

        consolidated_log = ConsolidatedLog(predict_log)
        consolidated_log._print_log_annotation_message()
        captured = capsys.readouterr()
        assert captured.out == "[P] - predict log\n"

    def test_print(self, capsys):
        access_log = OCILog(source="test", annotation="access")
        predict_log = OCILog(source="test", annotation="predict")
        consolidated_log = ConsolidatedLog(access_log, predict_log)

        consolidated_log._print(FORMATTED_LOGS)

        captured = capsys.readouterr()
        assert "[P] - predict log, [A] - access log\n" in captured.out
        assert "[A] - 2022-10-14 22:32:06 - POST /predict 1.1\n" in captured.out
        assert "[P] - 2022-10-14 22:32:30 - --- Logging error ---" in captured.out

    def test_print_log_details(self, capsys):
        access_log = OCILog(source="test", annotation="access")
        predict_log = OCILog(source="test", annotation="predict")
        consolidated_log = ConsolidatedLog(access_log, predict_log)

        consolidated_log._print_log_details(FORMATTED_LOGS[0])
        captured = capsys.readouterr()
        assert "[A] - 2022-10-14 22:32:06 - POST /predict 1.1\n" in captured.out

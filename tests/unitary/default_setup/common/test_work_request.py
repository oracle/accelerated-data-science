#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest
from unittest.mock import MagicMock, patch
from ads.common.work_request import ADSWorkRequest


class TestADSWorkRequest:

    @patch("ads.common.work_request.ADSWorkRequest._sync")
    @patch("ads.common.oci_datascience.OCIDataScienceMixin.__init__")
    def test_watch_succeed(self, mock_oci_datascience, mock_sync):
        ads_work_request = ADSWorkRequest(
            id="test_id",
            description = "Processing"
        )
        ads_work_request._percentage = 90
        ads_work_request._status = "SUCCEEDED"
        ads_work_request.watch(
            progress_callback=MagicMock(),
            poll_interval=0
        )
        mock_oci_datascience.assert_called()
        mock_sync.assert_called()

    @patch("ads.common.work_request.ADSWorkRequest._sync")
    @patch("ads.common.oci_datascience.OCIDataScienceMixin.__init__")
    def test_watch_failed_with_description(self, mock_oci_datascience, mock_sync):
        ads_work_request = ADSWorkRequest(
            id="test_id",
            description = "Backend Error"
        )
        ads_work_request._percentage = 30
        ads_work_request._status = "FAILED"
        with pytest.raises(Exception, match="Backend Error"):
            ads_work_request.watch(
                progress_callback=MagicMock(),
                poll_interval=0
            )
            mock_oci_datascience.assert_called()
            mock_sync.assert_called()

    @patch("ads.common.work_request.ADSWorkRequest._sync")
    @patch("ads.common.oci_datascience.OCIDataScienceMixin.__init__")
    def test_watch_failed_without_description(self, mock_oci_datascience, mock_sync):
        ads_work_request = ADSWorkRequest(
            id="test_id",
            description = None
        )
        ads_work_request._percentage = 30
        ads_work_request._status = "FAILED"
        with pytest.raises(
            Exception,
            match="Error occurred in attempt to perform the operation. "
                "Check the service logs to get more details. "
                f"Work request id: {ads_work_request.id}"
        ):
            ads_work_request.watch(
                progress_callback=MagicMock(),
                poll_interval=0
            )
            mock_oci_datascience.assert_called()
            mock_sync.assert_called()

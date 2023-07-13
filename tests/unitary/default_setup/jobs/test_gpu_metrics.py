#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import unittest
from unittest import mock
import oci
from ads.jobs.templates import oci_metrics


class MetricsTest(unittest.TestCase):
    @mock.patch("oci.monitoring.MonitoringClient.post_metric_data")
    @mock.patch("subprocess.check_output")
    def test_submit_metrics(self, check_output, post_metric):
        check_output.return_value = (
            b"00000000:00:04.0, 42.27, 40, 20, 16384, 15287\n"
            b"00000000:00:05.0, 41.30, 42, 0, 16384, 479\n"
        )
        oci_metrics.submit_metrics(
            oci.monitoring.MonitoringClient(config=oci.config.from_file())
        )
        metric_data = post_metric.call_args.kwargs[
            "post_metric_data_details"
        ].metric_data
        self.assertEqual(len(metric_data), 8)
        self.assertEqual(
            {data.name for data in metric_data},
            {
                "gpu.temperature",
                "gpu.power_draw",
                "gpu.gpu_utilization",
                "gpu.memory_usage",
            },
        )
        self.assertEqual(
            [data.datapoints[0].value for data in metric_data],
            [42.27, 40.0, 20.0, 93.3, 41.3, 42.0, 0.0, 2.92],
        )

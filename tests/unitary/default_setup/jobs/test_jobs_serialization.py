#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import unittest
from unittest import mock

import yaml
from ads.jobs import Job, DataScienceJob, DataScienceJobRun, ScriptRuntime


class JobSerializationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        return super().setUp()

    def test_job_run_serialization(self):
        """Tests job run serialization to YAML."""
        run = DataScienceJobRun(
            display_name="job-run",
            compartment_id="ocid1.compartment.oc1..<unique_ocid>",
            id="ocid1.datasciencejobrun.oc1.<unique_ocid>",
            job_infrastructure_configuration_details=dict(
                shape_name="VM.Standard2.4", job_infrastructure_type="ME_STANDALONE"
            ),
            log_details=dict(
                log_group_id="ocid1.loggroup.oc1.iad.<unique_ocid>",
                log_id="ocid1.log.oc1.iad.<unique_ocid>",
            ),
        )
        with mock.patch("ads.jobs.DataScienceJobRun.sync"), mock.patch(
            "ads.jobs.DataScienceJobRun.job", new_callable=mock.PropertyMock
        ) as mocked_job:
            mocked_job.return_value = (
                Job(name="job")
                .with_infrastructure(
                    DataScienceJob()
                    .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
                    .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
                    .with_log_group_id("ocid1.loggroup.oc1.iad.<unique_ocid>")
                    .with_log_id("ocid1.log.oc1.iad.<unique_ocid>")
                    .with_shape_name("VM.Standard2.1")
                )
                .with_runtime(ScriptRuntime().with_source("my_script.sh"))
            )
            job_run_dict = yaml.safe_load(run.to_yaml())

            self.assertDictEqual(
                job_run_dict,
                yaml.safe_load(
                    """
                    kind: jobRun
                    spec:
                        id: ocid1.datasciencejobrun.oc1.<unique_ocid>
                        infrastructure:
                            kind: infrastructure
                            spec:
                                compartmentId: ocid1.compartment.oc1..<unique_ocid>
                                jobType: DEFAULT
                                logGroupId: ocid1.loggroup.oc1.iad.<unique_ocid>
                                logId: ocid1.log.oc1.iad.<unique_ocid>
                                projectId: ocid1.datascienceproject.oc1.iad.<unique_ocid>
                                shapeName: VM.Standard2.4
                            type: dataScienceJob
                        name: job-run
                        runtime:
                            kind: runtime
                            spec:
                                scriptPathURI: my_script.sh
                            type: script
                    """
                ),
            )

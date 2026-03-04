#!/usr/bin/env python

# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest import TestCase, main, mock, skipUnless

from oci import Response

from ads.jobs import ContainerRuntime, DataScienceJob, Job, PyTorchDistributedRuntime
from ads.jobs.builders.infrastructure.dsc_job_runtime import MULTI_NODE_JOB_SUPPORT

test_cases = {"torchrun": "torchrun test_torch_distributed.py"}


LOG_GROUP_ID = "ocid1.loggroup.oc1.iad.aaa"
LOG_ID = "ocid1.log.oc1.iad.aaa"
SUBNET_ID = "ocid1.subnet.oc1.iad.aaa"
SHAPE_NAME = "VM.GPU.A10.2"
CONDA_NAME = "pytorch24_p310_gpu_x86_64_v1"

CONDA_ENV_VARS = {
    "CONDA_ENV_SLUG": CONDA_NAME,
    "CONDA_ENV_TYPE": "service",
    "JOB_RUN_ENTRYPOINT": "driver_pytorch.py",
    "NODE_COUNT": "2",
    "OCI_LOG_LEVEL": "DEBUG",
    "OCI__LAUNCH_CMD": "torchrun artifact.py",
}

CONTAINER_ENV_VARS = {
    "NODE_COUNT": "2",
    "OCI_LOG_LEVEL": "DEBUG",
}


@skipUnless(
    MULTI_NODE_JOB_SUPPORT,
    "Multi-Node Job is not supported by the OCI Python SDK installed.",
)
class MultiNodeJobTest(TestCase):

    def init_job_infra(self):
        return (
            DataScienceJob()
            .with_compartment_id("ocid1.compartment.oc1..aaa")
            .with_project_id("ocid1.datascienceproject.oc1.iad.aaa")
            .with_log_group_id(LOG_GROUP_ID)
            .with_log_id(LOG_ID)
            .with_shape_name(SHAPE_NAME)
            .with_block_storage_size(256)
        )

    def assert_create_job_details(self, create_job_details, envs):
        # Check log config
        log_config = create_job_details.job_log_configuration_details
        self.assertEqual(log_config.log_id, LOG_ID)
        self.assertEqual(log_config.log_group_id, LOG_GROUP_ID)

        # Check top level configs
        self.assertIsNone(create_job_details.job_configuration_details)
        self.assertIsNone(create_job_details.job_environment_configuration_details)
        self.assertIsNone(create_job_details.job_infrastructure_configuration_details)

        job_node_configuration_details = (
            create_job_details.job_node_configuration_details
        )
        self.assertIsNotNone(job_node_configuration_details)
        # Check network config
        self.assertEqual(
            job_node_configuration_details.job_network_configuration.job_network_type,
            "DEFAULT_NETWORK",
        )
        # Check node group config
        self.assertEqual(
            len(
                job_node_configuration_details.job_node_group_configuration_details_list
            ),
            1,
        )
        node_group_config = (
            job_node_configuration_details.job_node_group_configuration_details_list[0]
        )
        self.assertEqual(
            node_group_config.job_configuration_details.environment_variables,
            envs,
        )
        self.assertEqual(node_group_config.replicas, 2)
        # Check infra config
        infra_config = node_group_config.job_infrastructure_configuration_details
        self.assertEqual(infra_config.shape_name, "VM.GPU.A10.2")
        self.assertEqual(infra_config.block_storage_size_in_gbs, 256)
        self.assertEqual(infra_config.job_infrastructure_type, "MULTI_NODE")

    def assert_create_job_run_details(self, create_job_run_details):
        self.assertIsNone(create_job_run_details.job_configuration_override_details)
        self.assertIsNone(
            create_job_run_details.job_infrastructure_configuration_override_details
        )
        self.assertIsNone(create_job_run_details.job_log_configuration_override_details)
        self.assertIsNone(
            create_job_run_details.job_node_configuration_override_details
        )

    @mock.patch(
        "ads.jobs.builders.runtimes.pytorch_runtime.PyTorchDistributedArtifact.build"
    )
    @mock.patch("ads.jobs.builders.infrastructure.dsc_job.DSCJob.upload_artifact")
    @mock.patch("oci.data_science.DataScienceClient.create_job_run")
    @mock.patch("oci.data_science.DataScienceClient.create_job")
    def test_create_multi_node_job_with_conda(self, patched_create, patched_run, *args):
        patched_create.return_value = Response(
            status=200, headers=None, request=None, data=None
        )

        infra = self.init_job_infra()
        runtime = (
            PyTorchDistributedRuntime()
            # Specify the service conda environment by slug name.
            .with_service_conda(CONDA_NAME)
            .with_command("torchrun artifact.py")
            .with_environment_variable(OCI_LOG_LEVEL="DEBUG")
            .with_replica(2)
        )
        job = Job(name="DT Test").with_infrastructure(infra).with_runtime(runtime)
        job.create()
        create_job_details = patched_create.call_args.args[0]

        self.assert_create_job_details(
            create_job_details=create_job_details,
            envs=CONDA_ENV_VARS,
        )
        node_group_config = create_job_details.job_node_configuration_details.job_node_group_configuration_details_list[
            0
        ]
        self.assertIsNone(node_group_config.job_environment_configuration_details)

        # Create Job with subnet_id
        patched_create.reset_mock()
        infra.with_subnet_id(SUBNET_ID)
        job = Job(name="DT Test").with_infrastructure(infra).with_runtime(runtime)
        job.create()
        create_job_details = patched_create.call_args.args[0]
        job_node_configuration_details = (
            create_job_details.job_node_configuration_details
        )
        self.assertEqual(
            job_node_configuration_details.job_network_configuration.subnet_id,
            SUBNET_ID,
        )
        patched_run.return_value = Response(
            status=200, headers=None, request=None, data=None
        )

        # Check the payload for creating a job run
        job.run()
        create_job_run_details = patched_run.call_args.args[0]
        self.assert_create_job_run_details(create_job_run_details)

    @mock.patch("oci.data_science.DataScienceClient.create_job_run")
    @mock.patch("oci.data_science.DataScienceClient.create_job")
    def test_create_multi_node_job_with_container(
        self, patched_create, patched_run, *args
    ):
        patched_create.return_value = Response(
            status=200, headers=None, request=None, data=None
        )

        infra = self.init_job_infra()
        runtime = (
            ContainerRuntime()
            # Specify the service conda environment by slug name.
            .with_image("container_image")
            .with_environment_variable(OCI_LOG_LEVEL="DEBUG")
            .with_replica(2)
        )
        job = Job(name="DT Test").with_infrastructure(infra).with_runtime(runtime)
        job.create()
        create_job_details = patched_create.call_args.args[0]
        self.assert_create_job_details(
            create_job_details=create_job_details,
            envs=CONTAINER_ENV_VARS,
        )
        node_group_config = create_job_details.job_node_configuration_details.job_node_group_configuration_details_list[
            0
        ]
        container_config = node_group_config.job_environment_configuration_details
        self.assertEqual(container_config.job_environment_type, "OCIR_CONTAINER")
        self.assertEqual(container_config.image, "container_image")

        patched_run.return_value = Response(
            status=200, headers=None, request=None, data=None
        )

        # Check the payload for creating a job run
        job.run()
        create_job_run_details = patched_run.call_args.args[0]
        self.assert_create_job_run_details(create_job_run_details)


if __name__ == "__main__":
    main()

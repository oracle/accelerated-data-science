"""Contains tests for Jobs API authentication."""

from unittest import TestCase
from ads.jobs import Job

JOB_YAML = """
kind: job
apiVersion: v1.0
spec:
  name: llama2
  infrastructure:
    kind: infrastructure
    spec:
      blockStorageSize: 256
      compartmentId: "ocid1.compartment.oc1..aaa"
      logGroupId: "ocid1.loggroup.oc1.iad.aaa"
      logId: "ocid1.log.oc1.iad.aaa"
      projectId: "ocid1.datascienceproject.oc1.iad.aaa"
      subnetId: "ocid1.subnet.oc1.iad.aaa"
      shapeName: VM.GPU.A10.2
    type: dataScienceJob
  runtime:
    kind: runtime
    type: pyTorchDistributed
    spec:
      replicas: 2
      conda:
        type: service
        slug: pytorch20_p39_gpu_v2
      command: >-
        torchrun examples/finetuning.py
"""


class JobsAuthTest(TestCase):
    """Contains tests for Jobs API authentication."""

    def test_auth_from_yaml(self):
        """Test using different endpoints for different jobs."""
        auth1 = {"client_kwargs": {"endpoint": "endpoint1.com"}}
        auth2 = {"client_kwargs": {"endpoint": "endpoint2.com"}}
        job1 = Job(**auth1).from_yaml(JOB_YAML)
        job2 = Job(**auth2).from_yaml(JOB_YAML)
        job3 = Job.from_yaml(JOB_YAML)
        self.assertEqual(job1.auth, auth1)
        self.assertEqual(job1.infrastructure.auth, auth1)
        self.assertEqual(job2.auth, auth2)
        self.assertEqual(job2.infrastructure.auth, auth2)
        self.assertEqual(job3.auth, {})
        self.assertEqual(job3.infrastructure.auth, {})

from ads.jobs import Job, DataScienceJob, ContainerRuntime
from tests.unitary.default_setup.jobs.test_jobs_base import DataScienceJobPayloadTest


class JobTagTestCase(DataScienceJobPayloadTest):
    @staticmethod
    def runtime() -> ContainerRuntime:
        return ContainerRuntime().with_image(
            "iad.ocir.io/my_namespace/my_ubuntu_image",
            entrypoint="/bin/sh",
            cmd="-c,echo Hello World",
        )

    @staticmethod
    def infra() -> DataScienceJob:
        return (
            DataScienceJob()
            .with_compartment_id("ocid1.compartment.oc1..<unique_ocid>")
            .with_project_id("ocid1.datascienceproject.oc1.iad.<unique_ocid>")
        )

    def create_job(self, infra, runtime) -> dict:
        job = (
            Job(name=self.__class__.__name__)
            .with_infrastructure(infra)
            .with_runtime(runtime)
        )
        job = self.mock_create_job(job)
        return job.infrastructure.dsc_job.to_dict()

    def test_create_job_with_runtime_tags(self):
        runtime = (
            self.runtime()
            .with_freeform_tag(freeform_tag="freeform_tag_val")
            .with_defined_tag(Operations={"CostCenter": "42"})
        )
        payload = self.create_job(self.infra(), runtime)
        self.assertEqual(payload["freeformTags"], dict(freeform_tag="freeform_tag_val"))
        self.assertEqual(payload["definedTags"], {"Operations": {"CostCenter": "42"}})

    def test_create_job_with_infra_tags(self):
        infra = (
            self.infra()
            .with_freeform_tag(freeform_tag="freeform_tag_val")
            .with_defined_tag(Operations={"CostCenter": "42"})
        )
        payload = self.create_job(infra, self.runtime())
        self.assertEqual(payload["freeformTags"], dict(freeform_tag="freeform_tag_val"))
        self.assertEqual(payload["definedTags"], {"Operations": {"CostCenter": "42"}})

    def test_create_job_with_infra_and_runtime_tags(self):
        # Tags defined in runtime will have higher priority
        infra = (
            self.infra()
            .with_freeform_tag(freeform_tag="freeform_tag_val")
            .with_defined_tag(Operations={"CostCenter": "41"})
        )
        runtime = (
            self.runtime()
            .with_freeform_tag(freeform_tag="freeform_tag_val")
            .with_defined_tag(Operations={"CostCenter": "42"})
        )
        payload = self.create_job(infra, runtime)
        self.assertEqual(payload["freeformTags"], dict(freeform_tag="freeform_tag_val"))
        self.assertEqual(payload["definedTags"], {"Operations": {"CostCenter": "42"}})

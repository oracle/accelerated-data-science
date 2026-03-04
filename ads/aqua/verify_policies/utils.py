import logging
from ads.aqua.model.model import AquaModelApp
from ads.aqua.verify_policies.constants import DUMMY_TEST_BYTE, OBS_MANAGE_TEST_FILE, TEST_DEFAULT_JOB_SHAPE, TEST_MD_NAME, \
    TEST_MODEL_NAME
from ads.aqua.verify_policies.entities import PolicyStatus
from ads.common.auth import default_signer
from ads.common.oci_mixin import LIFECYCLE_STOP_STATE
from ads.config import COMPARTMENT_OCID, DATA_SCIENCE_SERVICE_NAME, TENANCY_OCID, PROJECT_OCID
from ads.common import oci_client
from rich.console import Console
from rich.logging import RichHandler

logger = logging.getLogger("aqua.policies")

import oci

class PolicyValidationError(Exception):
    def __init__(self, error: str):
        super().__init__(error)
    
class VerifyPoliciesUtils:
    """
    Utility class for verifying OCI IAM policies through operations on Data Science resources.
    Provides methods to interact with models, model deployments, jobs, object storage, and limits APIs
    using Oracle Accelerated Data Science (ADS) SDK.
    """
    def __init__(self):
        self.aqua_model = AquaModelApp()
        self.obs_client = oci_client.OCIClientFactory(**default_signer()).object_storage
        self.model_id = None
        self.job_id = None
        self.limit = 3

    def list_compartments(self, **kwargs):
        """
        List compartments in a given tenancy.

        Parameters:
            compartment_id (str, optional): OCID of the parent compartment. Defaults to TENANCY_OCID.
            limit (int, optional): Maximum number of compartments to return. Defaults to 3.

        Returns:
            List[oci.identity.models.Compartment]: List of compartment data objects.
        """
        compartment_id = kwargs.pop("compartment_id", TENANCY_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.identity_client.list_compartments(compartment_id=compartment_id, limit=limit,
                                                                 **kwargs).data

    def list_models(self, **kwargs):
        """
        List models registered in Data Science.

        Parameters:
            **kwParameters: Filters such as display_name, lifecycle_state, etc.

        Returns:
            List[oci.data_science.models.Model]: List of model metadata.
        """
        return self.aqua_model.list(**kwargs)

    def list_log_groups(self, **kwargs):
        """
        List log groups in the compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Maximum results. Defaults to 3.

        Returns:
            List[oci.logging.models.LogGroupSummary]: List of log groups.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.logging_client.list_log_groups(compartment_id=compartment_id, limit=limit, **kwargs).data

    def list_log(self, **kwargs):
        """
        List logs under a specific log group.

        Parameters:
            log_group_id (str): OCID of the log group.
            limit (int, optional): Maximum number of logs to return. Defaults to 3.

        Returns:
            List[oci.logging.models.LogSummary]: List of log metadata.
        """
        log_group_id = kwargs.pop("log_group_id")
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.logging_client.list_logs(log_group_id=log_group_id, limit=limit, **kwargs).data

    def list_project(self, **kwargs):
        """
        List Data Science projects in a compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Maximum number of projects to return. Defaults to 3.

        Returns:
            List[oci.data_science.models.ProjectSummary]: List of project summaries.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.ds_client.list_projects(compartment_id=compartment_id, limit=limit, **kwargs).data

    def list_model_version_sets(self, **kwargs):
        """
        List model version sets in a compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Max results. Defaults to 3.

        Returns:
            List[oci.data_science.models.ModelVersionSetSummary]: List of version sets.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.ds_client.list_model_version_sets(compartment_id=compartment_id, limit=limit,
                                                                 **kwargs).data

    def list_jobs(self, **kwargs):
        """
        List Data Science jobs in a compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Max results. Defaults to 3.

        Returns:
            List[oci.data_science.models.JobSummary]: List of job summaries.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.ds_client.list_jobs(compartment_id=compartment_id, limit=limit, **kwargs).data

    def list_job_runs(self, **kwargs):
        """
        List job runs in a compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Max results. Defaults to 3.

        Returns:
            List[oci.data_science.models.JobRunSummary]: List of job run records.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        return self.aqua_model.ds_client.list_job_runs(compartment_id=compartment_id, limit=limit, **kwargs).data

    def list_buckets(self, **kwargs):
        """
        List Object Storage buckets in a compartment.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Max results. Defaults to self.limit.

        Returns:
            List[oci.object_storage.models.BucketSummary]: List of buckets.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        limit = kwargs.pop("limit", self.limit)
        namespace_name = self.obs_client.get_namespace(compartment_id=compartment_id).data
        return self.obs_client.list_buckets(namespace_name=namespace_name, compartment_id=compartment_id, limit=limit,
                                            **kwargs).data

    def manage_bucket(self, **kwargs):
        """
        Verify Object Storage access by creating and deleting a test file in a bucket.

        Parameters:
            bucket (str): Name of the bucket to test access in.
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.

        Returns:
            bool: True if test object operations succeeded.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        namespace_name = self.obs_client.get_namespace(compartment_id=compartment_id).data
        bucket = kwargs.pop("bucket")
        logger.info(f"Creating file in object storage with name {OBS_MANAGE_TEST_FILE} in bucket {bucket}")
        self.obs_client.put_object(namespace_name, bucket, object_name=OBS_MANAGE_TEST_FILE, put_object_body="TEST")
        logger.info(f"Deleting file {OBS_MANAGE_TEST_FILE} from object storage")
        self.obs_client.delete_object(namespace_name, bucket, object_name=OBS_MANAGE_TEST_FILE)
        return True

    def list_model_deployment_shapes(self, **kwargs):
        """
        List available model deployment compute shapes.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            limit (int, optional): Max results. Defaults to 3.

        Returns:
            List[oci.data_science.models.ModelDeploymentShapeSummary]: List of shapes.
        """
        limit = kwargs.pop("limit", self.limit)
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        return self.aqua_model.ds_client.list_model_deployment_shapes(compartment_id=compartment_id, limit=limit,
                                                                      **kwargs).data

    def get_resource_availability(self, **kwargs):
        """
        Get quota availability for a specific resource.

        Parameters:
            limit_name (str): Name of the limit (e.g., 'ds-gpu-a10-count').
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.

        Returns:
            oci.limits.models.ResourceAvailability: Quota availability information.
        """
        limits_client = oci_client.OCIClientFactory(**default_signer()).limits
        limit_name = kwargs.pop("limit_name")
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        return limits_client.get_resource_availability(compartment_id=compartment_id,
                                                       service_name=DATA_SCIENCE_SERVICE_NAME,
                                                       limit_name=limit_name).data

    def register_model(self, **kwargs):
        """
        Register a new model with test metadata and a dummy artifact.

        Parameters:
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            project_id (str, optional): Project OCID. Defaults to PROJECT_OCID.

        Returns:
            str: OCID of the registered model.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        project_id = kwargs.pop("project_id", PROJECT_OCID)

        create_model_details = oci.data_science.models.CreateModelDetails(
            compartment_id=compartment_id,
            project_id=project_id,
            display_name=TEST_MODEL_NAME
        )
        logger.info(f"Registering test model `{TEST_MODEL_NAME}`")
        model_id = self.aqua_model.ds_client.create_model(create_model_details=create_model_details).data.id
        self.aqua_model.ds_client.create_model_artifact(
            model_id=model_id,
            model_artifact=DUMMY_TEST_BYTE
        ).data
        self.model_id = model_id
        return model_id

    def create_model_deployment(self, **kwargs):
        """
        Create and deploy a model using a predefined container image and configuration.

        Parameters:
            model_id (str): OCID of the model to deploy.
            instance_shape (str): Compute shape to use (e.g., 'VM.Standard2.1').

        Returns:
            str: OCID of the created model deployment.
        """
        model_id = kwargs.pop("model_id")
        instance_shape = kwargs.pop("instance_shape")
        model_deployment_instance_shape_config_details = oci.data_science.models.ModelDeploymentInstanceShapeConfigDetails(
            ocpus=1,
            memory_in_gbs=6)
        instance_configuration = oci.data_science.models.InstanceConfiguration(
            instance_shape_name=instance_shape,
            model_deployment_instance_shape_config_details=model_deployment_instance_shape_config_details
        )
        model_configuration_details = oci.data_science.models.ModelConfigurationDetails(
            model_id=model_id,
            instance_configuration=instance_configuration
        )

        model_deployment_configuration_details = oci.data_science.models.SingleModelDeploymentConfigurationDetails(
            model_configuration_details=model_configuration_details
        )
        create_model_deployment_details = oci.data_science.models.CreateModelDeploymentDetails(
            compartment_id=COMPARTMENT_OCID,
            project_id=PROJECT_OCID,
            display_name=TEST_MD_NAME,
            model_deployment_configuration_details=model_deployment_configuration_details,
        )
        md_ocid = self.aqua_model.ds_client.create_model_deployment(
            create_model_deployment_details=create_model_deployment_details).data.id
        waiter_result = oci.wait_until(
            self.aqua_model.ds_client,
            self.aqua_model.ds_client.get_model_deployment(md_ocid),
            evaluate_response=lambda r: self._evaluate_response(wait_message="Waiting for Model Deployment to finish",
                                                                response=r),
            max_interval_seconds=30,
        )
        logger.info("Model Deployment may result in FAILED state.")
        return md_ocid

    def _evaluate_response(self, wait_message, response):
        logger.info(f"{wait_message}, Current state: {response.data.lifecycle_state}")
        return getattr(response.data, 'lifecycle_state').upper() in LIFECYCLE_STOP_STATE

    def create_job(self, **kwargs):
        """
        Create a standalone Data Science job with default config and dummy artifact.

        Parameters:
            display_name (str): Display name of the job.
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            project_id (str, optional): Project OCID. Defaults to PROJECT_OCID.
            shape_name (str, optional): Compute shape name. Defaults to TEST_DEFAULT_JOB_SHAPE.
            subnet_id (str, optional): Optional subnet ID.

        Returns:
            str: OCID of the created job.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        project_id = kwargs.pop("project_id", PROJECT_OCID)
        shape_name = kwargs.pop("shape_name", TEST_DEFAULT_JOB_SHAPE)
        display_name = kwargs.pop("display_name")
        subnet_id = kwargs.pop("subnet_id", None)
        job_infrastructure_type = "STANDALONE" if subnet_id is not None else "ME_STANDALONE"

        response = self.aqua_model.ds_client.create_job(
            create_job_details=oci.data_science.models.CreateJobDetails(
                display_name=display_name,
                project_id=project_id,
                compartment_id=compartment_id,
                job_configuration_details=oci.data_science.models.DefaultJobConfigurationDetails(
                    job_type="DEFAULT",
                    environment_variables={}),
                job_infrastructure_configuration_details=oci.data_science.models.StandaloneJobInfrastructureConfigurationDetails(
                    job_infrastructure_type=job_infrastructure_type,
                    shape_name=shape_name,
                    subnet_id=subnet_id,
                    job_shape_config_details=oci.data_science.models.JobShapeConfigDetails(
                        ocpus=1,
                        memory_in_gbs=16),
                    block_storage_size_in_gbs=50
                )
            )
        )

        job_id = response.data.id
        self.aqua_model.ds_client.create_job_artifact(job_id=job_id, job_artifact=b"echo OK\n",
                                                      content_disposition="attachment; filename=entry.sh")
        self.job_id = job_id
        return job_id

    def create_job_run(self, **kwargs):
        """
        Start a job run from an existing job and wait for its completion.

        Parameters:
            job_id (str): OCID of the job to run.
            display_name (str): Display name of the job run.
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            project_id (str, optional): Project OCID. Defaults to PROJECT_OCID.

        Returns:
            oci.data_science.models.JobRun: Job run response after completion.
        """
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        project_id = kwargs.pop("project_id", PROJECT_OCID)
        job_id = kwargs.pop("job_id")
        display_name = kwargs.pop("display_name")
        response = self.aqua_model.ds_client.create_job_run(
            create_job_run_details=oci.data_science.models.CreateJobRunDetails(
                project_id=project_id,
                compartment_id=compartment_id,
                job_id=job_id,
                display_name=display_name
            )
        )
        job_run_id = response.data.id

        waiter_result = oci.wait_until(
            self.aqua_model.ds_client,
            self.aqua_model.ds_client.get_job_run(job_run_id),
            evaluate_response=lambda r: self._evaluate_response(wait_message="Waiting for job run to finish",
                                                                response=r),
            max_interval_seconds=30,
            max_wait_seconds=600
        )
        
        job_run_status = waiter_result.data
        if job_run_status.lifecycle_state  == "FAILED":
            logger.warning(f"Job run failed: {job_run_status.lifecycle_details}")
            raise PolicyValidationError("Job Run Failed")
        return job_run_status

    def create_model_version_set(self, **kwargs):
        """
        Create a new model version set with the specified name.

        Parameters:
            name (str): Name of the model version set.
            compartment_id (str, optional): Compartment OCID. Defaults to COMPARTMENT_OCID.
            project_id (str, optional): Project OCID. Defaults to PROJECT_OCID.

        Returns:
            oci.data_science.models.ModelVersionSet: Model version set creation response.
        """
        name = kwargs.pop("name")
        compartment_id = kwargs.pop("compartment_id", COMPARTMENT_OCID)
        project_id = kwargs.pop("project_id", PROJECT_OCID)
        return self.aqua_model.create_model_version_set(model_version_set_name=name, compartment_id=compartment_id,
                                                        project_id=project_id)


class RichStatusLog:
    def __init__(self):
        self.console = Console()
        # logger = logging.("aqua.policies")
        handler = RichHandler(console=self.console,
                              markup=True,
                              rich_tracebacks=False,
                              show_time=False,
                              show_path=False)
        logger.addHandler(handler)
        logger.propagate = False
        self.logger = logger

    def get_logger(self):
        return self.logger

    def get_status_emoji(self, status: PolicyStatus):
        if status == PolicyStatus.SUCCESS:
            return ":white_check_mark:[green]"
        if status == PolicyStatus.FAILURE:
            return ":cross_mark:[red]"
        if status == PolicyStatus.UNVERIFIED:
            return ":exclamation_question_mark:[yellow]"

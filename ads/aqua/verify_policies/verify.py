import logging

import click
import oci.exceptions

from ads.aqua.verify_policies.constants import TEST_EVALUATION_JOB_NAME, TEST_EVALUATION_JOB_RUN_NAME, \
    TEST_EVALUATION_MVS_NAME, TEST_MD_NAME, TEST_VM_SHAPE
from ads.aqua.verify_policies.messages import operation_messages
from ads.aqua.verify_policies.entities import OperationResultSuccess, OperationResultFailure, PolicyStatus
from ads.aqua.verify_policies.utils import VerifyPoliciesUtils, RichStatusLog
from functools import wraps

logger = logging.getLogger("aqua.policies")


def with_spinner(func):
    @wraps(func)
    def wrapper(self, function, **kwargs):
        operation_message = operation_messages[function.__name__]
        ignore_spinner = kwargs.pop("ignore_spinner", False)

        def run_func():
            return_value, result_status = func(self, function, **kwargs)
            result_message = f"{self._rich_ui.get_status_emoji(result_status.status)} {result_status.operation}"
            if result_status.status == PolicyStatus.SUCCESS:
                logger.info(result_message)
            else:
                logger.warning(result_message)
                logger.info(result_status.error)
                logger.info(f"Policy hint: {result_status.policy_hint}")
            return return_value, result_status

        if ignore_spinner:
            return run_func()
        else:
            with self._rich_ui.console.status(f"Verifying {operation_message['name']}") as status:
                return run_func()

    return wrapper


class AquaVerifyPoliciesApp:
    """Provide options to verify policies of common operation in AQUA such as Model Registration, Model Deployment, Evaluation & Fine-Tuning.

    Methods
    -------
    common_policies: 
        Verify if policies are properly defined for operation such as List Compartments, List Models, List Logs, List Projects, List Jobs, etc.
    model_register: 
        Verify if policies are properly defined to register new models into the Model Catalog. Required to use AQUA to upload and manage custom or service models.
    model_deployment: 
        Verify if policies are properly defined to allows users to deploy models to the OCI Model Deployment service. This operation provisions and starts a new deployment instance.
    evaluation: 
        Verify if policies are properly defined to access to resources such as models, jobs, job-runs and buckets needed throughout the evaluation process. 
    """

    def __init__(self):
        super().__init__()
        self._util = VerifyPoliciesUtils()
        self._rich_ui = RichStatusLog()
        self.model_id = None
        logger.propagate = False
        logger.setLevel(logging.INFO)

    def _get_operation_result(self, operation, status):
        operation_message = operation_messages[operation.__name__]
        if status == PolicyStatus.SUCCESS:
            return OperationResultSuccess(operation=operation_message["name"])
        if status == PolicyStatus.UNVERIFIED:
            return OperationResultSuccess(operation=operation_message["name"], status=status)
        if status == PolicyStatus.FAILURE:
            return OperationResultFailure(operation=operation_message["name"], error=operation_message["error"],
                                          policy_hint=operation_message["policy_hint"])

    @with_spinner
    def _execute(self, function, **kwargs):
        result = None
        try:
            result = function(**kwargs)
            status = PolicyStatus.SUCCESS
        except oci.exceptions.ServiceError as oci_error:
            if oci_error.status == 404:
                logger.debug(oci_error)
                status = PolicyStatus.FAILURE
            else:
                logger.error(oci_error)
                raise oci_error
        except Exception as e:
            logger.error(e)
            raise e
        return result, self._get_operation_result(function, status)

    def _test_model_register(self, **kwargs):
        result = []
        bucket = kwargs.pop("bucket")
        _, test_manage_obs_policy = self._execute(self._util.manage_bucket, bucket=bucket, **kwargs)
        result.append(test_manage_obs_policy.to_dict())

        if test_manage_obs_policy.status == PolicyStatus.SUCCESS:
            self.model_id, test_model_register = self._execute(self._util.register_model)
            result.append(test_model_register.to_dict())
        return result

    def _test_delete_model(self, **kwargs):
        if self.model_id is not None:
            _, test_delete_model_test = self._execute(self._util.aqua_model.ds_client.delete_model,
                                                      model_id=self.model_id, **kwargs)
            return [test_delete_model_test.to_dict()]
        else:
            return [self._get_operation_result(self._util.aqua_model.ds_client.delete_model,
                                               PolicyStatus.UNVERIFIED).to_dict()]

    def _test_model_deployment(self, **kwargs):
        logger.info(f"Creating Model Deployment with name {TEST_MD_NAME}")
        md_ocid, test_model_deployment = self._execute(self._util.create_model_deployment, model_id=self.model_id,
                                                       instance_shape=TEST_VM_SHAPE)
        _, test_delete_md = self._execute(self._util.aqua_deploy.delete, model_deployment_id=md_ocid)
        return [test_model_deployment.to_dict(), test_delete_md.to_dict()]

    def _test_delete_model_deployment(self, **kwargs):
        pass

    def _prompt(self, message, bool=False):
        if bool:
            return click.confirm(message, default=False)
        else:
            return click.prompt(message, type=str)

    def _consent(self):
        answer = self._prompt("Do you want to continue?", bool=True)
        if not answer:
            exit(0)

    def common_policies(self, **kwargs):
        logger.info("[magenta]Verifying Common Policies")
        basic_operations = [self._util.list_compartments, self._util.list_models, self._util.list_model_version_sets,
                            self._util.list_project, self._util.list_jobs, self._util.list_job_runs,
                            self._util.list_buckets,
                            self._util.list_log_groups
                            ]
        result = []
        for op in basic_operations:
            _, status = self._execute(op, **kwargs)
            result.append(status.to_dict())

        _, get_resource_availability_status = self._execute(self._util.get_resource_availability,
                                                            limit_name="ds-gpu-a10-count")
        result.append(get_resource_availability_status.to_dict())
        return result

    def model_register(self, **kwargs):
        logger.info("[magenta]Verifying Model Register")
        logger.info("Object and Model will be created.")
        kwargs.pop("consent", None) == True or self._consent()

        model_save_bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name where model artifacts will be saved")
        register_model_result = self._test_model_register(bucket=model_save_bucket)
        delete_model_result = self._test_delete_model(**kwargs)
        return [*register_model_result, *delete_model_result]

    def model_deployment(self, **kwargs):
        logger.info("[magenta]Verifying Model Deployment")
        logger.info("Object, Model, Model deployment will be created.")
        kwargs.pop("consent", None) == True or self._consent()
        model_save_bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name where model artifacts will be saved")
        model_register = self._test_model_register(bucket=model_save_bucket)
        model_deployment = self._test_model_deployment()
        delete_model_result = self._test_delete_model(**kwargs)

        return [*model_register, *model_deployment, *delete_model_result]

    def evaluation(self, **kwargs):
        logger.info("[magenta]Verifying Evaluation")
        logger.info("Model Version Set, Model, Object, Job and JobRun will be created.")
        kwargs.pop("consent", None) == True or self._consent()

        # Create & Delete MVS
        logger.info(f"Creating Model Version set with name {TEST_EVALUATION_MVS_NAME}")

        model_mvs, test_create_mvs = self._execute(self._util.aqua_model.create_model_version_set,
                                                   name=TEST_EVALUATION_MVS_NAME)
        model_mvs_id = model_mvs[0]
        if model_mvs_id:
            logger.info(f"Deleting Model Version set {TEST_EVALUATION_MVS_NAME}")
            _, delete_mvs = self._execute(self._util.aqua_model.ds_client.delete_model_version_set,
                                          model_version_set_id=model_mvs_id)
        else:
            delete_mvs = self._get_operation_result(self._util.aqua_model.ds_client.delete_model_version_set,
                                                    PolicyStatus.UNVERIFIED)

        # Create & Model
        model_save_bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name where model artifacts will be saved")
        register_model_result = self._test_model_register(bucket=model_save_bucket)
        delete_model_result = self._test_delete_model(**kwargs)

        # Create  Job & JobRun.
        evaluation_job_id, test_create_job = self._execute(self._util.create_job, display_name=TEST_EVALUATION_JOB_NAME,
                                                           **kwargs)
        _, test_create_job_run = self._execute(self._util.create_job_run, display_name=TEST_EVALUATION_JOB_RUN_NAME,
                                               job_id=evaluation_job_id, **kwargs)

        # Delete Job Run
        if evaluation_job_id:
            _, delete_job = self._execute(self._util.aqua_model.ds_client.delete_job, job_id=evaluation_job_id)
        else:
            delete_job = self._get_operation_result(self._util.aqua_model.ds_client.delete_job, PolicyStatus.UNVERIFIED)

        return [test_create_mvs.to_dict(), delete_mvs.to_dict(), *register_model_result, *delete_model_result,
                test_create_job.to_dict(), test_create_job_run.to_dict(), delete_job.to_dict()]



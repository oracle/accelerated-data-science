#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
import sys
from functools import wraps

import click
import oci.exceptions

from ads.aqua.verify_policies.constants import (
    POLICY_HELP_LINK,
    TEST_JOB_NAME,
    TEST_JOB_RUN_NAME,
    TEST_LIMIT_NAME,
    TEST_MD_NAME,
    TEST_MVS_NAME,
    TEST_VM_SHAPE,
)
from ads.aqua.verify_policies.entities import (
    OperationResultFailure,
    OperationResultSuccess,
    PolicyStatus,
)
from ads.aqua.verify_policies.messages import operation_messages
from ads.aqua.verify_policies.utils import (
    PolicyValidationError,
    RichStatusLog,
    VerifyPoliciesUtils,
)

logger = logging.getLogger("aqua.policies")


def with_spinner(func):
    """Decorator to wrap execution of a function with a rich UI spinner.

    Displays status while the operation runs and logs success or failure messages
    based on the policy verification result.
    """

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
                if getattr(result_status, "cleanup_hint", None):
                    logger.info(result_status.cleanup_hint)
                logger.info(f"Refer to: {POLICY_HELP_LINK}")

            return return_value, result_status

        if ignore_spinner:
            return run_func()
        else:
            with self._rich_ui.console.status(f"Verifying {operation_message['name']}"):
                return run_func()

    return wrapper


class AquaVerifyPoliciesApp:
    """
    AquaVerifyPoliciesApp provides methods to verify IAM policies required for
    various operations in OCI Data Science's AQUA (Accelerated Data Science) platform.

    This utility is intended to help users validate whether they have the necessary
    permissions to perform common AQUA workflows such as model registration,
    deployment, evaluation, and fine-tuning.

    Methods
    -------
    `common_policies()`: Validates basic read-level policies across AQUA components.
    `model_register()`: Checks policies for object storage access and model registration.
    `model_deployment()`: Validates policies for registering and deploying models.
    `evaluation()`: Confirms ability to manage model version sets, jobs, and storage for evaluation.
    `finetune()`: Verifies access required to fine-tune models.
    """

    def __init__(self):
        super().__init__()
        self._util = VerifyPoliciesUtils()
        self._rich_ui = RichStatusLog()
        self.model_id = None
        logger.propagate = False
        logger.setLevel(logging.INFO)

    def _get_operation_result(self, operation, status):
        """Maps a function and policy status to a corresponding result object.

        Parameters:
            operation (function): The operation being verified.
            status (PolicyStatus): The outcome of the policy verification.

        Returns:
            OperationResultSuccess or OperationResultFailure based on status.
        """
        operation_message = operation_messages[operation.__name__]
        if status == PolicyStatus.SUCCESS:
            return OperationResultSuccess(operation=operation_message["name"])
        if status == PolicyStatus.UNVERIFIED:
            return OperationResultSuccess(
                operation=operation_message["name"], status=status
            )
        if status == PolicyStatus.FAILURE:
            return OperationResultFailure(
                operation=operation_message["name"],
                error=operation_message["error"],
                policy_hint=f"{operation_message['policy_hint']}",
                cleanup_hint=operation_message.get("cleanup_hint"),
            )

    @with_spinner
    def _execute(self, function, **kwargs):
        """Executes a given operation function with policy validation and error handling.
        Parameters:
            function (callable): The function to execute.
            kwargs (dict): Keyword arguments to pass to the function.

        Returns:
            Tuple: (result, OperationResult)
        """
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
        except PolicyValidationError:
            status = PolicyStatus.FAILURE
        except Exception as e:
            logger.error(e)
            raise e
        return result, self._get_operation_result(function, status)

    def _test_model_register(self, **kwargs):
        """Verifies policies required to manage an object storage bucket and register a model.

        Returns:
            List of result dicts for bucket management and model registration.
        """
        result = []
        bucket = kwargs.pop("bucket")
        _, test_manage_obs_policy = self._execute(
            self._util.manage_bucket, bucket=bucket, **kwargs
        )
        result.append(test_manage_obs_policy.to_dict())

        if test_manage_obs_policy.status == PolicyStatus.SUCCESS:
            self.model_id, test_model_register = self._execute(
                self._util.register_model
            )
            result.append(test_model_register.to_dict())
        return result

    def _test_delete_model(self, **kwargs):
        """Attempts to delete the test model created during model registration.

        Returns:
            List containing the result of model deletion.
        """
        if self.model_id is not None:
            _, test_delete_model_test = self._execute(
                self._util.aqua_model.ds_client.delete_model,
                model_id=self.model_id,
                **kwargs,
            )
            return [test_delete_model_test.to_dict()]
        else:
            return [
                self._get_operation_result(
                    self._util.aqua_model.ds_client.delete_model,
                    PolicyStatus.UNVERIFIED,
                ).to_dict()
            ]

    def _test_model_deployment(self, **kwargs):  # noqa: ARG002
        """Verifies policies required to create and delete a model deployment.

        Returns:
            List of result dicts for deployment creation and deletion.
        """
        logger.info(f"Creating Model Deployment with name {TEST_MD_NAME}")
        md_ocid, test_model_deployment = self._execute(
            self._util.create_model_deployment,
            model_id=self.model_id,
            instance_shape=TEST_VM_SHAPE,
        )
        _, test_delete_md = self._execute(
            self._util.aqua_model.ds_client.delete_model_deployment,
            model_deployment_id=md_ocid,
        )
        return [test_model_deployment.to_dict(), test_delete_md.to_dict()]

    def _test_manage_mvs(self, **kwargs):  # noqa: ARG002
        """Verifies policies required to create and delete a model version set (MVS).

        Returns:
            List of result dicts for MVS creation and deletion.
        """
        logger.info(f"Creating ModelVersionSet with name {TEST_MVS_NAME}")

        model_mvs, test_create_mvs = self._execute(
            self._util.create_model_version_set, name=TEST_MVS_NAME
        )
        model_mvs_id = model_mvs[0]
        if model_mvs_id:
            logger.info(f"Deleting ModelVersionSet {TEST_MVS_NAME}")
            _, delete_mvs = self._execute(
                self._util.aqua_model.ds_client.delete_model_version_set,
                model_version_set_id=model_mvs_id,
            )
        else:
            delete_mvs = self._get_operation_result(
                self._util.aqua_model.ds_client.delete_model_version_set,
                PolicyStatus.UNVERIFIED,
            )
        return [test_create_mvs.to_dict(), delete_mvs.to_dict()]

    def _test_manage_job(self, **kwargs):
        """Verifies policies required to create a job, create a job run, and delete the job.

        Returns:
            List of result dicts for job creation, job run creation, and job deletion.
        """

        logger.info(f"Creating Job with name {TEST_JOB_NAME}")

        # Create Job & JobRun.
        job_id, test_create_job = self._execute(
            self._util.create_job, display_name=TEST_JOB_NAME, **kwargs
        )

        logger.info(f"Creating JobRun with name {TEST_JOB_RUN_NAME}")

        _, test_create_job_run = self._execute(
            self._util.create_job_run,
            display_name=TEST_JOB_RUN_NAME,
            job_id=job_id,
            **kwargs,
        )

        # Delete Job Run
        if job_id:
            _, delete_job = self._execute(
                self._util.aqua_model.ds_client.delete_job,
                job_id=job_id,
                delete_related_job_runs=True,
            )
        else:
            delete_job = self._get_operation_result(
                self._util.aqua_model.ds_client.delete_job, PolicyStatus.UNVERIFIED
            )

        return [
            test_create_job.to_dict(),
            test_create_job_run.to_dict(),
            delete_job.to_dict(),
        ]

    def _prompt(self, message, bool=False):
        """Wrapper for Click prompt or confirmation.

        Parameters:
            message (str): The prompt message.
            bool (bool): Whether to ask for confirmation instead of input.

        Returns:
            User input or confirmation (bool/str).
        """
        if bool:
            return click.confirm(message, default=False)
        else:
            return click.prompt(message, type=str)

    def _consent(self):
        """
        Prompts the user for confirmation before performing actions.
        Exits if the user does not consent.
        """
        answer = self._prompt("Do you want to continue?", bool=True)
        if not answer:
            sys.exit(0)

    def common_policies(self, **kwargs):
        """Verifies basic read-level policies across various AQUA components
        (e.g. compartments, models, jobs, buckets, logs).

        Returns:
            List of result dicts for each verified operation.
        """
        logger.info("[magenta]Verifying Common Policies")
        basic_operations = [
            self._util.list_compartments,
            self._util.list_models,
            self._util.list_model_version_sets,
            self._util.list_project,
            self._util.list_jobs,
            self._util.list_job_runs,
            self._util.list_buckets,
            self._util.list_log_groups,
        ]
        result = []
        for op in basic_operations:
            _, status = self._execute(op, **kwargs)
            result.append(status.to_dict())

        _, get_resource_availability_status = self._execute(
            self._util.get_resource_availability, limit_name=TEST_LIMIT_NAME
        )
        result.append(get_resource_availability_status.to_dict())
        return result

    def model_register(self, **kwargs):
        """Verifies policies required to register a model, including object storage access.

        Returns:
            List of result dicts for registration and cleanup.
        """
        logger.info("[magenta]Verifying Model Register")
        logger.info("Object and Model will be created.")
        kwargs.pop("consent", None) or self._consent()

        model_save_bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name where model artifacts will be saved"
        )
        register_model_result = self._test_model_register(bucket=model_save_bucket)
        delete_model_result = self._test_delete_model(**kwargs)
        return [*register_model_result, *delete_model_result]

    def model_deployment(self, **kwargs):
        """Verifies policies required to register and deploy a model, and perform cleanup.

        Returns:
            List of result dicts for registration, deployment, and cleanup.
        """
        logger.info("[magenta]Verifying Model Deployment")
        logger.info("Model, Model deployment will be created.")
        kwargs.pop("consent", None) or self._consent()

        self.model_id, test_model_register = self._execute(self._util.register_model)
        model_register = [test_model_register.to_dict()]

        model_deployment = self._test_model_deployment() if self.model_id else []
        delete_model_result = self._test_delete_model(**kwargs) if self.model_id else []

        return [*model_register, *model_deployment, *delete_model_result]

    def evaluation(self, **kwargs):
        """Verifies policies for evaluation workloads including model version set,
        job and job runs, and object storage access.

        Returns:
            List of result dicts for all evaluation steps.
        """
        logger.info("[magenta]Verifying Evaluation")
        logger.info("Model Version Set, Model, Object, Job and JobRun will be created.")
        kwargs.pop("consent", None) or self._consent()

        # Create & Delete MVS
        test_manage_mvs = self._test_manage_mvs(**kwargs)

        # Create & Model
        model_save_bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name where model artifacts will be saved"
        )
        register_model_result = self._test_model_register(bucket=model_save_bucket)
        delete_model_result = self._test_delete_model(**kwargs)

        # Manage Jobs & Job Runs
        test_job_and_job_run = self._test_manage_job(**kwargs)

        return [
            *test_manage_mvs,
            *register_model_result,
            *delete_model_result,
            *test_job_and_job_run,
        ]

    def finetune(self, **kwargs):
        """Verifies policies for fine-tuning jobs, including managing object storage,
        MVS.

        Returns:
            List of result dicts for each fine-tuning operation.
        """
        logger.info("[magenta]Verifying Finetuning")
        logger.info(
            "Object, Model Version Set, Job and JobRun will be created. VCN will be used."
        )
        kwargs.pop("consent", None) or self._consent()

        # Manage bucket
        bucket = kwargs.pop("bucket", None) or self._prompt(
            "Provide bucket name required to save training datasets, scripts, and fine-tuned model outputs"
        )

        subnet_id = kwargs.pop("subnet_id", None)
        ignore_subnet = kwargs.pop("ignore_subnet", False)

        if (
            subnet_id is None
            and not ignore_subnet
            and self._prompt("Do you want to use custom subnet", bool=True)
        ):
            subnet_id = self._prompt("Provide subnet id")

        _, test_manage_obs_policy = self._execute(
            self._util.manage_bucket, bucket=bucket, **kwargs
        )

        # Create & Delete MVS
        test_manage_mvs = self._test_manage_mvs(**kwargs)

        # Manage Jobs & Job Runs
        test_job_and_job_run = self._test_manage_job(subnet_id=subnet_id, **kwargs)

        return [
            *test_manage_mvs,
            *test_job_and_job_run,
            test_manage_obs_policy.to_dict(),
        ]

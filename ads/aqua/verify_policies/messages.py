from ads.aqua.verify_policies.constants import TEST_MODEL_NAME, OBS_MANAGE_TEST_FILE, TEST_JOB_NAME, TEST_MVS_NAME, TEST_MD_NAME
from ads.aqua.verify_policies.utils import VerifyPoliciesUtils

utils = VerifyPoliciesUtils()
operation_messages = {
    utils.list_compartments.__name__: {
        "name": "List Compartments",
        "error": "Unable to retrieve the list of compartments. Please verify that you have the required permissions to list compartments in your tenancy. ",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to inspect compartments in tenancy"
    },
    utils.list_models.__name__: {
        "name": "List Models",
        "error": "Failed to fetch available models. Ensure that the policies allow you to list models from the Model Catalog in the selected compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-models in compartment <your-compartment-name>"
    },
    utils.list_project.__name__: {
        "name": "List Projects",
        "error": "Failed to list Data Science projects. Verify that you have the appropriate permission to access projects in the selected compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-projects in compartment <your-compartment-name>"
    },
    utils.list_model_version_sets.__name__: {
        "name": "List Model Version Sets",
        "error": "Unable to load model version sets. Check your access rights to list model version sets in the selected compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-modelversionsets in compartment <your-compartment-name>"
    },
    utils.list_jobs.__name__: {
        "name": "List Jobs",
        "error": "Job list could not be retrieved. Please confirm that you have the necessary permissions to view jobs in the compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-jobs in compartment <your-compartment-name>"
    },
    utils.list_job_runs.__name__: {
        "name": "List Job Runs",
        "error": "Job Runs list could not be retrieved. Please confirm that you have the necessary permissions to view job runs in the compartme",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-job-runs in compartment <your-compartment-name>"
    },
    utils.list_buckets.__name__: {
        "name": "List Object Storage Buckets",
        "error": "Cannot fetch Object Storage buckets. Verify that you have access to list buckets within the specified compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to read objectstorage-namespaces in compartment <your-compartment-name>\nAllow dynamic-group aqua-dynamic-group to read buckets in compartment <your-compartment-name>"
    },
    utils.manage_bucket.__name__: {
        "name": "Object Storage Access",
        "error": "Failed to access the Object Storage bucket. Verify that the bucket exists and you have write permissions.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage object-family in compartment <your-compartment-name> where any {target.bucket.name='<your-bucket-name>'}"
    },
    utils.list_log_groups.__name__: {
        "name": "List Log Groups",
        "error": "Log groups or logs could not be retrieved. Please confirm you have logging read access for the selected compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to use logging-family in compartment <your-compartment-name>"
    },
    utils.get_resource_availability.__name__: {
        "name": "Verify Shape Limits",
        "error": "Failed to retrieve shape limits for your compartment. Make sure the required policies are in place to read shape and quota data.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to read resource-availability in compartment <your-compartment-name>"
    },
    utils.register_model.__name__: {
        "name": "Register Model",
        "error": "Model registration failed. Ensure you have the correct permissions to write to the Model Catalog and access Object Storage.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-models in compartment <your-compartment-name>"
    },
    utils.aqua_model.delete_model.__name__: {
        "name": "Delete Model",
        "error": "Could not delete model. Please confirm you have delete permissions for Model Catalog resources in the compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-models in compartment <your-compartment-name>",
        "cleanup_hint": (
            f'Automatic cleanup of resources failed due to missing policies. '
            f'A model with the name "{TEST_MODEL_NAME}" and a test file with the name "{OBS_MANAGE_TEST_FILE}" '
            f'have been created in the Object Storage bucket you specified. '
            f'Please manually delete the resources to prevent incurring charges.'
        )

    },
    utils.create_job.__name__: {
        "name": "Create Job",
        "error": "Job could not be created. Please check if you have permissions to create Data Science jobs.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-jobs in compartment <your-compartment-name>"
    },
    utils.create_job_run.__name__: {
        "name": "Create Job Run",
        "error": "Job Run could not be created. Confirm that you are allowed to run jobs in the selected compartment.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-job-runs in compartment <your-compartment-name>"
    },
    "delete_job": {
        "name": "Delete Job",
        "error": "Job could not be deleted. Please check if you have permissions to delete Data Science jobs.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-jobs in compartment <your-compartment-name>",
        "cleanup_hint": (
            f'Automatic cleanup of resources failed due to missing policies. '
            f'A job with the name "{TEST_JOB_NAME}" and a test file with the name "{OBS_MANAGE_TEST_FILE}" '
            f'have been created in the Object Storage bucket you specified. '
            f'Please manually delete the resources to prevent incurring charges.'
        )
    },
    utils.aqua_model.create_model_version_set.__name__: {
        "name": "Create Model Version Set",
        "error": "Unable to create a model version set for storing evaluation results. Ensure that required Model Catalog permissions are set.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-modelversionsets in compartment <your-compartment-name>"
    },
    utils.aqua_model.ds_client.delete_model_version_set.__name__: {
        "name": "Delete Model Version Set",
        "error": "Unable to delete a model version. Ensure that required Model Catalog permissions are set.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-modelversionsets in compartment <your-compartment-name>",
        "cleanup_hint": (
            f'Automatic cleanup of resources failed due to missing policies. '
            f'A model version set with the name "{TEST_MVS_NAME}" and a test file with the name "{OBS_MANAGE_TEST_FILE}" '
            f'have been created in the Object Storage bucket you specified. '
            f'Please manually delete the resources to prevent incurring charges.'
        )
    },
    utils.create_model_deployment.__name__: {
        "name": "Create Model Deployment",
        "error": "Model deployment could not be created. Confirm you have correct permissions to deploy models to the Model Deployment service.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-model-deployments in compartment <your-compartment-name>"
    },
    utils.aqua_model.ds_client.delete_model_deployment.__name__: {
        "name": "Delete Model Deployment",
        "error": "Unable to delete the model deployment. Please check if you have appropriate permissions to manage deployments.",
        "policy_hint": "Allow dynamic-group aqua-dynamic-group to manage data-science-model-deployments in compartment <your-compartment-name>",
        "cleanup_hint": (
            f'Automatic cleanup of resources failed due to missing policies. '
            f'A model deployment with the name "{TEST_MD_NAME}" and a test file with the name "{OBS_MANAGE_TEST_FILE}" '
            f'have been created in the Object Storage bucket you specified. '
            f'Please manually delete the resources to prevent incurring charges.'
        )
    }

}

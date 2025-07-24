from unittest.mock import MagicMock, patch
import pytest
from ads.aqua.verify_policies.verify import AquaVerifyPoliciesApp
from ads.aqua.verify_policies.entities import PolicyStatus
from ads.aqua.verify_policies.entities import OperationResultSuccess, OperationResultFailure
from ads.aqua.verify_policies.messages import operation_messages
from oci.exceptions import ServiceError
from contextlib import contextmanager

## Test Helper Operations

# Dummy functions
def mock_success_function():
    return "success result"

def mock_oci_error_function(**kwargs):
    raise ServiceError(status=404, code="NotAuthorizedOrNotFound", message="Resource not found", headers={}, request_id="")

# Register the function name in operation_messages for the test
operation_messages[mock_success_function.__name__] = {
    "name": "Mock Success Function",
    "error": "Mock error message",
    "policy_hint": "Mock policy hint"
}

operation_messages[mock_oci_error_function.__name__] = {
    "name": "Mock OCI Error",
    "error": "Expected error",
    "policy_hint": "Check IAM policies"
}

@pytest.fixture
def app():
    return AquaVerifyPoliciesApp()

@pytest.fixture(autouse=True)
def suppress_logger():
    with patch("ads.aqua.verify_policies.verify.logger", new=MagicMock()):
        yield

def test_get_operation_result_success(app):
    result = app._get_operation_result(mock_success_function, PolicyStatus.SUCCESS)
    assert isinstance(result, OperationResultSuccess)
    assert result.status == PolicyStatus.SUCCESS
    assert result.operation == "Mock Success Function"


def test_get_operation_result_unverified(app):
    result = app._get_operation_result(mock_success_function, PolicyStatus.UNVERIFIED)
    assert isinstance(result, OperationResultSuccess)
    assert result.status == PolicyStatus.UNVERIFIED
    assert result.operation == "Mock Success Function"


def test_get_operation_result_failure(app):
    result = app._get_operation_result(mock_success_function, PolicyStatus.FAILURE)
    assert isinstance(result, OperationResultFailure)
    assert result.status == PolicyStatus.FAILURE
    assert result.operation == "Mock Success Function"
    assert result.error == "Mock error message"
    assert result.policy_hint == "Mock policy hint"

def test_execute_success(app):
    result, status = app._execute(mock_success_function)
    assert result == "success result"
    assert status.status == PolicyStatus.SUCCESS
    assert status.operation == "Mock Success Function"
    
def test_execute_oci_failure_404(app):
    _, status = app._execute(mock_oci_error_function)
    assert status.status == PolicyStatus.FAILURE
    assert status.operation == "Mock OCI Error"
    assert "Expected error" in status.error

def test_test_model_register(app):
    with patch.object(app, '_execute') as mock_execute:
        # Setup the mock return values
        mock_execute.side_effect = [
            (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "manage_bucket"})),
            ("mock_model_id", MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "register_model"})),
        ]

        result = app._test_model_register(bucket="test-bucket")

        assert len(result) == 2
        assert result[0]["op"] == "manage_bucket"
        assert result[1]["op"] == "register_model"
        assert app.model_id == "mock_model_id"

def test_test_delete_model_with_model_id(app):
    app.model_id = "mock_model_id"

    with patch.object(app, "_execute") as mock_execute:
        mock_execute.return_value = (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "delete_model"}))
        result = app._test_delete_model()

        assert len(result) == 1
        assert result[0]["op"] == "delete_model"
        mock_execute.assert_called_once()

def test_test_delete_model_without_model_id(app):
    app.model_id = None  # Simulate missing model

    result = app._test_delete_model()

    assert len(result) == 1
    assert result[0]["status"] == PolicyStatus.UNVERIFIED
    
def test_test_model_deployment(app):
    app.model_id = "mock_model_id"

    with patch.object(app, "_execute") as mock_execute:
        # First call returns model deployment OCID + success status
        # Second call returns success for deletion
        mock_execute.side_effect = [
            ("mock_md_ocid", MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "create_md"})),
            (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "delete_md"})),
        ]

        result = app._test_model_deployment()

        assert len(result) == 2
        assert result[0]["op"] == "create_md"
        assert result[1]["op"] == "delete_md"

        assert mock_execute.call_count == 2
        mock_execute.assert_any_call(app._util.create_model_deployment, model_id="mock_model_id", instance_shape="VM.Standard.E4.Flex")
        mock_execute.assert_any_call(app._util.aqua_model.ds_client.delete_model_deployment, model_deployment_id="mock_md_ocid")

def test_test_manage_mvs(app):
    with patch.object(app, "_execute") as mock_execute:
        # Mock create_model_version_set returning MVS ID
        mock_execute.side_effect = [
            (["mock_mvs_id"], MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "create_mvs"})),
            (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "delete_mvs"})),
        ]

        result = app._test_manage_mvs()

        assert len(result) == 2
        assert result[0]["op"] == "create_mvs"
        assert result[1]["op"] == "delete_mvs"
        assert mock_execute.call_count == 2

def test_test_manage_job(app):
    with patch.object(app, "_execute") as mock_execute:
        # Set up sequential return values
        mock_execute.side_effect = [
            ("mock_job_id", MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "create_job"})),
            (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "create_job_run"})),
            (None, MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "delete_job"})),
        ]

        result = app._test_manage_job()

        assert len(result) == 3
        assert result[0]["op"] == "create_job"
        assert result[1]["op"] == "create_job_run"
        assert result[2]["op"] == "delete_job"
        assert mock_execute.call_count == 3


## Test Public Operations
@contextmanager
def patch_common_app_methods(app):
    with patch.object(app, "_consent", MagicMock(return_value=None)), \
         patch.object(app, "_prompt", MagicMock(return_value="mock-bucket")), \
         patch.object(app, "_test_manage_mvs", MagicMock(return_value=[{"op": "create_mvs"}, {"op": "delete_mvs"}])), \
         patch.object(app, "_test_model_register", MagicMock(return_value=[{"op": "register_model"}])), \
         patch.object(app, "_test_model_register", MagicMock(return_value=[{"op": "register_model"}])), \
         patch.object(app, "_test_delete_model", MagicMock(return_value=[{"op": "delete_model"}])), \
         patch.object(app, "_test_model_deployment", MagicMock(return_value=[{"op": "create_md"}, {"op": "delete_md"}])), \
         patch.object(app, "_test_manage_job", MagicMock(return_value=[{"op": "create_job"}, {"op": "create_job_run"}, {"op": "delete_job"}])):
        yield app


def test_common_policies(app):
    with patch.object(app, "_execute") as mock_execute:
            mock_execute.return_value = ("mock_return_value", MagicMock(status=PolicyStatus.SUCCESS, to_dict=lambda: {"op": "operation_success"}))
            app.common_policies()

            assert mock_execute.call_count == 9
            mock_execute.assert_any_call(app._util.list_compartments)
            mock_execute.assert_any_call(app._util.list_models)
            mock_execute.assert_any_call(app._util.list_model_version_sets)
            mock_execute.assert_any_call(app._util.list_project)
            mock_execute.assert_any_call(app._util.list_jobs)
            mock_execute.assert_any_call(app._util.list_job_runs)
            mock_execute.assert_any_call(app._util.list_buckets)
            mock_execute.assert_any_call(app._util.list_log_groups)
            mock_execute.assert_any_call(app._util.get_resource_availability, limit_name="ds-gpu-a10-count")

def test_model_register(app):
    with patch_common_app_methods(app) as mocks:
        result = app.model_register()

        # Assertions
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["op"] == "register_model"
        assert result[1]["op"] == "delete_model"

        mocks._consent.assert_called_once()
        mocks._prompt.assert_called_once()
        mocks._test_model_register.assert_called_once_with(bucket="mock-bucket")
        mocks._test_delete_model.assert_called_once()


def test_model_deployment(app):
    with patch_common_app_methods(app) as mocks:

        # Run method
        result = app.model_deployment()

        # Assertions
        assert len(result) == 4
        assert result[0]["op"] == "register_model"
        assert result[1]["op"] == "create_md"
        assert result[2]["op"] == "delete_md"
        assert result[3]["op"] == "delete_model"

        mocks._consent.assert_called_once()
        mocks._prompt.assert_called_once()
        mocks._test_model_register.assert_called_once_with(bucket="mock-bucket")
        mocks._test_model_deployment.assert_called_once()
        mocks._test_delete_model.assert_called_once()

def test_evaluation(app):
    with patch_common_app_methods(app) as mocks:

        result = app.evaluation()

        # Assertions
        assert len(result) == 7
        assert result[0]["op"] == "create_mvs"
        assert result[1]["op"] == "delete_mvs"
        assert result[2]["op"] == "register_model"
        assert result[3]["op"] == "delete_model"
        assert result[4]["op"] == "create_job"
        assert result[5]["op"] == "create_job_run"
        assert result[6]["op"] == "delete_job"

        mocks._consent.assert_called_once()
        mocks._prompt.assert_called_once()
        mocks._test_manage_mvs.assert_called_once()
        mocks._test_model_register.assert_called_once_with(bucket="mock-bucket")
        mocks._test_delete_model.assert_called_once()
        mocks._test_manage_job.assert_called_once()

def test_finetune(app):
    with patch.object(app, "_execute") as mock_execute, \
        patch_common_app_methods(app) as mocks:

        # Mock manage_bucket execution
        mock_execute.return_value = (None, MagicMock(
            status=PolicyStatus.SUCCESS,
            to_dict=lambda: {"op": "manage_bucket"}
        ))

        # Call method
        result = app.finetune()

        # Assertions
        assert len(result) == 6
        assert result[0]["op"] == "create_mvs"
        assert result[1]["op"] == "delete_mvs"
        assert result[2]["op"] == "create_job"
        assert result[3]["op"] == "create_job_run"
        assert result[4]["op"] == "delete_job"
        assert result[5]["op"] == "manage_bucket"

        mocks._consent.assert_called_once()
        assert mocks._prompt.call_count == 3
        mocks._execute.assert_called_once_with(app._util.manage_bucket, bucket="mock-bucket")
        mocks._test_manage_mvs.assert_called_once()
        mocks._test_manage_job.assert_called_once()
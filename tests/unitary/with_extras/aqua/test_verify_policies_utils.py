import pytest
from unittest.mock import patch, MagicMock
from ads.aqua.verify_policies.utils import VerifyPoliciesUtils
from ads.config import COMPARTMENT_OCID, PROJECT_OCID, TENANCY_OCID, DATA_SCIENCE_SERVICE_NAME

@pytest.fixture
def utils():
    return VerifyPoliciesUtils()

def test_list_compartments(utils):
    mock_compartments = [MagicMock(name="Compartment1"), MagicMock(name="Compartment2")]
    with patch.object(utils.aqua_model.identity_client, 'list_compartments') as mock_list:
        mock_list.return_value.data = mock_compartments
        result = utils.list_compartments()
        mock_list.assert_called_once_with(compartment_id=TENANCY_OCID, limit=3)
        assert result == mock_compartments

def test_list_models(utils):
    mock_models = [MagicMock(name="Model1"), MagicMock(name="Model2")]
    with patch.object(utils.aqua_model, 'list') as mock_list:
        mock_list.return_value = mock_models
        result = utils.list_models(display_name="TestModel")
        mock_list.assert_called_once_with(display_name="TestModel")
        assert result == mock_models

def test_list_log_groups(utils):
    mock_groups = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.logging_client, 'list_log_groups') as mock_list:
        mock_list.return_value.data = mock_groups
        result = utils.list_log_groups()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_groups

def test_list_log(utils):
    mock_logs = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.logging_client, 'list_logs') as mock_list:
        mock_list.return_value.data = mock_logs
        result = utils.list_log(log_group_id="dummy")
        mock_list.assert_called_once_with(log_group_id="dummy", limit=3)
        assert result == mock_logs

def test_list_project(utils):
    mock_projects = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.ds_client, 'list_projects') as mock_list:
        mock_list.return_value.data = mock_projects
        result = utils.list_project()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_projects

def test_list_model_version_sets(utils):
    mock_sets = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.ds_client, 'list_model_version_sets') as mock_list:
        mock_list.return_value.data = mock_sets
        result = utils.list_model_version_sets()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_sets

def test_list_jobs(utils):
    mock_jobs = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.ds_client, 'list_jobs') as mock_list:
        mock_list.return_value.data = mock_jobs
        result = utils.list_jobs()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_jobs

def test_list_job_runs(utils):
    mock_job_runs = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.ds_client, 'list_job_runs') as mock_list:
        mock_list.return_value.data = mock_job_runs
        result = utils.list_job_runs()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_job_runs

def test_list_buckets(utils):
    mock_buckets = [MagicMock(), MagicMock()]
    with patch.object(utils.obs_client, 'get_namespace') as mock_namespace, \
         patch.object(utils.obs_client, 'list_buckets') as mock_list:
        mock_namespace.return_value.data = "namespace"
        mock_list.return_value.data = mock_buckets
        result = utils.list_buckets()
        mock_list.assert_called_once_with(namespace_name="namespace", compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_buckets

def test_manage_bucket(utils):
    with patch.object(utils.obs_client, 'get_namespace') as mock_namespace, \
         patch.object(utils.obs_client, 'put_object') as mock_put, \
         patch.object(utils.obs_client, 'delete_object') as mock_delete:
        mock_namespace.return_value.data = "namespace"
        result = utils.manage_bucket(bucket="test-bucket")
        mock_put.assert_called_once()
        mock_delete.assert_called_once()
        assert result is True

def test_list_model_deployment_shapes(utils):
    mock_shapes = [MagicMock(), MagicMock()]
    with patch.object(utils.aqua_model.ds_client, 'list_model_deployment_shapes') as mock_list:
        mock_list.return_value.data = mock_shapes
        result = utils.list_model_deployment_shapes()
        mock_list.assert_called_once_with(compartment_id=COMPARTMENT_OCID, limit=3)
        assert result == mock_shapes

def test_get_resource_availability(utils):
    mock_availability = MagicMock()
    with patch('ads.aqua.verify_policies.utils.oci_client.OCIClientFactory') as mock_factory:
        mock_instance = mock_factory.return_value
        mock_instance.limits.get_resource_availability.return_value.data = mock_availability
        result = utils.get_resource_availability(limit_name="test_limit")
        mock_instance.limits.get_resource_availability.assert_called_once_with(
            compartment_id=COMPARTMENT_OCID,
            service_name=DATA_SCIENCE_SERVICE_NAME,
            limit_name="test_limit"
        )
        assert result == mock_availability


def test_register_model(utils):
    with patch.object(utils.aqua_model.ds_client, 'create_model') as mock_create_model, \
         patch.object(utils.aqua_model.ds_client, 'create_model_artifact') as mock_create_artifact:

        mock_model_response = MagicMock()
        mock_model_response.id = "mock_model_id"
        mock_create_model.return_value.data = mock_model_response

        result = utils.register_model()

        mock_create_model.assert_called_once()
        mock_create_artifact.assert_called_once_with(model_id="mock_model_id", model_artifact=b"7IV6cktUGcHIhur4bXTv")
        assert result == "mock_model_id"

def test_create_model_deployment(utils):
    with patch.object(utils.aqua_model.ds_client, 'create_model_deployment') as mock_create_md, \
         patch.object(utils.aqua_model.ds_client, 'get_model_deployment') as mock_get_md, \
         patch('oci.wait_until') as mock_wait:

        mock_md_response = MagicMock()
        mock_md_response.data.id = "mock_deployment_id"
        mock_create_md.return_value = mock_md_response

        mock_get_md.return_value = MagicMock(data=MagicMock(lifecycle_state="SUCCEEDED"))
        mock_wait.return_value = MagicMock(data=MagicMock(lifecycle_state="SUCCEEDED"))

        result = utils.create_model_deployment(model_id="mock_model_id", instance_shape="VM.Standard2.1")

        assert result == "mock_deployment_id"
        mock_create_md.assert_called_once()

def test_create_job(utils):
    with patch.object(utils.aqua_model.ds_client, 'create_job') as mock_create_job, \
         patch.object(utils.aqua_model.ds_client, 'create_job_artifact') as mock_artifact:

        mock_job_response = MagicMock()
        mock_job_response.data.id = "mock_job_id"
        mock_create_job.return_value = mock_job_response

        result = utils.create_job(display_name="Test Job", subnet_id="subnet123")

        assert result == "mock_job_id"
        mock_create_job.assert_called_once()
        mock_artifact.assert_called_once_with(
            job_id="mock_job_id", job_artifact=b"echo OK\n", content_disposition="attachment; filename=entry.sh"
        )

def test_create_job_run(utils):
    with patch.object(utils.aqua_model.ds_client, 'create_job_run') as mock_create_run, \
         patch.object(utils.aqua_model.ds_client, 'get_job_run') as mock_get_run, \
         patch('oci.wait_until') as mock_wait:

        mock_run_response = MagicMock()
        mock_run_response.data.id = "mock_job_run_id"
        mock_create_run.return_value = mock_run_response
        mock_get_run.return_value = MagicMock(data=MagicMock(lifecycle_state="SUCCEEDED"))
        mock_wait.return_value = MagicMock(data=MagicMock(lifecycle_state="SUCCEEDED"))

        result = utils.create_job_run(job_id="mock_job_id", display_name="Test Run")

        assert result.lifecycle_state == "SUCCEEDED"
        mock_create_run.assert_called_once()

def test_create_model_version_set(utils):
    with patch.object(utils.aqua_model, 'create_model_version_set') as mock_create_mvs:
        mock_response = MagicMock()
        mock_create_mvs.return_value = mock_response
        result = utils.create_model_version_set(name="TestMVS")
        mock_create_mvs.assert_called_once_with(
            model_version_set_name="TestMVS",
            compartment_id=COMPARTMENT_OCID,
            project_id=PROJECT_OCID
        )
        assert result == mock_response
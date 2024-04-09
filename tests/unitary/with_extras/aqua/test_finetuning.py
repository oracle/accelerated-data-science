from dataclasses import asdict
from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock

from mock import patch
from ads.aqua.base import AquaApp
from ads.aqua.finetune import AquaFineTuningApp
from ads.aqua.model import AquaFineTuneModel
from ads.jobs.ads_job import Job
from ads.model.datascience_model import DataScienceModel
from ads.model.model_version_set import ModelVersionSet


class FineTuningTestCase(TestCase):

    def setUp(self):
        self.app = AquaFineTuningApp()

    @patch.object(Job, "run")
    @patch("ads.jobs.ads_job.Job.name", new_callable=PropertyMock)
    @patch("ads.jobs.ads_job.Job.id", new_callable=PropertyMock)
    @patch.object(Job, "create")
    @patch("json.dumps")
    @patch.object(AquaFineTuningApp, "get_finetuning_config")
    @patch.object(AquaApp, "create_model_catalog")
    @patch.object(AquaApp, "create_model_version_set")
    @patch.object(AquaApp, "get_source")
    def test_create_fine_tuning(
        self,
        mock_get_source,
        mock_mvs_create,
        mock_ds_model_create,
        mock_get_finetuning_config,
        mock_json_dumps,
        mock_job_create,
        mock_job_id,
        mock_job_name,
        mock_job_run
    ):
        ft_source = MagicMock()
        ft_source.display_name = "test_ft_source_model"
        mock_get_source.return_value = ft_source

        mock_mvs_create.return_value = (
            "test_experiment_id",
            "test_experiment_name"
        )

        ft_model = MagicMock()
        ft_model.id = "test_ft_model_id"
        ft_model.display_name = "test_ft_model_name"

        oci_dsc_model = MagicMock()
        oci_dsc_model.time_created = "test_time_created"
        ft_model.dsc_model = oci_dsc_model
        mock_ds_model_create.return_value = ft_model

        mock_get_finetuning_config.return_value = {}
        mock_json_dumps.return_value = "test_json_string"

        mock_job_id.return_value = "test_ft_job_id"
        mock_job_name.return_value = "test_ft_job_name"

        ft_job_run = MagicMock()
        ft_job_run.id = "test_ft_job_run_id"
        ft_job_run.lifecycle_details = "Job run artifact execution in progress."
        ft_job_run.lifecycle_state = "IN_PROGRESS"
        mock_job_run.return_value = ft_job_run

        self.app.ds_client.update_model = MagicMock()
        self.app.ds_client.update_model_provenance = MagicMock()

        create_aqua_ft_details = dict(
            ft_source_id="ocid1.datasciencemodel.oc1.iad.<OCID>",
            ft_name="test_ft_name",
            dataset_path="oci://ds_bucket@namespace/prefix/dataset.jsonl",
            report_path="oci://report_bucket@namespace/prefix/",
            ft_parameters={
                "epochs":1,
                "learning_rate":0.02
            },
            shape_name="VM.GPU.A10",
            replica=1,
            validation_set_size=0.2,
            block_storage_size=1,
            experiment_name="test_experiment_name",
        )

        aqua_ft_summary = self.app.create(
            **create_aqua_ft_details
        )

        assert asdict(aqua_ft_summary) == {}

    def test_exit_code_message(self):
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run artifact execution failed with exit code 100."
        )
        print(message)
        self.assertEqual(
            message,
            "CUDA out of memory. GPU does not have enough memory to train the model. "
            "Please use a shape with more GPU memory. (exit code 100)",
        )
        # No change should be made for exit code 1
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run artifact execution failed with exit code 1."
        )
        print(message)
        self.assertEqual(message, "Job run artifact execution failed with exit code 1.")

        # No change should be made for other status.
        message = AquaFineTuneModel(
            model=DataScienceModel()
        )._extract_job_lifecycle_details(
            "Job run could not be started due to service issues. Please try again later."
        )
        print(message)
        self.assertEqual(
            message,
            "Job run could not be started due to service issues. Please try again later.",
        )

from unittest import TestCase
from ads.aqua.model import AquaFineTuneModel
from ads.model.datascience_model import DataScienceModel


class FineTuningTestCase(TestCase):
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

from ads.opctl.model.cmds import _download_model, download_model
import pytest
from unittest.mock import ANY, call, patch
from ads.model.datascience_model import DataScienceModel
from unittest.mock import MagicMock, Mock
import os


@patch.object(DataScienceModel, "from_id")
def test_model__download_model(mock_from_id):
    mock_datascience_model = MagicMock()
    mock_from_id.return_value = mock_datascience_model
    _download_model(
        "fake_model_id", "fake_dir", "region", "bucket_uri", 36, False, "api_key", 
    )
    mock_from_id.assert_called_with("fake_model_id")
    mock_datascience_model.download_artifact.assert_called_with(
        target_dir="fake_dir",
        force_overwrite=False,
        overwrite_existing_artifact=True,
        remove_existing_artifact=True,
        region="region",
        timeout=36,
        bucket_uri="bucket_uri",
    )


@patch.object(DataScienceModel, "from_id", side_effect=Exception())
def test_model__download_model_error(mock_from_id):
    with pytest.raises(Exception):
        _download_model(
            "fake_model_id", "fake_dir", "region", "bucket_uri", 36, False,  "api_key",
        )


@patch("ads.opctl.model.cmds._download_model")
def test_download_model(mock__download_model):
    download_model(ocid="fake_model_id")
    mock__download_model.assert_called_once_with(
        ocid="fake_model_id",
        artifact_directory=os.path.expanduser("~/.ads_ops/models/fake_model_id"),
        region=None,
        bucket_uri=None,
        timeout=None,
        force_overwrite=False,
        auth='api_key',
        profile='DEFAULT'
    )

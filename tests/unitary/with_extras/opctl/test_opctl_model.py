from ads.opctl.model.cmds import _download_model, download_model
import pytest
from unittest.mock import ANY, call, patch
from ads.model.datascience_model import DataScienceModel
from unittest.mock import MagicMock, Mock
from ads.opctl.model.cmds import create_signer
import os


@patch.object(DataScienceModel, "from_id")
def test_model__download_model(mock_from_id):
    mock_datascience_model = MagicMock()
    mock_from_id.return_value = mock_datascience_model
    _download_model(
        "fake_model_id", "fake_dir", "fake_auth", "region", "bucket_uri", 36, False
    )
    mock_from_id.assert_called_with("fake_model_id")
    mock_datascience_model.download_artifact.assert_called_with(
        target_dir="fake_dir",
        force_overwrite=False,
        overwrite_existing_artifact=True,
        remove_existing_artifact=True,
        auth="fake_auth",
        region="region",
        timeout=36,
        bucket_uri="bucket_uri",
    )


@patch.object(DataScienceModel, "from_id", side_effect=Exception("Fake error."))
def test_model__download_model_error(mock_from_id):
    with pytest.raises(Exception, match="Fake error."):
        _download_model(
            "fake_model_id", "fake_dir", "fake_auth", "region", "bucket_uri", 36, False
        )


@patch("ads.opctl.model.cmds._download_model")
@patch("ads.opctl.model.cmds.create_signer")
def test_download_model(mock_create_signer, mock__download_model):
    auth_mock = MagicMock()
    mock_create_signer.return_value = auth_mock
    download_model(ocid="fake_model_id")
    mock_create_signer.assert_called_once()
    mock__download_model.assert_called_once_with(
        ocid="fake_model_id",
        artifact_directory=os.path.expanduser("~/.ads_ops/models/fake_model_id"),
        region=None,
        bucket_uri=None,
        timeout=None,
        force_overwrite=False,
        oci_auth=auth_mock,
    )

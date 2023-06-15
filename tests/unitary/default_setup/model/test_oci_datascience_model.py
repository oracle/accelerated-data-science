#!/usr/bin/env python

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest
from oci.data_science.models import (
    ArtifactExportDetailsObjectStorage,
    ArtifactImportDetailsObjectStorage,
    ExportModelArtifactDetails,
    ImportModelArtifactDetails,
    Model,
    ModelProvenance,
    WorkRequest,
    WorkRequestLogEntry,
)
from oci.exceptions import ServiceError
from oci.response import Response

from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.oci_mixin import OCIModelMixin
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.dataset.progress import TqdmProgressBar
from ads.model.datascience_model import _MAX_ARTIFACT_SIZE_IN_BYTES
from ads.model.service.oci_datascience_model import (
    ModelArtifactNotFoundError,
    ModelProvenanceNotFoundError,
    ModelWithActiveDeploymentError,
    OCIDataScienceModel,
)

MODEL_OCID = "ocid1.datasciencemodel.oc1.iad.<unique_ocid>"

OCI_MODEL_PAYLOAD = {
    "id": MODEL_OCID,
    "compartment_id": "ocid1.compartment.oc1..<unique_ocid>",
    "project_id": "ocid1.datascienceproject.oc1.iad.<unique_ocid>",
    "display_name": "Generic Model With Small Artifact new",
    "description": "The model description",
    "lifecycle_state": "ACTIVE",
    "created_by": "ocid1.user.oc1..<unique_ocid>",
    "freeform_tags": {"key1": "value1"},
    "defined_tags": {"key1": {"skey1": "value1"}},
    "time_created": "2022-08-24T17:07:39.200000Z",
    "custom_metadata_list": [
        {
            "key": "CondaEnvironment",
            "value": "oci://bucket@namespace/service_pack/cpu/Data_Exploration_and_Manipulation_for_CPU_Python_3.7/3.0/dataexpl_p37_cpu_v3",
            "description": "The conda environment where the model was trained.",
            "category": "Training Environment",
        },
    ],
    "defined_metadata_list": [
        {"key": "Algorithm", "value": "test"},
        {"key": "Framework"},
        {"key": "FrameworkVersion"},
        {"key": "UseCaseType", "value": "multinomial_classification"},
        {"key": "Hyperparameters"},
        {"key": "ArtifactTestResults"},
    ],
    "input_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
    "output_schema": '{"schema": [{"dtype": "int64", "feature_type": "Integer", "name": 0, "domain": {"values": "", "stats": {}, "constraints": []}, "required": true, "description": "0", "order": 0}], "version": "1.1"}',
}

OCI_MODEL_PROVENANCE_PAYLOAD = {
    "git_branch": "master",
    "git_commit": "7c8c8502896ba36837f15037b67e05a3cf9722c7",
    "repository_url": "file:///home/datascience",
    "script_dir": "test_script_dir",
    "training_id": None,
    "training_script": None,
}

ARTIFACT_HEADER_INFO = {
    "Date": "Sun, 13 Nov 2022 06:01:27 GMT",
    "opc-request-id": "E4F7",
    "ETag": "77156317-8bb9-4c4a-882b-0d85f8140d93",
    "Content-Disposition": "attachment; filename=new_ocid1.datasciencemodel.oc1.iad.<unique_ocid>.zip",
    "Last-Modified": "Sun, 09 Oct 2022 16:50:14 GMT",
    "Content-Type": "application/json",
    "Content-MD5": "orMy3Gs386GZLjYWATJWuA==",
    "X-Content-Type-Options": "nosniff",
    "Content-Length": _MAX_ARTIFACT_SIZE_IN_BYTES + 100,
}


class TestOCIDataScienceModel:
    def setup_class(cls):

        # Mock delete model response
        cls.mock_delete_model_response = Response(
            data=None, status=None, headers=None, request=None
        )

        # Mock create/update model response
        cls.mock_create_model_response = Response(
            data=Model(**OCI_MODEL_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )
        cls.mock_update_model_response = Response(
            data=Model(**OCI_MODEL_PAYLOAD),
            status=None,
            headers=None,
            request=None,
        )

        # Mock model provenance response
        cls.mock_model_provenance = ModelProvenance(**OCI_MODEL_PROVENANCE_PAYLOAD)
        cls.mock_model_provenance_response = Response(
            data=cls.mock_model_provenance,
            status=None,
            headers=None,
            request=None,
        )

        # Mock model artifact content response
        cls.mock_artifact_content_response = Response(
            data=MagicMock(content=b"test"),
            status=None,
            headers=None,
            request=None,
        )

        # Mock import/export artifact
        cls.mock_import_artifact_response = Response(
            data=None,
            status=None,
            headers={"opc-work-request-id": "work_request_id"},
            request=None,
        )
        cls.mock_export_artifact_response = Response(
            data=None,
            status=None,
            headers={"opc-work-request-id": "work_request_id"},
            request=None,
        )

    def setup_method(self):
        self.mock_model = OCIDataScienceModel(**OCI_MODEL_PAYLOAD)

    @pytest.fixture(scope="class")
    def mock_client(self):
        mock_client = MagicMock()
        mock_client.create_model = MagicMock(
            return_value=self.mock_create_model_response
        )
        mock_client.update_model = MagicMock(
            return_value=self.mock_update_model_response
        )
        mock_client.delete_model = MagicMock(
            return_value=self.mock_delete_model_response
        )
        mock_client.create_model_provenance = MagicMock(
            return_value=self.mock_model_provenance_response
        )
        mock_client.update_model_provenance = MagicMock(
            return_value=self.mock_model_provenance_response
        )
        mock_client.get_model_provenance = MagicMock(
            return_value=self.mock_model_provenance_response
        )
        mock_client.get_model_provenance = MagicMock(
            return_value=self.mock_model_provenance_response
        )
        mock_client.get_model_artifact_content = MagicMock(
            return_value=self.mock_artifact_content_response
        )
        mock_client.import_model_artifact = MagicMock(
            return_value=self.mock_import_artifact_response
        )
        mock_client.export_model_artifact = MagicMock(
            return_value=self.mock_export_artifact_response
        )
        return mock_client

    def test_create_fail(self):
        """Ensures creating model fails in case of wrong input params."""
        with pytest.raises(
            ValueError,
            match="The `compartment_id` must be specified.",
        ):
            OCIDataScienceModel().create()
        with pytest.raises(
            ValueError,
            match="The `project_id` must be specified.",
        ):
            OCIDataScienceModel(compartment_id="test").create()

    def test_create_success(self, mock_client):
        """Ensures creating model passes in case of valid input params."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            with patch.object(OCIDataScienceModel, "to_oci_model") as mock_to_oci_model:
                with patch.object(
                    OCIDataScienceModel, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_model
                    mock_oci_model = Model(**OCI_MODEL_PAYLOAD)
                    mock_to_oci_model.return_value = mock_oci_model
                    result = self.mock_model.create()
                    mock_client.create_model.assert_called_with(mock_oci_model)
                    assert result == self.mock_model

    def test_update(self, mock_client):
        """Tests updating datascience Model."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            with patch.object(OCIDataScienceModel, "to_oci_model") as mock_to_oci_model:
                mock_oci_model = Model(**OCI_MODEL_PAYLOAD)
                mock_to_oci_model.return_value = mock_oci_model
                with patch.object(
                    OCIDataScienceModel, "update_from_oci_model"
                ) as mock_update_from_oci_model:
                    mock_update_from_oci_model.return_value = self.mock_model
                    result = self.mock_model.update()
                    mock_client.update_model.assert_called_with(
                        self.mock_model.id, mock_oci_model
                    )
                    assert result == self.mock_model

    def test_delete_success(self, mock_client):
        """Ensures model can be deleted."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            with patch.object(
                OCIDataScienceModel, "model_deployment"
            ) as mock_model_deployment:
                mock_model_deployment.return_value = [
                    MagicMock(lifecycle_state="ACTIVE", identifier="md_id")
                ]
                with patch("ads.model.deployment.ModelDeployment.from_id") as mock_from_id:
                    with patch.object(OCIDataScienceModel, "sync") as mock_sync:
                        self.mock_model.delete(delete_associated_model_deployment=True)
                        mock_from_id.assert_called_with("md_id")
                        mock_client.delete_model.assert_called_with(self.mock_model.id)
                        mock_sync.assert_called()

    def test_delete_fail(self, mock_client):
        """Ensures deleting model fails in case if there are active model deployments."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            with patch.object(
                OCIDataScienceModel, "model_deployment"
            ) as mock_model_deployment:
                mock_model_deployment.return_value = [
                    MagicMock(lifecycle_state="ACTIVE", identifier="md_id")
                ]
                with pytest.raises(ModelWithActiveDeploymentError):
                    self.mock_model.delete(delete_associated_model_deployment=False)

    @patch.object(OCIModelMixin, "from_ocid")
    def test_from_id(self, mock_from_ocid):
        """Tests getting a model by OCID."""
        OCIDataScienceModel.from_id(MODEL_OCID)
        mock_from_ocid.assert_called_with(MODEL_OCID)

    def test_create_model_provenance(self, mock_client):
        """Tests creating model provenance metadata."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            test_result = self.mock_model.create_model_provenance(
                self.mock_model_provenance
            )
            mock_client.create_model_provenance.assert_called_with(
                MODEL_OCID, self.mock_model_provenance
            )
            assert test_result == self.mock_model_provenance

    def test_update_model_provenance(self, mock_client):
        """Test updating model provenance metadata."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            test_result = self.mock_model.update_model_provenance(
                self.mock_model_provenance
            )
            mock_client.update_model_provenance.assert_called_with(
                MODEL_OCID, self.mock_model_provenance
            )
            assert test_result == self.mock_model_provenance

    def test_get_model_provenance_success(self, mock_client):
        """Tests getting model provenance metadata."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            test_result = self.mock_model.get_model_provenance()
            mock_client.get_model_provenance.assert_called_with(MODEL_OCID)
            assert test_result == self.mock_model_provenance

    @patch.object(OCIDataScienceModel, "client")
    def test_get_model_provenance_fail(self, mock_client):
        """Tests getting model provenance metadata."""
        mock_client.get_model_provenance = MagicMock(
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            )
        )
        with pytest.raises(ModelProvenanceNotFoundError):
            self.mock_model.get_model_provenance()
            mock_client.get_model_provenance.assert_called_with(MODEL_OCID)

    def test_get_artifact_info_success(self, mock_client):
        """Tests getting model artifact attachment information."""
        mock_client.head_model_artifact = MagicMock(
            return_value=Response(
                status=None,
                data=None,
                request=None,
                headers=ARTIFACT_HEADER_INFO,
            )
        )
        with patch.object(OCIDataScienceModel, "client", mock_client):
            test_result = self.mock_model.get_artifact_info()
            assert test_result == ARTIFACT_HEADER_INFO

    @patch.object(OCIDataScienceModel, "client")
    def test_get_artifact_info_fail(self, mock_client):
        """Tests getting model artifact attachment information."""
        mock_client.head_model_artifact = MagicMock(
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            )
        )
        with pytest.raises(ModelArtifactNotFoundError):
            self.mock_model.get_artifact_info()
            mock_client.head_model_artifact.assert_called_with(MODEL_OCID)

    def test_get_model_artifact_content_success(self, mock_client):
        """Tests getting model artifact content."""
        with patch.object(OCIDataScienceModel, "client", mock_client):
            test_result = self.mock_model.get_model_artifact_content()
            mock_client.get_model_artifact_content.assert_called_with(
                model_id=MODEL_OCID
            )
            assert test_result == b"test"

    @patch.object(OCIDataScienceModel, "client")
    def test_get_model_artifact_content_fail(self, mock_client):
        """Tests getting model artifact content."""
        mock_client.get_model_artifact_content = MagicMock(
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            )
        )
        with pytest.raises(ModelArtifactNotFoundError):
            self.mock_model.get_model_artifact_content()
            mock_client.get_model_artifact_content.assert_called_with(MODEL_OCID)

    @patch.object(OCIDataScienceModel, "client")
    def test_create_model_artifact(self, mock_client):
        """Tests creating model artifact for specified model."""
        mock_client.create_model_artifact = MagicMock()
        test_data = b"test"
        self.mock_model.create_model_artifact(test_data)
        mock_client.create_model_artifact.assert_called_with(
            MODEL_OCID,
            test_data,
            content_disposition=f'attachment; filename="{MODEL_OCID}.zip"',
        )

    @patch.object(OCIResource, "search")
    def test_model_deployment(self, mock_search):
        """Tests getting the list of model deployments by model ID across the compartments."""
        self.mock_model.model_deployment(
            config={"key": "value"},
            tenant_id="tenant_id",
            limit=100,
            page="1",
            **{"kwargkey": "kwargvalue"},
        )
        mock_search.assert_called_with(
            f"query datasciencemodeldeployment resources where ModelId='{MODEL_OCID}'",
            type=SEARCH_TYPE.STRUCTURED,
            config={"key": "value"},
            tenant_id="tenant_id",
            limit=100,
            page="1",
            **{"kwargkey": "kwargvalue"},
        )

    @patch.object(OCIDataScienceModel, "_wait_for_work_request")
    def test_import_model_artifact_success(
        self,
        mock_wait_for_work_request,
        mock_client,
    ):
        """Tests importing model artifact content from the model catalog."""
        test_bucket_uri = "oci://bucket@namespace/prefix"
        test_bucket_details = ObjectStorageDetails.from_path(test_bucket_uri)
        test_region = "test_region"
        with patch.object(OCIDataScienceModel, "client", mock_client):
            self.mock_model.import_model_artifact(
                bucket_uri=test_bucket_uri, region=test_region
            )
            mock_client.import_model_artifact.assert_called_with(
                model_id=self.mock_model.id,
                import_model_artifact_details=ImportModelArtifactDetails(
                    artifact_import_details=ArtifactImportDetailsObjectStorage(
                        namespace=test_bucket_details.namespace,
                        destination_bucket=test_bucket_details.bucket,
                        destination_object_name=test_bucket_details.filepath,
                        destination_region=test_region,
                    )
                ),
            )
            mock_wait_for_work_request.assert_called_with(
                work_request_id="work_request_id",
                num_steps=2,
            )

    @patch.object(OCIDataScienceModel, "client")
    def test_import_model_artifact_fail(self, mock_client):
        """Tests importing model artifact content from the model catalog."""
        test_bucket_uri = "oci://bucket@namespace/prefix"
        mock_client.import_model_artifact = MagicMock(
            side_effect=ServiceError(
                status=404, code="code", message="message", headers={}
            )
        )
        with pytest.raises(ModelArtifactNotFoundError):
            self.mock_model.import_model_artifact(
                bucket_uri=test_bucket_uri, region="test_region"
            )

    @patch.object(OCIDataScienceModel, "_wait_for_work_request")
    def test_export_model_artifact(
        self,
        mock_wait_for_work_request,
        mock_client,
    ):
        """Tests exporting model artifact to the model catalog."""
        test_bucket_uri = "oci://bucket@namespace/prefix"
        test_bucket_details = ObjectStorageDetails.from_path(test_bucket_uri)
        test_region = "test_region"
        with patch.object(OCIDataScienceModel, "client", mock_client):
            self.mock_model.export_model_artifact(
                bucket_uri=test_bucket_uri, region=test_region
            )
            mock_client.export_model_artifact.assert_called_with(
                model_id=self.mock_model.id,
                export_model_artifact_details=ExportModelArtifactDetails(
                    artifact_export_details=ArtifactExportDetailsObjectStorage(
                        namespace=test_bucket_details.namespace,
                        source_bucket=test_bucket_details.bucket,
                        source_object_name=test_bucket_details.filepath,
                        source_region=test_region,
                    )
                ),
            )
            mock_wait_for_work_request.assert_called_with(
                work_request_id="work_request_id",
                num_steps=3,
            )

    @patch.object(TqdmProgressBar, "update")
    def test__wait_for_work_request_fail(self, mock_tqdm_update, mock_client):
        mock_client.get_work_request = MagicMock(
            return_value=Response(
                data=WorkRequest(id="work_request_id", status="FAILED"),
                status=None,
                headers={"opc-work-request-id": "work_request_id"},
                request=None,
            )
        )
        mock_client.list_work_request_logs = MagicMock(
            return_value=Response(
                data=[
                    WorkRequestLogEntry(message="test_message_1"),
                    WorkRequestLogEntry(message="error_message_1"),
                ],
                status=None,
                headers=None,
                request=None,
            )
        )
        with patch.object(
            OCIDataScienceModel,
            "client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ):
            with pytest.raises(Exception, match="error_message_1"):
                self.mock_model._wait_for_work_request(
                    work_request_id="work_request_id", num_steps=2
                )
                mock_tqdm_update.assert_has_calls(
                    [
                        call("test_message_1"),
                        call("error_message_1"),
                    ]
                )
                assert mock_tqdm_update.call_count == 2

    @patch.object(TqdmProgressBar, "update")
    def test__wait_for_work_request_fail_generic(self, mock_tqdm_update, mock_client):
        mock_client.get_work_request = MagicMock(
            return_value=Response(
                data=WorkRequest(id="work_request_id", status="FAILED"),
                status=None,
                headers={"opc-work-request-id": "work_request_id"},
                request=None,
            )
        )
        mock_client.list_work_request_logs = MagicMock(
            return_value=Response(
                data=[],
                status=None,
                headers=None,
                request=None,
            )
        )
        with patch.object(
            OCIDataScienceModel,
            "client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ):
            with pytest.raises(
                Exception, match="^Error occurred in attempt to perform the operation*"
            ):
                self.mock_model._wait_for_work_request(
                    work_request_id="work_request_id", num_steps=2
                )
                mock_tqdm_update.assert_not_called()

    @patch.object(TqdmProgressBar, "update")
    def test__wait_for_work_request_success(self, mock_tqdm_update, mock_client):
        mock_client.get_work_request = MagicMock(
            return_value=Response(
                data=WorkRequest(id="work_request_id", status="SUCCEEDED"),
                status=None,
                headers={"opc-work-request-id": "work_request_id"},
                request=None,
            )
        )
        mock_client.list_work_request_logs = MagicMock(
            return_value=Response(
                data=[
                    WorkRequestLogEntry(message="test_message_1"),
                    WorkRequestLogEntry(message="test_message_2"),
                ],
                status=None,
                headers=None,
                request=None,
            )
        )
        with patch.object(
            OCIDataScienceModel,
            "client",
            new_callable=PropertyMock,
            return_value=mock_client,
        ):
            self.mock_model._wait_for_work_request(
                work_request_id="work_request_id", num_steps=2
            )
            # mock_tqdm_update.assert_has_calls(
            #     [call("test_message_1"), call("test_message_2")]
            # )
            # assert mock_tqdm_update.call_count == 2
            # mock_tqdm_update.assert_called()
            # assert mock_tqdm_update.call_count == 2

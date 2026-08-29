#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from unittest.mock import MagicMock, patch

import oci
import oci.data_science.models as data_science_models
import oci.pagination
import pytest
from oci.response import Response

from ads.model.datascience_model import DataScienceModel
from ads.model.service.oci_datascience_model import ModelNotSavedError
from ads.model.service.oci_datascience_model_artifact_signature import (
    DataScienceModelArtifactSignature,
)

MODEL_OCID = "ocid1.datasciencemodel.oc1.iad.<unique_ocid>"
COMPARTMENT_OCID = "ocid1.compartment.oc1..<unique_ocid>"
MODEL_ARTIFACT_SIGNATURE_OCID = (
    "ocid1.datasciencemodelartifactsignature.oc1.iad.<unique_ocid>"
)
KMS_KEY_ID = "ocid1.key.oc1.iad.<unique_ocid>"
KMS_KEY_VERSION_ID = "ocid1.keyversion.oc1.iad.<unique_ocid>"
SIGNING_ALGORITHM = "SHA_256_RSA_PKCS_PSS"
SDK_SUPPORTS_MODEL_ARTIFACT_SIGNATURE = all(
    hasattr(data_science_models, name)
    for name in (
        "ChangeModelArtifactSignatureCompartmentDetails",
        "CreateModelArtifactSignatureDetails",
        "UpdateModelArtifactSignatureDetails",
    )
) and all(
    hasattr(oci.data_science.DataScienceClient, name)
    for name in (
        "change_model_artifact_signature_compartment",
        "create_model_artifact_signature",
        "delete_model_artifact_signature",
        "get_model_artifact_signature",
        "list_model_artifact_signatures",
        "update_model_artifact_signature",
        "verify_model_artifact_signature",
    )
)


def test_from_model():
    """Tests constructing a model artifact signature helper from a model."""
    model = MagicMock(
        id=MODEL_OCID,
        compartment_id=COMPARTMENT_OCID,
        config={"key": "value"},
        signer="signer",
        kwargs={"timeout": 10},
    )

    signature = DataScienceModelArtifactSignature.from_model(model)

    assert signature.model_id == MODEL_OCID
    assert signature.compartment_id == COMPARTMENT_OCID
    assert signature.config == {"key": "value"}
    assert signature.signer == "signer"
    assert signature.kwargs == {"timeout": 10}


def test_create_requires_saved_model():
    """Ensures signature operations fail before SDK validation when model id is missing."""
    with pytest.raises(
        ModelNotSavedError,
        match="Model needs to be saved to the Model Catalog before an artifact signature can be created.",
    ):
        DataScienceModelArtifactSignature(model_id=None).create(
            kms_key_id=KMS_KEY_ID,
            kms_key_version_id=KMS_KEY_VERSION_ID,
            signing_algorithm=SIGNING_ALGORITHM,
        )


def test_datascience_model_delegates_to_model_artifact_signature_service():
    """Tests DataScienceModel signature operations delegate to service helper."""
    dsc_model = object.__new__(DataScienceModel)
    dsc_model.dsc_model = MagicMock()
    model_artifact_signature = MagicMock()

    with patch(
        "ads.model.datascience_model.DataScienceModelArtifactSignature.from_model",
        return_value=model_artifact_signature,
    ) as mock_from_model:
        dsc_model.create_model_artifact_signature(
            kms_key_id="kms_key_id",
            kms_key_version_id="kms_key_version_id",
            signing_algorithm="SHA_256_RSA_PKCS_PSS",
            display_name="signature",
        )
        model_artifact_signature.create.assert_called_with(
            kms_key_id="kms_key_id",
            kms_key_version_id="kms_key_version_id",
            signing_algorithm="SHA_256_RSA_PKCS_PSS",
            compartment_id=None,
            display_name="signature",
            freeform_tags=None,
            defined_tags=None,
        )

        dsc_model.list_model_artifact_signatures(display_name="signature")
        model_artifact_signature.list.assert_called_with(
            compartment_id=None,
            display_name="signature",
        )

        dsc_model.get_model_artifact_signature(MODEL_ARTIFACT_SIGNATURE_OCID)
        model_artifact_signature.get.assert_called_with(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

        dsc_model.update_model_artifact_signature(
            MODEL_ARTIFACT_SIGNATURE_OCID,
            display_name="signature",
        )
        model_artifact_signature.update.assert_called_with(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
            display_name="signature",
            freeform_tags=None,
            defined_tags=None,
        )

        dsc_model.delete_model_artifact_signature(MODEL_ARTIFACT_SIGNATURE_OCID)
        model_artifact_signature.delete.assert_called_with(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

        dsc_model.change_model_artifact_signature_compartment(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
            compartment_id="new_compartment_id",
        )
        model_artifact_signature.change_compartment.assert_called_with(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
            compartment_id="new_compartment_id",
        )

        dsc_model.verify_model_artifact_signature(MODEL_ARTIFACT_SIGNATURE_OCID)
        model_artifact_signature.verify.assert_called_with(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

    assert mock_from_model.call_count == 7
    mock_from_model.assert_called_with(dsc_model.dsc_model)


@pytest.mark.skipif(
    not SDK_SUPPORTS_MODEL_ARTIFACT_SIGNATURE,
    reason="OCI SDK does not include model artifact signature APIs.",
)
@patch.object(oci.pagination, "list_call_get_all_results")
def test_model_artifact_signature_operations(
    mock_list_call_get_all_results,
):
    """Tests model artifact signature operations."""
    signature = MagicMock(id=MODEL_ARTIFACT_SIGNATURE_OCID)
    signature_response = Response(
        data=signature, status=None, headers=None, request=None
    )
    empty_response = Response(data=None, status=None, headers=None, request=None)
    list_response = Response(data=[signature], status=None, headers=None, request=None)

    mock_client = MagicMock()
    mock_client.create_model_artifact_signature = MagicMock(
        return_value=signature_response
    )
    mock_list_call_get_all_results.return_value = list_response
    mock_client.list_model_artifact_signatures = MagicMock(return_value=list_response)
    mock_client.get_model_artifact_signature = MagicMock(
        return_value=signature_response
    )
    mock_client.update_model_artifact_signature = MagicMock(
        return_value=signature_response
    )
    mock_client.delete_model_artifact_signature = MagicMock(return_value=empty_response)
    mock_client.change_model_artifact_signature_compartment = MagicMock(
        return_value=empty_response
    )
    mock_client.verify_model_artifact_signature = MagicMock(
        return_value=signature_response
    )

    model_signature = DataScienceModelArtifactSignature(
        model_id=MODEL_OCID,
        compartment_id=COMPARTMENT_OCID,
    )

    with patch.object(DataScienceModelArtifactSignature, "client", mock_client):
        assert (
            model_signature.create(
                display_name="signature",
                kms_key_id=KMS_KEY_ID,
                kms_key_version_id=KMS_KEY_VERSION_ID,
                signing_algorithm=SIGNING_ALGORITHM,
                freeform_tags={"key": "value"},
            )
            == signature
        )
        create_kwargs = mock_client.create_model_artifact_signature.call_args.kwargs
        create_details = create_kwargs["create_model_artifact_signature_details"]
        assert create_kwargs["model_id"] == MODEL_OCID
        assert create_details.compartment_id == COMPARTMENT_OCID
        assert create_details.display_name == "signature"
        assert create_details.kms_key_id == KMS_KEY_ID
        assert create_details.kms_key_version_id == KMS_KEY_VERSION_ID
        assert create_details.signing_algorithm == SIGNING_ALGORITHM
        assert create_details.freeform_tags == {"key": "value"}

        assert model_signature.list(display_name="signature") == [signature]
        mock_list_call_get_all_results.assert_called_with(
            mock_client.list_model_artifact_signatures,
            MODEL_OCID,
            COMPARTMENT_OCID,
            display_name="signature",
        )

        assert model_signature.get(MODEL_ARTIFACT_SIGNATURE_OCID) == signature
        mock_client.get_model_artifact_signature.assert_called_with(
            model_id=MODEL_OCID,
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

        assert (
            model_signature.update(
                MODEL_ARTIFACT_SIGNATURE_OCID,
                display_name="updated-signature",
            )
            == signature
        )
        update_kwargs = mock_client.update_model_artifact_signature.call_args.kwargs
        update_details = update_kwargs["update_model_artifact_signature_details"]
        assert update_kwargs["model_id"] == MODEL_OCID
        assert update_kwargs["artifact_signature_id"] == MODEL_ARTIFACT_SIGNATURE_OCID
        assert update_details.display_name == "updated-signature"

        model_signature.delete(MODEL_ARTIFACT_SIGNATURE_OCID)
        mock_client.delete_model_artifact_signature.assert_called_with(
            model_id=MODEL_OCID,
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

        model_signature.change_compartment(
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
            compartment_id="new_compartment_id",
        )
        change_kwargs = (
            mock_client.change_model_artifact_signature_compartment.call_args.kwargs
        )
        change_details = change_kwargs[
            "change_model_artifact_signature_compartment_details"
        ]
        assert change_kwargs["model_id"] == MODEL_OCID
        assert change_kwargs["artifact_signature_id"] == MODEL_ARTIFACT_SIGNATURE_OCID
        assert change_details.compartment_id == "new_compartment_id"

        assert model_signature.verify(MODEL_ARTIFACT_SIGNATURE_OCID) == signature
        mock_client.verify_model_artifact_signature.assert_called_with(
            model_id=MODEL_OCID,
            artifact_signature_id=MODEL_ARTIFACT_SIGNATURE_OCID,
        )

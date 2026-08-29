#!/usr/bin/env python

# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from functools import wraps
from typing import Callable, Dict, List, Optional

import oci
import oci.data_science
import oci.pagination

from ads.common.oci_datascience import OCIDataScienceMixin
from ads.model.service.oci_datascience_model import ModelNotSavedError

MODEL_ARTIFACT_SIGNATURE_NEEDS_MODEL = (
    "Model needs to be saved to the Model Catalog before it can be accessed."
)
_MODEL_ARTIFACT_SIGNATURE_BASE = getattr(
    oci.data_science.models, "ModelArtifactSignature", object
)


def validate_model_artifact_signature_sdk_support():
    """Validates the installed OCI SDK supports model artifact signature APIs."""
    required_model_classes = (
        "ChangeModelArtifactSignatureCompartmentDetails",
        "CreateModelArtifactSignatureDetails",
        "UpdateModelArtifactSignatureDetails",
    )
    required_client_methods = (
        "change_model_artifact_signature_compartment",
        "create_model_artifact_signature",
        "delete_model_artifact_signature",
        "get_model_artifact_signature",
        "list_model_artifact_signatures",
        "update_model_artifact_signature",
        "verify_model_artifact_signature",
    )
    if not (
        all(hasattr(oci.data_science.models, name) for name in required_model_classes)
        and all(
            hasattr(oci.data_science.DataScienceClient, name)
            for name in required_client_methods
        )
    ):
        raise OSError(
            "Model artifact signature is not supported in the installed OCI SDK."
        )


def check_for_model_id(msg: str = MODEL_ARTIFACT_SIGNATURE_NEEDS_MODEL):
    """Checks that the related model is saved before managing signatures."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.model_id:
                raise ModelNotSavedError(msg)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class DataScienceModelArtifactSignature(
    OCIDataScienceMixin, _MODEL_ARTIFACT_SIGNATURE_BASE
):
    """Client operations for OCI Data Science model artifact signatures."""

    def __init__(
        self,
        model_id: str,
        compartment_id: Optional[str] = None,
        config: dict = None,
        signer: oci.signer.Signer = None,
        client_kwargs: dict = None,
    ) -> None:
        super().__init__(config=config, signer=signer, client_kwargs=client_kwargs)
        self.model_id = model_id
        self.compartment_id = compartment_id

    @classmethod
    def from_model(
        cls, model: oci.data_science.models.Model
    ) -> "DataScienceModelArtifactSignature":
        """Builds a model artifact signature helper from an OCI Data Science model."""
        return cls(
            model_id=model.id,
            compartment_id=model.compartment_id,
            config=getattr(model, "config", None),
            signer=getattr(model, "signer", None),
            client_kwargs=getattr(model, "kwargs", None),
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be created."
    )
    def create(
        self,
        kms_key_id: str,
        kms_key_version_id: str,
        signing_algorithm: str,
        compartment_id: Optional[str] = None,
        display_name: Optional[str] = None,
        freeform_tags: Optional[Dict[str, str]] = None,
        defined_tags: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs: Dict,
    ):
        """Creates a model artifact signature."""
        validate_model_artifact_signature_sdk_support()
        details = oci.data_science.models.CreateModelArtifactSignatureDetails(
            compartment_id=compartment_id or self.compartment_id,
            display_name=display_name,
            kms_key_id=kms_key_id,
            kms_key_version_id=kms_key_version_id,
            signing_algorithm=signing_algorithm,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
        )
        return self.client.create_model_artifact_signature(
            create_model_artifact_signature_details=details,
            model_id=self.model_id,
            **kwargs,
        ).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before artifact signatures can be listed."
    )
    def list(
        self,
        compartment_id: Optional[str] = None,
        **kwargs: Dict,
    ) -> List:
        """Lists model artifact signatures."""
        validate_model_artifact_signature_sdk_support()
        return oci.pagination.list_call_get_all_results(
            self.client.list_model_artifact_signatures,
            self.model_id,
            compartment_id or self.compartment_id,
            **kwargs,
        ).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be read."
    )
    def get(self, artifact_signature_id: str, **kwargs: Dict):
        """Gets a model artifact signature."""
        validate_model_artifact_signature_sdk_support()
        return self.client.get_model_artifact_signature(
            model_id=self.model_id,
            artifact_signature_id=artifact_signature_id,
            **kwargs,
        ).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be updated."
    )
    def update(
        self,
        artifact_signature_id: str,
        display_name: Optional[str] = None,
        freeform_tags: Optional[Dict[str, str]] = None,
        defined_tags: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs: Dict,
    ):
        """Updates a model artifact signature."""
        validate_model_artifact_signature_sdk_support()
        details = oci.data_science.models.UpdateModelArtifactSignatureDetails(
            display_name=display_name,
            freeform_tags=freeform_tags,
            defined_tags=defined_tags,
        )
        return self.client.update_model_artifact_signature(
            update_model_artifact_signature_details=details,
            model_id=self.model_id,
            artifact_signature_id=artifact_signature_id,
            **kwargs,
        ).data

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be deleted."
    )
    def delete(self, artifact_signature_id: str, **kwargs: Dict) -> None:
        """Deletes a model artifact signature."""
        validate_model_artifact_signature_sdk_support()
        self.client.delete_model_artifact_signature(
            model_id=self.model_id,
            artifact_signature_id=artifact_signature_id,
            **kwargs,
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be moved."
    )
    def change_compartment(
        self,
        artifact_signature_id: str,
        compartment_id: str,
        **kwargs: Dict,
    ) -> None:
        """Moves a model artifact signature to another compartment."""
        validate_model_artifact_signature_sdk_support()
        details = oci.data_science.models.ChangeModelArtifactSignatureCompartmentDetails(
            compartment_id=compartment_id,
        )
        self.client.change_model_artifact_signature_compartment(
            change_model_artifact_signature_compartment_details=details,
            model_id=self.model_id,
            artifact_signature_id=artifact_signature_id,
            **kwargs,
        )

    @check_for_model_id(
        msg="Model needs to be saved to the Model Catalog before an artifact signature can be verified."
    )
    def verify(self, artifact_signature_id: str, **kwargs: Dict):
        """Verifies a model artifact signature."""
        validate_model_artifact_signature_sdk_support()
        return self.client.verify_model_artifact_signature(
            model_id=self.model_id,
            artifact_signature_id=artifact_signature_id,
            **kwargs,
        ).data

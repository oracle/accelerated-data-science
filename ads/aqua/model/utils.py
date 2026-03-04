#!/usr/bin/env python
# Copyright (c) 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
"""AQUA model utils"""

from typing import Tuple

from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaValueError
from ads.aqua.common.utils import get_model_by_reference_paths
from ads.aqua.constants import AQUA_FINE_TUNE_MODEL_VERSION
from ads.aqua.finetuning.constants import FineTuneCustomMetadata
from ads.common.object_storage_details import ObjectStorageDetails
from ads.model.datascience_model import DataScienceModel


def extract_base_model_from_ft(aqua_model: DataScienceModel) -> Tuple[str, str]:
    """Extracts the model_name and base model OCID (config_source_id) OCID for a fine-tuned model"""

    config_source_id = aqua_model.custom_metadata_list.get(
        FineTuneCustomMetadata.FINE_TUNE_SOURCE
    ).value
    model_name = aqua_model.custom_metadata_list.get(
        FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME
    ).value

    if not config_source_id or not model_name:
        raise AquaValueError(
            f"Either {FineTuneCustomMetadata.FINE_TUNE_SOURCE} or {FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME} is missing "
            f"from custom metadata for the model {config_source_id}"
        )

    return config_source_id, model_name


def extract_fine_tune_artifacts_path(aqua_model: DataScienceModel) -> Tuple[str, str]:
    """Extracts the fine tuning source (fine_tune_output_path) and base model path from the DataScienceModel Object"""

    is_ft_model_v2 = (
        aqua_model.freeform_tags.get(Tags.AQUA_FINE_TUNE_MODEL_VERSION, "").lower()
        == AQUA_FINE_TUNE_MODEL_VERSION
    )
    base_model_path, fine_tune_output_path = get_model_by_reference_paths(
        aqua_model.model_file_description, is_ft_model_v2
    )

    if not fine_tune_output_path or not ObjectStorageDetails.is_oci_path(
        fine_tune_output_path
    ):
        raise AquaValueError(
            "Fine tuned output path is not available in the model artifact."
        )

    os_path = ObjectStorageDetails.from_path(fine_tune_output_path)
    fine_tune_output_path = os_path.filepath.rstrip("/")

    return base_model_path, fine_tune_output_path

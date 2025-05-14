#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""
aqua.model.entities
~~~~~~~~~~~~~~~~~~~

This module contains dataclasses for Aqua Model.
"""

import re
from dataclasses import InitVar, dataclass, field
from typing import List, Optional

import oci
from huggingface_hub import hf_api
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from ads.aqua import logger
from ads.aqua.app import CLIBuilderMixin
from ads.aqua.common import utils
from ads.aqua.config.utils.serializer import Serializable
from ads.aqua.constants import (
    LIFECYCLE_DETAILS_MISSING_JOBRUN,
    MODEL_FILE_DESCRIPTION_TYPE,
    MODEL_FILE_DESCRIPTION_VERSION,
    UNKNOWN_VALUE,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.model.enums import FineTuningDefinedMetadata
from ads.aqua.training.exceptions import exit_code_dict
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_log_links
from ads.model.datascience_model import DataScienceModel
from ads.model.model_metadata import MetadataTaxonomyKeys


class AquaModelReadme(BaseModel):
    """Represents the response of Get Model Readme."""

    id: str = field(default_factory=str)
    model_card: str = field(default_factory=str)


@dataclass(repr=False)
class FineTuningShapeInfo(DataClassSerializable):
    instance_shape: str = field(default_factory=str)
    replica: int = field(default_factory=int)


# TODO: give a better name
@dataclass(repr=False)
class AquaFineTuneValidation(DataClassSerializable):
    type: str = "Automatic split"
    value: str = ""


@dataclass(repr=False)
class ModelValidationResult:
    model_file: Optional[str] = None
    model_formats: List[str] = field(default_factory=list)
    telemetry_model_name: str = None
    tags: Optional[dict] = None


@dataclass(repr=False)
class AquaFineTuningMetric(DataClassSerializable):
    name: str = field(default_factory=str)
    category: str = field(default_factory=str)
    scores: list = field(default_factory=list)


@dataclass(repr=False)
class AquaModelLicense(DataClassSerializable):
    """Represents the response of Get Model License."""

    id: str = field(default_factory=str)
    license: str = field(default_factory=str)


@dataclass(repr=False)
class AquaModelSummary(DataClassSerializable):
    """Represents a summary of Aqua model."""

    compartment_id: str = None
    icon: str = None
    id: str = None
    is_fine_tuned_model: bool = None
    license: str = None
    name: str = None
    organization: str = None
    project_id: str = None
    tags: dict = None
    task: str = None
    time_created: str = None
    console_link: str = None
    search_text: str = None
    ready_to_deploy: bool = True
    ready_to_finetune: bool = False
    ready_to_import: bool = False
    nvidia_gpu_supported: bool = False
    arm_cpu_supported: bool = False
    model_file: Optional[str] = None
    model_formats: List[str] = field(default_factory=list)


@dataclass(repr=False)
class AquaModel(AquaModelSummary, DataClassSerializable):
    """Represents an Aqua model."""

    inference_container: str = None
    inference_container_uri: str = None
    finetuning_container: str = None
    evaluation_container: str = None
    artifact_location: str = None


@dataclass(repr=False)
class HFModelContainerInfo:
    """Container defauls for model"""

    inference_container: str = None
    finetuning_container: str = None


@dataclass(repr=False)
class HFModelSummary:
    """Represents a summary of Hugging Face model."""

    model_info: hf_api.ModelInfo = field(default_factory=hf_api.ModelInfo)
    aqua_model_info: Optional[AquaModelSummary] = field(
        default_factory=AquaModelSummary
    )


@dataclass(repr=False)
class AquaEvalFTCommon(DataClassSerializable):
    """Represents common fields for evaluation and fine-tuning."""

    lifecycle_state: str = None
    lifecycle_details: str = None
    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log_group: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    log: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)

    model: InitVar = None
    region: InitVar = None
    jobrun: InitVar = None

    def __post_init__(
        self, model, region: str, jobrun: oci.data_science.models.JobRun = None
    ):
        try:
            log_id = jobrun.log_details.log_id
        except Exception as e:
            logger.debug(f"No associated log found. {str(e)}")
            log_id = ""

        try:
            loggroup_id = jobrun.log_details.log_group_id
        except Exception as e:
            logger.debug(f"No associated loggroup found. {str(e)}")
            loggroup_id = ""

        loggroup_url = get_log_links(region=region, log_group_id=loggroup_id)
        log_url = (
            get_log_links(
                region=region,
                log_group_id=loggroup_id,
                log_id=log_id,
                compartment_id=jobrun.compartment_id,
                source_id=jobrun.id,
            )
            if jobrun
            else ""
        )

        log_name = None
        loggroup_name = None

        if log_id:
            try:
                log = utils.query_resource(log_id, return_all=False)
                log_name = log.display_name if log else ""
            except Exception:
                pass

        if loggroup_id:
            try:
                loggroup = utils.query_resource(loggroup_id, return_all=False)
                loggroup_name = loggroup.display_name if loggroup else ""
            except Exception:
                pass

        experiment_id, experiment_name = utils._get_experiment_info(model)

        self.log_group = AquaResourceIdentifier(
            loggroup_id, loggroup_name, loggroup_url
        )
        self.log = AquaResourceIdentifier(log_id, log_name, log_url)
        self.experiment = utils._build_resource_identifier(
            id=experiment_id, name=experiment_name, region=region
        )
        self.job = utils._build_job_identifier(job_run_details=jobrun, region=region)
        self.lifecycle_details = (
            LIFECYCLE_DETAILS_MISSING_JOBRUN if not jobrun else jobrun.lifecycle_details
        )


@dataclass(repr=False)
class AquaFineTuneModel(AquaModel, AquaEvalFTCommon, DataClassSerializable):
    """Represents an Aqua Fine Tuned Model."""

    dataset: str = field(default_factory=str)
    validation: AquaFineTuneValidation = field(default_factory=AquaFineTuneValidation)
    shape_info: FineTuningShapeInfo = field(default_factory=FineTuningShapeInfo)
    metrics: List[AquaFineTuningMetric] = field(default_factory=list)

    def __post_init__(
        self,
        model: DataScienceModel,
        region: str,
        jobrun: oci.data_science.models.JobRun = None,
    ):
        super().__post_init__(model=model, region=region, jobrun=jobrun)

        if jobrun is not None:
            jobrun_env_vars = (
                jobrun.job_configuration_override_details.environment_variables or {}
            )
            self.shape_info = FineTuningShapeInfo(
                instance_shape=jobrun.job_infrastructure_configuration_details.shape_name,
                # TODO: use variable for `NODE_COUNT` in ads/jobs/builders/runtimes/base.py
                replica=jobrun_env_vars.get("NODE_COUNT", UNKNOWN_VALUE),
            )

        try:
            model_hyperparameters = model.defined_metadata_list.get(
                MetadataTaxonomyKeys.HYPERPARAMETERS
            ).value
        except Exception as e:
            logger.debug(
                f"Failed to extract model hyperparameters from {model.id}: " f"{str(e)}"
            )
            model_hyperparameters = {}

        self.dataset = model_hyperparameters.get(
            FineTuningDefinedMetadata.TRAINING_DATA
        )
        if not self.dataset:
            logger.debug(
                f"Key={FineTuningDefinedMetadata.TRAINING_DATA} not found in model hyperparameters."
            )

        self.validation = AquaFineTuneValidation(
            value=model_hyperparameters.get(FineTuningDefinedMetadata.VAL_SET_SIZE)
        )
        if not self.validation:
            logger.debug(
                f"Key={FineTuningDefinedMetadata.VAL_SET_SIZE} not found in model hyperparameters."
            )

        if self.lifecycle_details:
            self.lifecycle_details = self._extract_job_lifecycle_details(
                self.lifecycle_details
            )

    def _extract_job_lifecycle_details(self, lifecycle_details):
        message = lifecycle_details
        try:
            # Extract exit code
            match = re.search(r"exit code (\d+)", lifecycle_details)
            if match:
                exit_code = int(match.group(1))
                if exit_code == 1:
                    return message
                # Match exit code to message
                exception = exit_code_dict().get(
                    exit_code,
                    lifecycle_details,
                )
                message = f"{exception.reason} (exit code {exit_code})"
        except Exception:
            pass

        return message


@dataclass
class ImportModelDetails(CLIBuilderMixin):
    model: str
    os_path: str
    download_from_hf: Optional[bool] = True
    local_dir: Optional[str] = None
    cleanup_model_cache: Optional[bool] = False
    inference_container: Optional[str] = None
    finetuning_container: Optional[str] = None
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    model_file: Optional[str] = None
    inference_container_uri: Optional[str] = None
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: Optional[List[str]] = None
    freeform_tags: Optional[dict] = None
    defined_tags: Optional[dict] = None
    ignore_model_artifact_check: Optional[bool] = None

    def __post_init__(self):
        self._command = "model register"


class ModelFileInfo(Serializable):
    """Describes the file information of this model.

    Attributes:
        name (str): The name of the model artifact file.
        version (str): The version of the model artifact file.
        size_in_bytes (int): The size of the model artifact file in bytes.
    """

    name: str = Field(..., description="The name of model artifact file.")
    version: str = Field(..., description="The version of model artifact file.")
    size_in_bytes: int = Field(
        ..., description="The size of model artifact file in bytes."
    )

    class Config:
        alias_generator = to_camel
        extra = "allow"


class ModelArtifactInfo(Serializable):
    """Describes the artifact information of this model.

    Attributes:
        namespace (str): The namespace of the model artifact location.
        bucket_name (str): The bucket name of model artifact location.
        prefix (str): The prefix of model artifact location.
        objects: (List[ModelFileInfo]): A list of model artifact objects.
    """

    namespace: str = Field(
        ..., description="The name space of model artifact location."
    )
    bucket_name: str = Field(
        ..., description="The bucket name of model artifact location."
    )
    prefix: str = Field(..., description="The prefix of model artifact location.")
    objects: List[ModelFileInfo] = Field(
        ..., description="List of model artifact objects."
    )

    class Config:
        alias_generator = to_camel
        extra = "allow"


class ModelFileDescription(Serializable):
    """Describes the model file description.

    Attributes:
        version (str): The version of the model file description. Defaults to `1.0`.
        type (str): The type of model file description. Defaults to `modelOSSReferenceDescription`.
        models List[ModelArtifactInfo]: A list of model artifact information.
    """

    version: str = Field(
        default=MODEL_FILE_DESCRIPTION_VERSION,
        description="The version of model file description.",
    )
    type: str = Field(
        default=MODEL_FILE_DESCRIPTION_TYPE,
        description="The type of model file description.",
    )
    models: List[ModelArtifactInfo] = Field(
        ..., description="List of model artifact information."
    )

    class Config:
        alias_generator = to_camel
        extra = "allow"

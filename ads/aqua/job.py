#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


from enum import Enum
import logging
from dataclasses import dataclass
import os
import tempfile
from typing import List, Optional
from ads.aqua.base import AquaApp
from ads.aqua.exception import AquaValueError
from ads.aqua.utils import create_model_catalog, create_model_version_set, get_source, is_valid_ocid, upload_file_to_os
from ads.common.object_storage_details import ObjectStorageDetails
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.model.model_metadata import MetadataTaxonomyKeys, ModelCustomMetadata, ModelTaxonomyMetadata

from oci.data_science.models import (
    Metadata,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)

logger = logging.getLogger(__name__)


@dataclass
class AquaJobSummary:
    """Represents an Aqua job summary."""

    id: str
    compartment_id: str
    project_id: str

    model_id: str
    task: str


@dataclass
class AquaJob(AquaJobSummary):
    """Represents an Aqua job."""

    dataset: str


@dataclass
class AquaFineTuningJob(AquaJob):
    """Represents an Aqua fine-tuning job."""

    epoch: int


class FTCustomMetadata(Enum):
    FINE_TUNE_SOURCE = "fine_tune_source"
    FINE_TUNE_SOURCE_NAME = "fine_tune_source_name"
    FINE_TUNE_OUTPUT_PATH = "fine_tune_output_path"
    FINE_TUNE_JOB_ID = "fine_tune_job_id"
    FINE_TUNE_JOB_RUN_ID = "fine_tune_job_run_id"
    FINE_TUNE_TOTAL_STEPS = "total_steps"
    TRAINING_METRICS_FINAL = "training_metrics_final"
    VALIDATION_METRICS_FINAL = "validation_metrics_final"
    TRAINING_METRICS_X = "training_metrics_x.xx"
    VALIDATION_METRICS_X = "validation_metrics_x.xx"


class FTJobTags(Enum):
    OCI_AQUA = "OCI_AQUA"
    AQUA_FINE_TUNED_MODEL = "aqua_fine_tuned_model"


@dataclass(repr=False)
class CreateFineTuningDetails():
    ft_source_id: str
    ft_name: str
    ft_description: str
    compartment_id: str
    project_id: str
    dataset_path: str
    report_path: str
    finetuned_model_path: str # ?
    validation_split: float # ?
    experiment_id: str
    experiment_name: str
    experiment_description: str
    ft_parameters: dict # can pass through env
    shape_name: str
    replica: int # in runtime
    log_id: str
    log_group_id: str
    subnet_id: str
    block_storage_size: Optional[int] = 256




class AquaJobApp(AquaApp):
    """Contains APIs for Aqua jobs."""


class AquaFineTuningApp(AquaApp):
    """Contains APIs for Aqua fine-tuning jobs."""

    def get(self, job_id) -> AquaFineTuningJob:
        """Gets the information of an Aqua model."""
        return AquaFineTuningJob(
            id=job_id,
            compartment_id="ocid.compartment.xxx",
            project_id="ocid.project.xxx",
            model_id="ocid.model.xxx",
            task="fine-tuning",
            dataset="dummy",
            epoch=2,
        )

    def list(self, compartment_id, project_id=None, **kwargs) -> List[AquaJobSummary]:
        """Lists Aqua models."""
        return [
            AquaJobSummary(
                id=f"ocid{i}.xxx",
                compartment_id=compartment_id,
                project_id=project_id,
                model_id="ocid.model.xxx",
                task="fine-tuning",
            )
            for i in range(5)
        ]
    
    def create(self, create_fine_tuning_details: CreateFineTuningDetails, **kwargs):
        source = get_source(create_fine_tuning_details.ft_source_id)

        if not ObjectStorageDetails.is_oci_path(
            create_fine_tuning_details.report_path
        ):
            raise AquaValueError(
                "Evaluation report path must be an object storage path."
            )

        target_compartment = (
            create_fine_tuning_details.compartment_id or COMPARTMENT_OCID
        )
        target_project = create_fine_tuning_details.project_id or PROJECT_OCID

        experiment_model_version_set_id = create_fine_tuning_details.experiment_id
        experiment_model_version_set_name = (
            create_fine_tuning_details.experiment_name
        )

        if (
            not experiment_model_version_set_id
            and not experiment_model_version_set_name
        ):
            raise AquaValueError(
                "Either experiment id or experiment name must be provided."
            )
        
        (
            experiment_model_version_set_id,
            experiment_model_version_set_name
        ) = create_model_version_set(
            model_version_set_id=experiment_model_version_set_id,
            model_version_set_name=experiment_model_version_set_name,
            description=create_fine_tuning_details.experiment_description,
            compartment_id=target_compartment,
            project_id=target_project
        )

        ft_model_custom_metadata = ModelCustomMetadata()
        ft_model_custom_metadata.add(
            key=FTCustomMetadata.FINE_TUNE_SOURCE.value,
            value=create_fine_tuning_details.ft_source_id,
        )
        ft_model_custom_metadata.add(
            key=FTCustomMetadata.FINE_TUNE_OUTPUT_PATH.value,
            value=create_fine_tuning_details.report_path,
        )
        ft_model_custom_metadata.add(
            key=FTCustomMetadata.FINE_TUNE_SOURCE_NAME.value,
            value=source.display_name,
        )
        # ft_model_custom_metadata.add(
        #     key=FTCustomMetadata.FINE_TUNE_TOTAL_STEPS.value,
        #     value="", # ?
        # )

        ft_model_taxonomy_metadata = ModelTaxonomyMetadata()
        ft_model_taxonomy_metadata[
            MetadataTaxonomyKeys.HYPERPARAMETERS
        ].value = create_fine_tuning_details.ft_parameters

        ft_model = create_model_catalog(
            display_name=create_fine_tuning_details.ft_name,
            description=create_fine_tuning_details.ft_description,
            model_version_set_id=experiment_model_version_set_id,
            model_custom_metadata=ft_model_custom_metadata,
            model_taxonomy_metadata=ft_model_taxonomy_metadata,
            compartment_id=target_compartment,
            project_id=target_project
        )

        # upload dataset if it's local path
        ft_dataset_path = create_fine_tuning_details.dataset_path
        if not ObjectStorageDetails.is_oci_path(ft_dataset_path):
            # format: oci://<bucket>@<namespace>/<dataset_file_name>
            dst_uri = f"{create_fine_tuning_details.report_path}/{os.path.basename(ft_dataset_path)}"
            upload_file_to_os(
                src_uri=ft_dataset_path,
                dst_uri=dst_uri,
                auth=self._auth,
                force_overwrite=False,
            )
            logger.debug(
                f"Uploaded local file {ft_dataset_path} to object storage {dst_uri}."
            )
            ft_dataset_path = dst_uri

        ft_job_freeform_tags = {
            FTJobTags.OCI_AQUA.value: FTJobTags.OCI_AQUA.value,
            FTJobTags.AQUA_FINE_TUNED_MODEL.value: f"{source.id}#{source.display_name}", # discuss with ming
        }

        try:
            with tempfile.TemporaryDirectory() as temp_directory:
                ft_job = Job(
                    name=ft_model.display_name
                ).with_infrastructure(
                    DataScienceJob()
                    .with_log_group_id(create_fine_tuning_details.log_group_id)
                    .with_log_id(create_fine_tuning_details.log_id)
                    .with_compartment_id(target_compartment)
                    .with_project_id(target_project)
                    .with_shape_name(create_fine_tuning_details.shape_name)
                    .with_block_storage_size(
                        create_fine_tuning_details.block_storage_size
                    ) 
                    .with_freeform_tag(**ft_job_freeform_tags)
                    #.with_subnet_id(create_fine_tuning_details.subnet_id) # check if replica is larger than 1
                )
                
                ft_job.with_runtime(
                    self._build_evaluation_runtime(
                        evaluation_id=evaluation_model.id,
                        evaluation_source_id=(
                            create_aqua_evaluation_details.evaluation_source_id
                        ),
                        dataset_path=evaluation_dataset_path,
                        report_path=create_aqua_evaluation_details.report_path,
                        model_parameters=create_aqua_evaluation_details.model_parameters,
                        metrics=create_aqua_evaluation_details.metrics,
                        source_folder=temp_directory,
                    )
                ).create(
                    **kwargs
                )  ## TODO: decide what parameters will be needed
                logger.debug(
                    f"Successfully created fine tuning job {ft_job.id} for {create_fine_tuning_details.ft_source_id}."
                )

                ft_job_run = ft_job.run(
                    name=ft_model.display_name,
                    freeform_tags=ft_job_freeform_tags,
                    wait=False,
                )
                logger.debug(
                    f"Successfully created fine tuning job run {ft_job_run.id} for {create_fine_tuning_details.ft_source_id}."
                )

                self.ds_client.update_model(
                    model_id=ft_model.id,
                    update_model_details=UpdateModelDetails(
                        freeform_tags={
                            # aqua_fine_tuning
                        }
                    ),
                )
        except:
            # TODO: quick fix, revisit this later
            self.ds_client.update_model(
                model_id=ft_model.id,
                update_model_details=UpdateModelDetails(
                    freeform_tags={
                        # ft_model id?
                    }
                ),
            )
            raise

        ft_model_custom_metadata.add(
            key=FTCustomMetadata.FINE_TUNE_JOB_ID.value,
            value=ft_job.id,
        )
        ft_model_custom_metadata.add(
            key=FTCustomMetadata.FINE_TUNE_JOB_RUN_ID.value,
            value=ft_job_run.id,
        )
        updated_custom_metadata_list = [
            Metadata(**metadata)
            for metadata in ft_model_custom_metadata.to_dict()["data"]
        ]

        self.ds_client.update_model(
            model_id=ft_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list
            ),
        )

        self.ds_client.update_model_provenance(
            model_id=ft_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=ft_job_run.id
            ),
        )

        return None

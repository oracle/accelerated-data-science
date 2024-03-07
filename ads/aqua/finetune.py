#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import asdict, dataclass, field
from enum import Enum
import os
from typing import Optional
from ads.aqua.exception import AquaValueError
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.serializer import DataClassSerializable
from ads.aqua.base import AquaApp
from ads.aqua.job import AquaJobSummary
from ads.aqua.data import Resource, AquaResourceIdentifier, Tags
from ads.common.utils import get_console_link
from ads.aqua.utils import (
    DEFAULT_FT_BLOCK_STORAGE_SIZE,
    DEFAULT_REPLICA,
    FINE_TUNING_RUNTIME_CONTAINER,
    UNKNOWN,
    logger,
    create_model_catalog,
    create_model_version_set,
    get_source,
    upload_file_to_os
)
from ads.config import COMPARTMENT_OCID, PROJECT_OCID
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.jobs.builders.runtimes.base import Runtime
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.model.model_metadata import (
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelTaxonomyMetadata
)

from oci.data_science.models import (
    Metadata,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)


class FineTuneCustomMetadata(Enum):
    FINE_TUNE_SOURCE = "fine_tune_source"
    FINE_TUNE_SOURCE_NAME = "fine_tune_source_name"
    FINE_TUNE_OUTPUT_PATH = "fine_tune_output_path"
    FINE_TUNE_JOB_ID = "fine_tune_job_id"
    FINE_TUNE_JOB_RUN_ID = "fine_tune_job_run_id"


@dataclass(repr=False)
class AquaFineTuningParams(DataClassSerializable):
    epochs: int = None
    learning_rate: float = None


@dataclass(repr=False)
class AquaFineTuningSummary(AquaJobSummary, DataClassSerializable):
    parameters: AquaFineTuningParams = field(default_factory=AquaFineTuningParams)


@dataclass(repr=False)
class CreateFineTuningDetails(DataClassSerializable):
    ft_source_id: str
    ft_name: str
    dataset_path: str
    report_path: str
    validation_split: float
    ft_parameters: dict # TODO: revisit to allow pass through env
    shape_name: str
    replica: int
    ft_description: Optional[str] = None
    compartment_id: Optional[str] = None
    project_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_description: Optional[str] = None
    block_storage_size: Optional[int] = None
    subnet_id: Optional[str] = None
    log_id: Optional[str] = None
    log_group_id: Optional[str] = None


class AquaFineTuningApp(AquaApp):
    """Contains APIs for Aqua fine-tuning jobs."""

    def create(
        self,
        create_fine_tuning_details: CreateFineTuningDetails,
        **kwargs
    ) -> "AquaFineTuningSummary":
        """Creates a aqua fine tuned model."""

        # todo : parse kwargs and convert to CreateAquaFineTuneDetails object
        #   with CreateAquaFineTuneDetails(**kwargs)

        source = get_source(create_fine_tuning_details.ft_source_id)

        if not ObjectStorageDetails.is_oci_path(
            create_fine_tuning_details.report_path
        ):
            raise AquaValueError(
                "Fine tuning report path must be an object storage path."
            )

        target_compartment = (
            create_fine_tuning_details.compartment_id or COMPARTMENT_OCID
        )
        target_project = create_fine_tuning_details.project_id or PROJECT_OCID

        ft_parameters = None
        try:
            ft_parameters = AquaFineTuningParams(
                **create_fine_tuning_details.ft_parameters,
            )
        except:
            raise AquaValueError(
                "Invalid model parameters. Model parameters should "
                f"be a dictionary with keys: {', '.join(list(asdict(AquaFineTuningParams).keys()))}."
            )

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
            key=FineTuneCustomMetadata.FINE_TUNE_SOURCE.value,
            value=create_fine_tuning_details.ft_source_id,
        )
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_OUTPUT_PATH.value,
            value=create_fine_tuning_details.report_path,
        )
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME.value,
            value=source.display_name,
        )

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

        ft_job_freeform_tags = {
            Tags.AQUA_TAG.value: UNKNOWN,
            Tags.AQUA_FINE_TUNED_MODEL_TAG.value: f"{source.id}#{source.display_name}",
        }

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
                create_fine_tuning_details.block_storage_size or DEFAULT_FT_BLOCK_STORAGE_SIZE
            ) 
            .with_freeform_tag(**ft_job_freeform_tags)
        )

        if create_fine_tuning_details.replica > DEFAULT_REPLICA:
            ft_job.infrastructure.with_subnet_id(
                create_fine_tuning_details.subnet_id
            )
        
        ft_job.with_runtime(
            self._build_fine_tuning_runtime(
                source_id=source.id,
                dataset_path=ft_dataset_path,
                report_path=create_fine_tuning_details.report_path,
                replica=create_fine_tuning_details.replica,
                parameters=ft_parameters
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

        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_JOB_ID.value,
            value=ft_job.id,
        )
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_JOB_RUN_ID.value,
            value=ft_job_run.id,
        )
        updated_custom_metadata_list = [
            Metadata(**metadata)
            for metadata in ft_model_custom_metadata.to_dict()["data"]
        ]

        self.ds_client.update_model(
            model_id=ft_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list,
                freeform_tags={
                    Tags.AQUA_TAG.value: UNKNOWN
                }
            ),
        )

        self.ds_client.update_model_provenance(
            model_id=ft_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=ft_job_run.id
            ),
        )

        return AquaFineTuningSummary(
            id=ft_model.id,
            name=ft_model.display_name,
            console_url=get_console_link(
                resource=Resource.MODEL.value,
                ocid=ft_model.id,
                region=self.region,
            ),
            time_created=str(ft_model.time_created),
            lifecycle_state=ft_job_run.lifecycle_state or UNKNOWN,
            lifecycle_details=ft_job_run.lifecycle_details or UNKNOWN,
            experiment=AquaResourceIdentifier(
                id=experiment_model_version_set_id,
                name=experiment_model_version_set_name,
                url=get_console_link(
                    resource=Resource.MODEL_VERSION_SET.value,
                    ocid=experiment_model_version_set_id,
                    region=self.region,
                ),
            ),
            source=AquaResourceIdentifier(
                id=source.id,
                name=source.display_name,
                url=get_console_link(
                    resource=Resource.MODEL.value,
                    ocid=source.id,
                    region=self.region,
                ),
            ),
            job=AquaResourceIdentifier(
                id=ft_job.id,
                name=ft_job.name,
                url=get_console_link(
                    resource=Resource.JOB.value,
                    ocid=ft_job.id,
                    region=self.region,
                ),
            ),
            tags=dict(
                aqua_finetuning=Tags.AQUA_FINE_TUNING.value,
                finetuning_job_id=ft_job.id,
                finetuning_source=source.id,
                finetuning_experiment_id=experiment_model_version_set_id,
            ),
            parameters=ft_parameters,
        )
    
    def _build_fine_tuning_runtime(
        self,
        source_id: str,
        dataset_path: str,
        report_path: str,
        replica: int,
        parameters: AquaFineTuningParams,
    ) -> Runtime:
        """Builds fine tuning runtime for Job."""
        runtime = (
            ContainerRuntime()
            .with_environment_variable(
                **{
                    "BASE_MODEL": source_id,
                    "CONTAINER_CUSTOM_IMAGE": FINE_TUNING_RUNTIME_CONTAINER,
                    "OCI_LOG_LEVEL": "DEBUG",
                    "OCI__LAUNCH_CMD": f"--base_model $BASE_MODEL --micro_batch_size 4 --num_epochs {parameters.epochs} --learning_rate {parameters.learning_rate} --training_data {dataset_path} --output_dir {report_path}",
                    "OCI__METRICS_NAMESPACE": "qq_job_runs"
                }
            )
            .with_image(
                image=FINE_TUNING_RUNTIME_CONTAINER
            )
            .with_replica(replica)
        )

        return runtime

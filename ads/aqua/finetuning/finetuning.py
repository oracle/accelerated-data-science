#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import os
import time
import traceback
from typing import Dict

from oci.data_science.models import (
    Metadata,
    UpdateModelDetails,
    UpdateModelProvenanceDetails,
)
from pydantic import ValidationError

from ads.aqua import logger
from ads.aqua.app import AquaApp
from ads.aqua.common.enums import Resource, Tags
from ads.aqua.common.errors import AquaFileExistsError, AquaValueError
from ads.aqua.common.utils import (
    DEFINED_METADATA_TO_FILE_MAP,
    build_pydantic_error_message,
    upload_local_to_os,
)
from ads.aqua.constants import (
    AQUA_FINE_TUNE_MODEL_VERSION,
    DEFAULT_FT_BATCH_SIZE,
    DEFAULT_FT_BLOCK_STORAGE_SIZE,
    DEFAULT_FT_REPLICA,
    DEFAULT_FT_VALIDATION_SET_SIZE,
    JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING,
    UNKNOWN_DICT,
)
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.finetuning.constants import (
    ENV_AQUA_FINE_TUNING_CONTAINER,
    FineTuneCustomMetadata,
)
from ads.aqua.finetuning.entities import (
    AquaFineTuningConfig,
    AquaFineTuningParams,
    AquaFineTuningSummary,
    CreateFineTuningDetails,
)
from ads.aqua.model.constants import AquaModelMetadataKeys
from ads.common.auth import default_signer
from ads.common.object_storage_details import ObjectStorageDetails
from ads.common.utils import UNKNOWN, get_console_link
from ads.config import (
    AQUA_FINETUNING_CONTAINER_OVERRIDE_FLAG_METADATA_NAME,
    AQUA_JOB_SUBNET_ID,
    COMPARTMENT_OCID,
    CONDA_BUCKET_NS,
    PROJECT_OCID,
)
from ads.jobs.ads_job import Job
from ads.jobs.builders.infrastructure.dsc_job import DataScienceJob
from ads.jobs.builders.runtimes.base import Runtime
from ads.jobs.builders.runtimes.container_runtime import ContainerRuntime
from ads.model.common.utils import MetadataArtifactPathType
from ads.model.model_metadata import (
    MetadataTaxonomyKeys,
    ModelCustomMetadata,
    ModelTaxonomyMetadata,
)
from ads.telemetry import telemetry


class AquaFineTuningApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua fine-tuned models within the Oracle
    Cloud Infrastructure Data Science service, serving as an interface for creating fine-tuned models.

    Methods
    -------
    create(...) -> AquaFineTuningSummary
        Creates a fine-tuned Aqua model.
    get_finetuning_config(self, model_id: str) -> Dict:
        Gets the finetuning config for given Aqua model.

    Note:
        Use `ads aqua finetuning <method_name> --help` to get more details on the parameters available.
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    @telemetry(entry_point="plugin=finetuning&action=create", name="aqua")
    def create(
        self, create_fine_tuning_details: CreateFineTuningDetails = None, **kwargs
    ) -> "AquaFineTuningSummary":
        """Creates Aqua fine tuning for model.\n
        For detailed information about CLI flags see: https://github.com/oracle-samples/oci-data-science-ai-samples/blob/f271ca63d12e3c256718f23a14d93da4b4fc086b/ai-quick-actions/cli-tips.md#create-fine-tuned-model

        Parameters
        ----------
        create_fine_tuning_details: CreateFineTuningDetails
            The CreateFineTuningDetails data class which contains all
            required and optional fields to create the aqua fine tuning.
            kwargs:
                ft_source_id: str The fine tuning source id. Must be model OCID.
                ft_name: str
                    The name for fine tuning.
                dataset_path: str
                    The dataset path for fine tuning. Could be either a local path from notebook session
                    or an object storage path.
                report_path: str
                    The report path for fine tuning. Must be an object storage path.
                ft_parameters: dict
                    The parameters for fine tuning.
                shape_name: str
                    The shape name for fine tuning job infrastructure.
                replica: int
                    The replica for fine tuning job runtime.
                validation_set_size: float
                    The validation set size for fine tuning job. Must be a float in between [0,1).
                ft_description: (str, optional). Defaults to `None`.
                    The description for fine tuning.
                compartment_id: (str, optional). Defaults to `None`.
                    The compartment id for fine tuning.
                project_id: (str, optional). Defaults to `None`.
                    The project id for fine tuning.
                experiment_id: (str, optional). Defaults to `None`.
                    The fine tuning model version set id. If provided,
                    fine tuning model will be associated with it.
                experiment_name: (str, optional). Defaults to `None`.
                    The fine tuning model version set name. If provided,
                    the fine tuning version set with the same name will be used if exists,
                    otherwise a new model version set will be created with the name.
                experiment_description: (str, optional). Defaults to `None`.
                    The description for fine tuning model version set.
                block_storage_size: (int, optional). Defaults to 256.
                    The storage for fine tuning job infrastructure.
                subnet_id: (str, optional). Defaults to `None`.
                    The custom egress for fine tuning job.
                log_group_id: (str, optional). Defaults to `None`.
                    The log group id for fine tuning job infrastructure.
                log_id: (str, optional). Defaults to `None`.
                    The log id for fine tuning job infrastructure.
                watch_logs: (bool, optional). Defaults to `False`.
                    The flag to watch the job run logs when a fine-tuning job is created.
                force_overwrite: (bool, optional). Defaults to `False`.
                    Whether to force overwrite the existing file in object storage.
                freeform_tags: (dict, optional)
                    Freeform tags for the fine-tuning model
                defined_tags: (dict, optional)
                    Defined tags for the fine-tuning model
        kwargs:
            The kwargs for creating CreateFineTuningDetails instance if
            no create_fine_tuning_details provided.

        Returns
        -------
        AquaFineTuningSummary:
            The instance of AquaFineTuningSummary.
        """
        if not create_fine_tuning_details:
            try:
                create_fine_tuning_details = CreateFineTuningDetails(**kwargs)
            except ValidationError as ex:
                custom_errors = build_pydantic_error_message(ex)
                raise AquaValueError(
                    f"Invalid parameters for creating a fine-tuned model. Error details: {custom_errors}."
                ) from ex

        target_compartment = (
            create_fine_tuning_details.compartment_id or COMPARTMENT_OCID
        )
        target_project = create_fine_tuning_details.project_id or PROJECT_OCID

        if not ObjectStorageDetails.is_oci_path(create_fine_tuning_details.report_path):
            raise AquaValueError(
                "Fine tuning report path must be an object storage path."
            )

        if (
            create_fine_tuning_details.validation_set_size < 0
            or create_fine_tuning_details.validation_set_size >= 1
        ):
            raise AquaValueError(
                "Fine tuning validation set size should be a float number in between [0, 1)."
            )

        if create_fine_tuning_details.replica < DEFAULT_FT_REPLICA:
            raise AquaValueError(
                f"Fine tuning replica must be equal to or larger than {DEFAULT_FT_REPLICA}."
            )

        subnet_id = create_fine_tuning_details.subnet_id or AQUA_JOB_SUBNET_ID
        if not subnet_id and create_fine_tuning_details.replica > DEFAULT_FT_REPLICA:
            raise AquaValueError(
                f"Custom egress must be provided if replica is larger than {DEFAULT_FT_REPLICA}. "
                "Specify the subnet id via API or environment variable AQUA_JOB_SUBNET_ID."
            )

        if create_fine_tuning_details.replica > DEFAULT_FT_REPLICA and not (
            create_fine_tuning_details.log_id
            and create_fine_tuning_details.log_group_id
        ):
            raise AquaValueError(
                f"Logging is required for fine tuning if replica is larger than {DEFAULT_FT_REPLICA}."
            )

        if create_fine_tuning_details.watch_logs and not (
            create_fine_tuning_details.log_id
            and create_fine_tuning_details.log_group_id
        ):
            raise AquaValueError(
                "Logging is required for fine tuning if watch_logs is set to True. "
                "Please provide log_id and log_group_id with the request parameters."
            )

        ft_parameters = self._get_finetuning_params(
            create_fine_tuning_details.ft_parameters
        )

        experiment_model_version_set_id = create_fine_tuning_details.experiment_id
        experiment_model_version_set_name = create_fine_tuning_details.experiment_name

        if (
            not experiment_model_version_set_id
            and not experiment_model_version_set_name
        ):
            raise AquaValueError(
                "Either experiment id or experiment name must be provided for fine tuning."
            )

        # upload dataset if it's local path
        ft_dataset_path = create_fine_tuning_details.dataset_path
        if not ObjectStorageDetails.is_oci_path(ft_dataset_path):
            # format: oci://<bucket>@<namespace>/<dataset_file_name>
            dataset_file = os.path.basename(ft_dataset_path)
            dst_uri = (
                f"{create_fine_tuning_details.report_path.rstrip('/')}/{dataset_file}"
            )
            try:
                upload_local_to_os(
                    src_uri=ft_dataset_path,
                    dst_uri=dst_uri,
                    auth=default_signer(),
                    force_overwrite=create_fine_tuning_details.force_overwrite,
                )
            except FileExistsError as fe:
                raise AquaFileExistsError(
                    f"Dataset {dataset_file} already exists in {create_fine_tuning_details.report_path}. "
                    "Please use a new dataset file name, report path or set `force_overwrite` as True."
                ) from fe
            logger.debug(
                f"Uploaded local file {ft_dataset_path} to object storage {dst_uri}."
            )
            # tracks the size of dataset uploaded by user to the destination.
            self.telemetry.record_event_async(
                category="aqua/finetune/upload",
                action="size",
                detail=os.path.getsize(os.path.expanduser(ft_dataset_path)),
            )
            ft_dataset_path = dst_uri

        (
            experiment_model_version_set_id,
            experiment_model_version_set_name,
        ) = self.create_model_version_set(
            model_version_set_id=experiment_model_version_set_id,
            model_version_set_name=experiment_model_version_set_name,
            description=create_fine_tuning_details.experiment_description,
            compartment_id=target_compartment,
            project_id=target_project,
            freeform_tags=create_fine_tuning_details.freeform_tags,
            defined_tags=create_fine_tuning_details.defined_tags,
        )

        source = self.get_source(create_fine_tuning_details.ft_source_id)

        ft_model_custom_metadata = ModelCustomMetadata()
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_SOURCE,
            value=create_fine_tuning_details.ft_source_id,
        )
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_SOURCE_NAME,
            value=source.display_name,
        )
        service_model_artifact_location = source.custom_metadata_list.get(
            FineTuneCustomMetadata.SERVICE_MODEL_ARTIFACT_LOCATION
        )
        service_model_deployment_container = source.custom_metadata_list.get(
            FineTuneCustomMetadata.SERVICE_MODEL_DEPLOYMENT_CONTAINER
        )

        ft_model_custom_metadata.add(
            key=service_model_artifact_location.key,
            value=service_model_artifact_location.value,
            description=service_model_artifact_location.description,
        )
        ft_model_custom_metadata.add(
            key=service_model_deployment_container.key,
            value=service_model_deployment_container.value,
            description=service_model_deployment_container.description,
        )

        ft_model_taxonomy_metadata = ModelTaxonomyMetadata()
        ft_model_taxonomy_metadata[MetadataTaxonomyKeys.HYPERPARAMETERS].value = {
            **create_fine_tuning_details.ft_parameters,
            "val_set_size": create_fine_tuning_details.validation_set_size,
            "training_data": ft_dataset_path,
        }
        # needs to add 'fine_tune_model_version' tag when creating the ft model for the
        # ft container to block merging base model artifact with ft model artifact.
        ft_model_freeform_tags = {
            Tags.AQUA_FINE_TUNE_MODEL_VERSION: AQUA_FINE_TUNE_MODEL_VERSION
        }

        ft_model = self.create_model_catalog(
            display_name=create_fine_tuning_details.ft_name,
            description=create_fine_tuning_details.ft_description,
            model_version_set_id=experiment_model_version_set_id,
            model_custom_metadata=ft_model_custom_metadata,
            model_taxonomy_metadata=ft_model_taxonomy_metadata,
            compartment_id=target_compartment,
            project_id=target_project,
            model_by_reference=True,
            freeform_tags=ft_model_freeform_tags,
            defined_tags=create_fine_tuning_details.defined_tags,
        )
        defined_metadata_dict = {}
        defined_metadata_list_source = source.defined_metadata_list._to_oci_metadata()
        for defined_metadata in defined_metadata_list_source:
            if (
                defined_metadata.has_artifact
                and defined_metadata.key.lower()
                != AquaModelMetadataKeys.FINE_TUNING_CONFIGURATION.lower()
            ):
                content = self.ds_client.get_model_defined_metadatum_artifact_content(
                    source.id, defined_metadata.key
                ).data.content
                defined_metadata_dict[defined_metadata.key] = content

        for key, value in defined_metadata_dict.items():
            ft_model.create_defined_metadata_artifact(
                key, value, MetadataArtifactPathType.CONTENT
            )

        ft_job_freeform_tags = {
            Tags.AQUA_TAG: UNKNOWN,
            Tags.AQUA_FINE_TUNED_MODEL_TAG: f"{source.id}#{source.display_name}",
            **(create_fine_tuning_details.freeform_tags or {}),
        }

        ft_job = Job(name=ft_model.display_name).with_infrastructure(
            DataScienceJob()
            .with_log_group_id(create_fine_tuning_details.log_group_id)
            .with_log_id(create_fine_tuning_details.log_id)
            .with_compartment_id(target_compartment)
            .with_project_id(target_project)
            .with_shape_name(create_fine_tuning_details.shape_name)
            .with_block_storage_size(
                create_fine_tuning_details.block_storage_size
                or DEFAULT_FT_BLOCK_STORAGE_SIZE
            )
            .with_freeform_tag(**ft_job_freeform_tags)
            .with_defined_tag(**(create_fine_tuning_details.defined_tags or {}))
        )

        if not subnet_id:
            # apply default subnet id for job by setting ME_STANDALONE
            # so as to avoid using the notebook session's networking when running on it
            # https://accelerated-data-science.readthedocs.io/en/latest/user_guide/jobs/infra_and_runtime.html#networking
            ft_job.infrastructure.with_job_infrastructure_type(
                JOB_INFRASTRUCTURE_TYPE_DEFAULT_NETWORKING
            )
        else:
            ft_job.infrastructure.with_subnet_id(subnet_id)

        ft_config = self.get_finetuning_config(source.id)

        ft_container = source.custom_metadata_list.get(
            FineTuneCustomMetadata.SERVICE_MODEL_FINE_TUNE_CONTAINER
        ).value
        is_custom_container = False
        try:
            # Check if the container override flag is set. If set, then the user has chosen custom image
            if source.custom_metadata_list.get(
                AQUA_FINETUNING_CONTAINER_OVERRIDE_FLAG_METADATA_NAME
            ).value:
                is_custom_container = True
        except Exception:
            pass

        if not is_custom_container and ENV_AQUA_FINE_TUNING_CONTAINER in os.environ:
            ft_container = os.environ[ENV_AQUA_FINE_TUNING_CONTAINER]
            logger.info(
                "Using container set by environment variable %s=%s",
                ENV_AQUA_FINE_TUNING_CONTAINER,
                ft_container,
            )
            is_custom_container = True

        ft_parameters.batch_size = ft_parameters.batch_size or (
            (ft_config.shape if ft_config else UNKNOWN_DICT)
            .get(create_fine_tuning_details.shape_name, UNKNOWN_DICT)
            .get("batch_size", DEFAULT_FT_BATCH_SIZE)
        )
        finetuning_params = ft_config.finetuning_params if ft_config else UNKNOWN

        ft_job.with_runtime(
            self._build_fine_tuning_runtime(
                source_id=source.id,
                ft_model_id=ft_model.id,
                dataset_path=ft_dataset_path,
                report_path=create_fine_tuning_details.report_path,
                replica=create_fine_tuning_details.replica,
                finetuning_params=finetuning_params,
                val_set_size=(
                    create_fine_tuning_details.validation_set_size
                    or DEFAULT_FT_VALIDATION_SET_SIZE
                ),
                parameters=ft_parameters,
                ft_container=ft_container,
                is_custom_container=is_custom_container,
            )
        ).create()
        logger.debug(
            f"Successfully created fine tuning job {ft_job.id} for {create_fine_tuning_details.ft_source_id}."
        )

        ft_job_run = ft_job.run(
            name=ft_model.display_name,
            freeform_tags=ft_job_freeform_tags,
            defined_tags=create_fine_tuning_details.defined_tags or {},
            wait=False,
        )
        logger.debug(
            f"Successfully created fine tuning job run {ft_job_run.id} for {create_fine_tuning_details.ft_source_id}."
        )

        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_JOB_ID,
            value=ft_job.id,
        )
        ft_model_custom_metadata.add(
            key=FineTuneCustomMetadata.FINE_TUNE_JOB_RUN_ID,
            value=ft_job_run.id,
        )
        updated_custom_metadata_list = [
            Metadata(**metadata)
            for metadata in ft_model_custom_metadata.to_dict()["data"]
        ]

        model_freeform_tags = source.freeform_tags or {}
        model_freeform_tags.pop(Tags.LICENSE, None)
        model_freeform_tags.pop(Tags.BASE_MODEL_CUSTOM, None)

        model_freeform_tags = {
            **model_freeform_tags,
            **(ft_model.freeform_tags or {}),
            Tags.READY_TO_FINE_TUNE: "false",
            Tags.AQUA_TAG: UNKNOWN,
            Tags.AQUA_FINE_TUNED_MODEL_TAG: f"{source.id}#{source.display_name}",
            **(create_fine_tuning_details.freeform_tags or {}),
        }

        self.update_model(
            model_id=ft_model.id,
            update_model_details=UpdateModelDetails(
                custom_metadata_list=updated_custom_metadata_list,
                freeform_tags=model_freeform_tags,
            ),
        )
        logger.debug(
            f"Successfully updated model custom metadata list and freeform tags for the model {ft_model.id}."
        )

        self.update_model_provenance(
            model_id=ft_model.id,
            update_model_provenance_details=UpdateModelProvenanceDetails(
                training_id=ft_job_run.id
            ),
        )
        logger.debug(
            f"Successfully updated model provenance for the model {ft_model.id}."
        )

        # tracks the shape and replica used for fine-tuning the service models
        telemetry_kwargs = (
            {"ocid": ft_job.id[-6:]} if ft_job and len(ft_job.id) > 6 else {}
        )
        # track shapes that were used for fine-tune creation
        self.telemetry.record_event_async(
            category="aqua/service/finetune/create/shape/",
            action=f"{create_fine_tuning_details.shape_name}x{create_fine_tuning_details.replica}",
            **telemetry_kwargs,
        )
        # tracks unique fine-tuned models that were created in the user compartment
        # TODO: retrieve the service model name for FT custom models.
        self.telemetry.record_event_async(
            category="aqua/service/finetune",
            action="create",
            detail=source.display_name,
            **telemetry_kwargs,
        )
        # track combination of model and shape used for fine-tune creation
        self.telemetry.record_event_async(
            category="aqua/service/finetune/create",
            action="shape",
            detail=f"{create_fine_tuning_details.shape_name}x{create_fine_tuning_details.replica}",
            value=source.display_name,
        )

        if create_fine_tuning_details.watch_logs:
            logger.info(
                f"Watching fine-tuning job run logs for {ft_job_run.id}. Press Ctrl+C to stop watching logs.\n"
            )
            try:
                ft_job_run.watch()
            except KeyboardInterrupt:
                logger.info(f"\nStopped watching logs for {ft_job_run.id}.\n")
                time.sleep(1)
            except Exception:
                logger.debug(
                    f"Something unexpected occurred while watching logs.\n{traceback.format_exc()}"
                )

        return AquaFineTuningSummary(
            id=ft_model.id,
            name=ft_model.display_name,
            console_url=get_console_link(
                resource=Resource.MODEL,
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
                    resource=Resource.MODEL_VERSION_SET,
                    ocid=experiment_model_version_set_id,
                    region=self.region,
                ),
            ),
            source=AquaResourceIdentifier(
                id=source.id,
                name=source.display_name,
                url=get_console_link(
                    resource=Resource.MODEL,
                    ocid=source.id,
                    region=self.region,
                ),
            ),
            job=AquaResourceIdentifier(
                id=ft_job.id,
                name=ft_job.name,
                url=get_console_link(
                    resource=Resource.JOB,
                    ocid=ft_job.id,
                    region=self.region,
                ),
            ),
            tags={
                "aqua_finetuning": Tags.AQUA_FINE_TUNING,
                "finetuning_job_id": ft_job.id,
                "finetuning_source": source.id,
                "finetuning_experiment_id": experiment_model_version_set_id,
                **model_freeform_tags,
                **(create_fine_tuning_details.defined_tags or {}),
            },
            parameters=ft_parameters,
        )

    def _build_fine_tuning_runtime(
        self,
        source_id: str,
        ft_model_id: str,
        dataset_path: str,
        report_path: str,
        replica: int,
        val_set_size: float,
        parameters: AquaFineTuningParams,
        ft_container: str = None,
        finetuning_params: str = None,
        is_custom_container: bool = False,
    ) -> Runtime:
        """Builds fine tuning runtime for Job."""
        container = (
            self.get_container_image(
                container_type=ft_container,
            )
            if not is_custom_container
            else ft_container
        )
        runtime = (
            ContainerRuntime()
            .with_environment_variable(
                **{
                    "AIP_SMC_FT_ARGUMENTS": json.dumps(
                        {
                            "baseModel": {"type": "modelCatalog", "modelId": source_id},
                            "outputModel": {
                                "type": "modelCatalog",
                                "modelId": ft_model_id,
                            },
                        }
                    ),
                    "OCI__LAUNCH_CMD": self._build_oci_launch_cmd(
                        dataset_path=dataset_path,
                        report_path=report_path,
                        val_set_size=val_set_size,
                        parameters=parameters,
                        finetuning_params=finetuning_params,
                    ),
                    "CONDA_BUCKET_NS": CONDA_BUCKET_NS,
                }
            )
            .with_image(image=container)
            .with_replica(replica)
        )

        return runtime

    @staticmethod
    def _build_oci_launch_cmd(
        dataset_path: str,
        report_path: str,
        val_set_size: float,
        parameters: AquaFineTuningParams,
        finetuning_params: str = None,
    ) -> str:
        """Builds the oci launch cmd for fine tuning container runtime."""
        oci_launch_cmd = f"--training_data {dataset_path} --output_dir {report_path} --val_set_size {val_set_size} "
        for key, value in parameters.to_dict().items():
            if value is not None:
                if key == "batch_size":
                    oci_launch_cmd += f"--micro_{key} {value} "
                elif key == "epochs":
                    oci_launch_cmd += f"--num_{key} {value} "
                elif key == "lora_target_modules":
                    oci_launch_cmd += f"--{key} {','.join(str(k) for k in value)} "
                elif key == "early_stopping_patience":
                    if value != 0:
                        oci_launch_cmd += f"--{key} {value} "
                elif key == "early_stopping_threshold":
                    if "early_stopping_patience" in oci_launch_cmd:
                        oci_launch_cmd += f"--{key} {value} "
                else:
                    oci_launch_cmd += f"--{key} {value} "

        oci_launch_cmd += f"{finetuning_params}" if finetuning_params else ""
        return oci_launch_cmd.rstrip()

    @telemetry(
        entry_point="plugin=finetuning&action=get_finetuning_config", name="aqua"
    )
    def get_finetuning_config(self, model_id: str) -> AquaFineTuningConfig:
        """Gets the finetuning config for given Aqua model.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            A dict of allowed finetuning configs.
        """
        config = self.get_config_from_metadata(
            model_id, AquaModelMetadataKeys.FINE_TUNING_CONFIGURATION
        ).config
        if config:
            logger.info(
                f"Fetched {AquaModelMetadataKeys.FINE_TUNING_CONFIGURATION} from defined metadata for model: {model_id}."
            )
            return AquaFineTuningConfig(**(config or UNKNOWN_DICT))
        config = self.get_config(
            model_id,
            DEFINED_METADATA_TO_FILE_MAP.get(
                AquaModelMetadataKeys.FINE_TUNING_CONFIGURATION.lower()
            ),
        ).config
        if not config:
            logger.debug(
                f"Fine-tuning config for custom model: {model_id} is not available. Use defaults."
            )
        return AquaFineTuningConfig(**(config or UNKNOWN_DICT))

    @telemetry(
        entry_point="plugin=finetuning&action=get_finetuning_default_params",
        name="aqua",
    )
    def get_finetuning_default_params(self, model_id: str) -> Dict:
        """Gets the default params set in the finetuning configs for the given model. Only the fields that are
        available in AquaFineTuningParams will be accessible for user overrides.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            Dict of parameters from the loaded from finetuning config json file. If config information is not available,
            then an empty dict is returned.
        """
        default_params = {"params": {}}
        finetuning_config = self.get_finetuning_config(model_id)
        config_parameters = (
            finetuning_config.configuration if finetuning_config else UNKNOWN_DICT
        )
        dataclass_fields = self._get_finetuning_params(
            config_parameters, validate=False
        ).to_dict()
        for name, value in config_parameters.items():
            if name in dataclass_fields:
                if name == "micro_batch_size":
                    name = "batch_size"
                default_params["params"][name] = value

        return default_params

    @staticmethod
    def _get_finetuning_params(
        params: Dict = None, validate: bool = True
    ) -> AquaFineTuningParams:
        """
        Get and validate the fine-tuning params, and return an error message if validation fails. In order to skip
        @model_validator decorator's validation, pass validate=False.
        """
        try:
            finetuning_params = AquaFineTuningParams(
                **{**params, **{"_validate": validate}}
            )
        except ValidationError as ex:
            custom_errors = build_pydantic_error_message(ex)
            raise AquaValueError(
                f"Invalid finetuning parameters. Error details: {custom_errors}."
            ) from ex
        return finetuning_params

    def validate_finetuning_params(self, params: Dict = None) -> Dict:
        """Validate if the fine-tuning parameters passed by the user can be overridden. Parameter values are not
        validated, only param keys are validated.

        Parameters
        ----------
        params :Dict, optional
            Params passed by the user.

        Returns
        -------
            Return a list of restricted params.
        """
        self._get_finetuning_params(params or {})
        return {"valid": True}

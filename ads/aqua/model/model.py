#!/usr/bin/env python
# Copyright (c) 2024, 2025 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import pathlib
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Union

import oci
from cachetools import TTLCache
from huggingface_hub import snapshot_download
from oci.data_science.models import JobRun, Metadata, Model, UpdateModelDetails

from ads.aqua import logger
from ads.aqua.app import AquaApp
from ads.aqua.common.entities import AquaMultiModelRef
from ads.aqua.common.enums import (
    ConfigFolder,
    CustomInferenceContainerTypeFamily,
    FineTuningContainerTypeFamily,
    InferenceContainerTypeFamily,
    ModelFormat,
    Platform,
    Tags,
)
from ads.aqua.common.errors import (
    AquaFileNotFoundError,
    AquaRuntimeError,
    AquaValueError,
)
from ads.aqua.common.utils import (
    LifecycleStatus,
    _build_resource_identifier,
    cleanup_local_hf_model_artifact,
    create_word_icon,
    generate_tei_cmd_var,
    get_artifact_path,
    get_hf_model_info,
    list_os_files_with_extension,
    load_config,
    upload_folder,
)
from ads.aqua.config.container_config import AquaContainerConfig
from ads.aqua.constants import (
    AQUA_FINE_TUNE_MODEL_VERSION,
    AQUA_MODEL_ARTIFACT_CONFIG,
    AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME,
    AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE,
    AQUA_MODEL_ARTIFACT_FILE,
    AQUA_MODEL_TOKENIZER_CONFIG,
    AQUA_MODEL_TYPE_CUSTOM,
    HF_METADATA_FOLDER,
    LICENSE,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    README,
    READY_TO_DEPLOY_STATUS,
    READY_TO_FINE_TUNE_STATUS,
    READY_TO_IMPORT_STATUS,
    TRAINING_METRICS_FINAL,
    TRINING_METRICS,
    VALIDATION_METRICS,
    VALIDATION_METRICS_FINAL,
)
from ads.aqua.model.constants import (
    AquaModelMetadataKeys,
    FineTuningCustomMetadata,
    FineTuningMetricCategories,
    ModelCustomMetadataFields,
    ModelType,
)
from ads.aqua.model.entities import (
    AquaFineTuneModel,
    AquaFineTuningMetric,
    AquaModel,
    AquaModelLicense,
    AquaModelReadme,
    AquaModelSummary,
    ImportModelDetails,
    ModelValidationResult,
)
from ads.common.auth import default_signer
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import UNKNOWN, get_console_link, is_path_exists, read_file
from ads.config import (
    AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME,
    AQUA_EVALUATION_CONTAINER_METADATA_NAME,
    AQUA_FINETUNING_CONTAINER_METADATA_NAME,
    AQUA_SERVICE_MODELS,
    COMPARTMENT_OCID,
    PROJECT_OCID,
    SERVICE,
    TENANCY_OCID,
    USER,
)
from ads.model import DataScienceModel
from ads.model.common.utils import MetadataArtifactPathType
from ads.model.datascience_model_group import DataScienceModelGroup
from ads.model.model_metadata import (
    MetadataCustomCategory,
    ModelCustomMetadata,
    ModelCustomMetadataItem,
)
from ads.telemetry import telemetry


class AquaModelApp(AquaApp):
    """Provides a suite of APIs to interact with Aqua models within the Oracle
    Cloud Infrastructure Data Science service, serving as an interface for
    managing machine learning models.


    Methods
    -------
    create(model_id: str, project_id: str, compartment_id: str = None, **kwargs) -> "AquaModel"
        Creates custom aqua model from service model.
    get(model_id: str) -> AquaModel:
        Retrieves details of an Aqua model by its unique identifier.
    list(compartment_id: str = None, project_id: str = None, **kwargs) -> List[AquaModelSummary]:
        Lists all Aqua models within a specified compartment and/or project.
    clear_model_list_cache()
        Allows clear list model cache items from the service models compartment.
    register(model: str, os_path: str, local_dir: str = None)

    Note:
        This class is designed to work within the Oracle Cloud Infrastructure
        and requires proper configuration and authentication set up to interact
        with OCI services.
    """

    _service_models_cache = TTLCache(
        maxsize=10, ttl=timedelta(hours=5), timer=datetime.now
    )
    # Used for saving service model details
    _service_model_details_cache = TTLCache(
        maxsize=10, ttl=timedelta(hours=5), timer=datetime.now
    )
    _cache_lock = Lock()

    @telemetry(entry_point="plugin=model&action=create", name="aqua")
    def create(
        self,
        model: Union[str, AquaMultiModelRef],
        project_id: Optional[str] = None,
        compartment_id: Optional[str] = None,
        freeform_tags: Optional[Dict] = None,
        defined_tags: Optional[Dict] = None,
        **kwargs,
    ) -> Union[DataScienceModel, DataScienceModelGroup]:
        """
        Creates a custom Aqua model or model group from a service model.

        Parameters
        ----------
        model : Union[str, AquaMultiModelRef]
            The model ID as a string or a AquaMultiModelRef instance to be deployed.
        project_id : Optional[str]
            The project ID for the custom model.
        compartment_id : Optional[str]
            The compartment ID for the custom model. Defaults to None.
            If not provided, the compartment ID will be fetched from environment variables.
        freeform_tags : Optional[Dict]
            Freeform tags for the model.
        defined_tags : Optional[Dict]
            Defined tags for the model.

        Returns
        -------
        Union[DataScienceModel, DataScienceModelGroup]
            The instance of DataScienceModel or DataScienceModelGroup.
        """
        fine_tune_weights = []
        model_name = ""
        is_stacked = False
        if isinstance(model, AquaMultiModelRef):
            fine_tune_weights = model.fine_tune_weights or []
            model_name = model.model_name
            model = model.model_id
            is_stacked = True

        service_model = DataScienceModel.from_id(model)
        target_project = project_id or PROJECT_OCID
        target_compartment = compartment_id or COMPARTMENT_OCID

        # combine tags
        combined_freeform_tags = {
            **(service_model.freeform_tags or {}),
            **(freeform_tags or {}),
        }
        combined_defined_tags = {
            **(service_model.defined_tags or {}),
            **(defined_tags or {}),
        }

        custom_model = None
        if is_stacked:
            combined_freeform_tags.update({Tags.STACKED_MODEL_TYPE_TAG: "true"})
            custom_model = self._create_model_group(
                model_id=model,
                model_name=model_name,
                compartment_id=target_compartment,
                project_id=target_project,
                freeform_tags=combined_freeform_tags,
                defined_tags=combined_defined_tags,
                fine_tune_weights=fine_tune_weights,
                service_model=service_model,
            )

            logger.info(
                f"Aqua Model Group {custom_model.id} created with the service model {model}."
            )
        else:
            # Skip model copying if it is registered model or fine-tuned model
            if (
                Tags.BASE_MODEL_CUSTOM in service_model.freeform_tags
                or Tags.AQUA_FINE_TUNED_MODEL_TAG in service_model.freeform_tags
            ):
                logger.info(
                    f"Aqua Model {model} already exists in the user's compartment."
                    "Skipped copying."
                )
                return service_model

            custom_model = self._create_model(
                compartment_id=target_compartment,
                project_id=target_project,
                freeform_tags=combined_freeform_tags,
                defined_tags=combined_defined_tags,
                service_model=service_model,
                **kwargs,
            )
            logger.info(
                f"Aqua Model {custom_model.id} created with the service model {model}."
            )

        # Track unique models that were created in the user's compartment
        self.telemetry.record_event_async(
            category="aqua/service/model",
            action="create",
            detail=service_model.display_name,
        )

        return custom_model

    def _create_model(
        self,
        compartment_id: str,
        project_id: str,
        freeform_tags: Dict,
        defined_tags: Dict,
        service_model: DataScienceModel,
        **kwargs,
    ):
        """Creates a data science model by reference."""
        custom_model = (
            DataScienceModel()
            .with_compartment_id(compartment_id)
            .with_project_id(project_id)
            .with_model_file_description(json_dict=service_model.model_file_description)
            .with_display_name(service_model.display_name)
            .with_description(service_model.description)
            .with_freeform_tags(**freeform_tags)
            .with_defined_tags(**defined_tags)
            .with_custom_metadata_list(service_model.custom_metadata_list)
            .with_defined_metadata_list(service_model.defined_metadata_list)
            .with_provenance_metadata(service_model.provenance_metadata)
            .create(model_by_reference=True, **kwargs)
        )

        return custom_model

    def _create_model_group(
        self,
        model_id: str,
        model_name: str,
        compartment_id: str,
        project_id: str,
        freeform_tags: Dict,
        defined_tags: Dict,
        fine_tune_weights: List,
        service_model: DataScienceModel,
    ):
        """Creates a data science model group."""
        member_models = [
            {
                "inference_key": fine_tune_weight.model_name,
                "model_id": fine_tune_weight.model_id,
            }
            for fine_tune_weight in fine_tune_weights
        ]
        # must also include base model info in member models to create stacked model group
        member_models.append(
            {
                "inference_key": model_name,
                "model_id": model_id,
            }
        )
        custom_model = (
            DataScienceModelGroup()
            .with_compartment_id(compartment_id)
            .with_project_id(project_id)
            .with_display_name(service_model.display_name)
            .with_description(service_model.description)
            .with_freeform_tags(**freeform_tags)
            .with_defined_tags(**defined_tags)
            .with_custom_metadata_list(service_model.custom_metadata_list)
            .with_base_model_id(model_id)
            .with_member_models(member_models)
            .create()
        )

        return custom_model

    @telemetry(entry_point="plugin=model&action=create", name="aqua")
    def create_multi(
        self,
        models: List[AquaMultiModelRef],
        model_custom_metadata: ModelCustomMetadata,
        model_group_display_name: str,
        model_group_description: str,
        tags: Dict,
        combined_model_names: str,
        project_id: Optional[str] = None,
        compartment_id: Optional[str] = None,
        defined_tags: Optional[Dict] = None,
        **kwargs,  # noqa: ARG002
    ) -> DataScienceModelGroup:
        """
        Creates a multi-model grouping using the provided model list.

        Parameters
        ----------
        models : List[AquaMultiModelRef]
            List of AquaMultiModelRef instances for creating a multi-model group.
        model_custom_metadata : ModelCustomMetadata
            Custom metadata for creating model group.
            All model group custom metadata, including 'multi_model_metadata' and 'MULTI_MODEL_CONFIG' will be translated as a
            list of dict and placed under environment variable 'OCI_MODEL_GROUP_CUSTOM_METADATA' in model deployment.
        model_group_display_name: str
            The model group display name.
        model_group_description: str
            The model group description.
        tags: Dict
            The tags of model group.
        combined_model_names: str
            The name of models to be grouped and deployed.
        project_id : Optional[str]
            The project ID for the multi-model group.
        compartment_id : Optional[str]
            The compartment ID for the multi-model group.
        defined_tags : Optional[Dict]
            Defined tags for the model.

        Returns
        -------
        DataScienceModelGroup
            Instance of DataScienceModelGroup object.
        """
        member_model_ids = [{"model_id": model.model_id} for model in models]
        for model in models:
            if model.fine_tune_weights:
                member_model_ids.extend(
                    [
                        {"model_id": fine_tune_model.model_id}
                        for fine_tune_model in model.fine_tune_weights
                    ]
                )

        custom_model_group = (
            DataScienceModelGroup()
            .with_compartment_id(compartment_id)
            .with_project_id(project_id)
            .with_display_name(model_group_display_name)
            .with_description(model_group_description)
            .with_freeform_tags(**tags)
            .with_defined_tags(**(defined_tags or {}))
            .with_custom_metadata_list(model_custom_metadata)
            # TODO: add member model inference key
            .with_member_models(member_model_ids)
        )
        custom_model_group.create()

        logger.info(
            f"Aqua Model Group'{custom_model_group.id}' created with models: {combined_model_names}."
        )

        # Track telemetry event
        self.telemetry.record_event_async(
            category="aqua/multimodel",
            action="create",
            detail=combined_model_names,
        )

        return custom_model_group

    @telemetry(entry_point="plugin=model&action=get", name="aqua")
    def get(self, model_id: str) -> "AquaModel":
        """Gets the information of an Aqua model.

        Parameters
        ----------
        model_id: str
            The model OCID.

        Returns
        -------
        AquaModel:
            The instance of AquaModel.
        """

        cached_item = self._service_model_details_cache.get(model_id)
        if cached_item:
            logger.info(f"Fetching model details for model {model_id} from cache.")
            return cached_item

        logger.info(f"Fetching model details for model {model_id}.")
        ds_model = DataScienceModel.from_id(model_id)

        if not self._if_show(ds_model):
            raise AquaRuntimeError(
                f"Target model `{ds_model.id} `is not an Aqua model as it does not contain "
                f"{Tags.AQUA_TAG} tag."
            )

        is_fine_tuned_model = bool(
            ds_model.freeform_tags
            and ds_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG)
        )

        inference_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER),
        ).value
        inference_container_uri = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER_URI,
            ModelCustomMetadataItem(
                key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER_URI
            ),
        ).value
        evaluation_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.EVALUATION_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.EVALUATION_CONTAINER),
        ).value
        finetuning_container: str = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.FINETUNE_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.FINETUNE_CONTAINER),
        ).value
        artifact_location = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.ARTIFACT_LOCATION,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.ARTIFACT_LOCATION),
        ).value

        aqua_model_attributes = dict(
            **self._process_model(ds_model, self.region),
            project_id=ds_model.project_id,
            inference_container=inference_container,
            inference_container_uri=inference_container_uri,
            finetuning_container=finetuning_container,
            evaluation_container=evaluation_container,
            artifact_location=artifact_location,
        )

        if not is_fine_tuned_model:
            model_details = AquaModel(**aqua_model_attributes)
            self._service_model_details_cache.__setitem__(
                key=model_id, value=model_details
            )

        else:
            try:
                jobrun_ocid = ds_model.provenance_metadata.training_id
                jobrun = self.ds_client.get_job_run(jobrun_ocid).data
            except Exception as e:
                logger.debug(
                    f"Missing jobrun information in the provenance metadata of the given model {model_id}."
                    f"\nError: {str(e)}"
                )
                jobrun = None

            try:
                source_id = ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE
                ).value
            except ValueError as e:
                logger.debug(
                    f"Custom metadata is missing {FineTuningCustomMetadata.FT_SOURCE} key for "
                    f"model {model_id}.\nError: {str(e)}"
                )
                source_id = UNKNOWN

            try:
                source_name = ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE_NAME
                ).value
            except ValueError as e:
                logger.debug(
                    f"Custom metadata is missing {FineTuningCustomMetadata.FT_SOURCE_NAME} key for "
                    f"model {model_id}.\nError: {str(e)}"
                )
                source_name = UNKNOWN

            source_identifier = _build_resource_identifier(
                id=source_id,
                name=source_name,
                region=self.region,
            )

            ft_metrics = self._build_ft_metrics(ds_model.custom_metadata_list)

            job_run_status = (
                jobrun.lifecycle_state
                if jobrun and jobrun.lifecycle_state != JobRun.LIFECYCLE_STATE_DELETED
                else (
                    JobRun.LIFECYCLE_STATE_SUCCEEDED
                    if self.if_artifact_exist(ds_model.id)
                    else JobRun.LIFECYCLE_STATE_FAILED
                )
            )
            # TODO: change the argument's name.
            lifecycle_state = LifecycleStatus.get_status(
                evaluation_status=ds_model.lifecycle_state,
                job_run_status=job_run_status,
            )

            model_details = AquaFineTuneModel(
                **aqua_model_attributes,
                source=source_identifier,
                lifecycle_state=(
                    Model.LIFECYCLE_STATE_ACTIVE
                    if lifecycle_state == JobRun.LIFECYCLE_STATE_SUCCEEDED
                    else lifecycle_state
                ),
                metrics=ft_metrics,
                model=ds_model,
                jobrun=jobrun,
                region=self.region,
            )

        return model_details

    @telemetry(entry_point="plugin=model&action=delete", name="aqua")
    def delete_model(self, model_id):
        ds_model = DataScienceModel.from_id(model_id)
        is_registered_model = ds_model.freeform_tags.get(Tags.BASE_MODEL_CUSTOM, None)
        is_fine_tuned_model = ds_model.freeform_tags.get(
            Tags.AQUA_FINE_TUNED_MODEL_TAG, None
        )
        if is_registered_model or is_fine_tuned_model:
            logger.info(f"Deleting model {model_id}.")
            return ds_model.delete()
        else:
            raise AquaRuntimeError(
                f"Failed to delete model:{model_id}. Only registered models or finetuned model can be deleted."
            )

    @telemetry(entry_point="plugin=model&action=edit", name="aqua")
    def edit_registered_model(
        self, id, inference_container, inference_container_uri, enable_finetuning, task
    ):
        """Edits the default config of unverified registered model.

        Parameters
        ----------
        id: str
            The model OCID.
        inference_container: str.
            The inference container family name
        inference_container_uri: str
            The inference container uri for embedding models
        enable_finetuning: str
            Flag to enable or disable finetuning over the model. Defaults to None
        task:
            The usecase type of the model. e.g , text-generation , text_embedding etc.

        Returns
        -------
        Model:
            The instance of oci.data_science.models.Model.

        """
        ds_model = DataScienceModel.from_id(id)
        if ds_model.freeform_tags.get(Tags.BASE_MODEL_CUSTOM, None):
            if ds_model.freeform_tags.get(Tags.AQUA_SERVICE_MODEL_TAG, None):
                raise AquaRuntimeError(
                    "Only registered unverified models can be edited."
                )
            else:
                custom_metadata_list = ds_model.custom_metadata_list
                freeform_tags = ds_model.freeform_tags
                if inference_container:
                    if (
                        inference_container in CustomInferenceContainerTypeFamily
                        and inference_container_uri is None
                    ):
                        raise AquaRuntimeError(
                            "Inference container URI must be provided."
                        )
                    else:
                        custom_metadata_list.add(
                            key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
                            value=inference_container,
                            category=MetadataCustomCategory.OTHER,
                            description="Deployment container mapping for SMC",
                            replace=True,
                        )
                if inference_container_uri:
                    if (
                        inference_container in CustomInferenceContainerTypeFamily
                        or inference_container is None
                    ):
                        custom_metadata_list.add(
                            key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER_URI,
                            value=inference_container_uri,
                            category=MetadataCustomCategory.OTHER,
                            description=f"Inference container URI for {ds_model.display_name}",
                            replace=True,
                        )
                    else:
                        raise AquaRuntimeError(
                            f"Inference container URI can be edited only with container values: {CustomInferenceContainerTypeFamily.values()}"
                        )

                if enable_finetuning is not None:
                    if enable_finetuning.lower() == "true":
                        custom_metadata_list.add(
                            key=ModelCustomMetadataFields.FINETUNE_CONTAINER,
                            value=FineTuningContainerTypeFamily.AQUA_FINETUNING_CONTAINER_FAMILY,
                            category=MetadataCustomCategory.OTHER,
                            description="Fine-tuning container mapping for SMC",
                            replace=True,
                        )
                        freeform_tags.update({Tags.READY_TO_FINE_TUNE: "true"})
                    elif enable_finetuning.lower() == "false":
                        try:
                            custom_metadata_list.remove(
                                ModelCustomMetadataFields.FINETUNE_CONTAINER
                            )
                            freeform_tags.pop(Tags.READY_TO_FINE_TUNE)
                        except Exception as ex:
                            raise AquaRuntimeError(
                                f"The given model already doesn't support finetuning: {ex}"
                            ) from ex

                custom_metadata_list.remove("modelDescription")
                if task:
                    freeform_tags.update({Tags.TASK: task})
                updated_custom_metadata_list = [
                    Metadata(**metadata)
                    for metadata in custom_metadata_list.to_dict()["data"]
                ]
                update_model_details = UpdateModelDetails(
                    custom_metadata_list=updated_custom_metadata_list,
                    freeform_tags=freeform_tags,
                )
                AquaApp().update_model(id, update_model_details)
                logger.info(f"Updated model details for the model {id}.")
        else:
            raise AquaRuntimeError("Only registered unverified models can be edited.")

    def convert_fine_tune(
        self,
        model_id: str,
        delete_model: Optional[bool] = True,
        model_display_name: Optional[str] = None,
    ) -> DataScienceModel:
        """Converts legacy fine tuned model to fine tuned model v2.
        1. 'fine_tune_model_version' tag will be added as 'v2' to new fine tuned model.
        2. 'model_file_description' json will only contain fine tuned artifacts for new fine tuned model.

        Parameters
        ----------
        model_id: str
            The legacy fine tuned model OCID.
        delete_model: bool
            Flag whether to delete the legacy model or not. Defaults to True.
        model_display_name: str
            The name of fine tuned model v2 converted. Legacy model's name will be used if not provided. Defaults to None.

        Returns
        -------
        DataScienceModel:
            The instance of DataScienceModel.
        """
        legacy_fine_tuned_model = DataScienceModel.from_id(model_id)
        legacy_tags = legacy_fine_tuned_model.freeform_tags or {}

        if (
            Tags.AQUA_TAG not in legacy_tags
            or Tags.AQUA_FINE_TUNED_MODEL_TAG not in legacy_tags
        ):
            raise AquaValueError(
                f"Model '{model_id}' is not eligible for conversion. Only legacy AQUA fine-tuned models "
                f"without the 'fine_tune_model_version={AQUA_FINE_TUNE_MODEL_VERSION}' tag are supported."
            )

        if (
            legacy_tags.get(Tags.AQUA_FINE_TUNE_MODEL_VERSION, UNKNOWN).lower()
            == AQUA_FINE_TUNE_MODEL_VERSION
        ):
            raise AquaValueError(
                f"Model '{model_id}' is already a fine-tuned model in version '{AQUA_FINE_TUNE_MODEL_VERSION}'. "
                "No conversion is necessary."
            )

        if not legacy_fine_tuned_model.model_file_description:
            raise AquaValueError(
                f"Model '{model_id}' is missing required metadata and cannot be converted. "
                "This may indicate the model was not created properly or is not a supported legacy AQUA fine-tuned model."
            )

        # add 'fine_tune_model_version' tag as 'v2'
        fine_tune_model_v2_tags = {
            **legacy_tags,
            Tags.AQUA_FINE_TUNE_MODEL_VERSION: AQUA_FINE_TUNE_MODEL_VERSION,
        }

        # remove base model artifacts in 'model_file_description' json file
        # base model artifacts are placed as the first entry in 'models' list
        legacy_fine_tuned_model.model_file_description["models"].pop(0)

        fine_tune_model_v2 = (
            DataScienceModel()
            .with_compartment_id(legacy_fine_tuned_model.compartment_id)
            .with_project_id(legacy_fine_tuned_model.project_id)
            .with_model_file_description(
                json_dict=legacy_fine_tuned_model.model_file_description
            )
            .with_display_name(
                model_display_name or legacy_fine_tuned_model.display_name
            )
            .with_description(legacy_fine_tuned_model.description)
            .with_freeform_tags(**fine_tune_model_v2_tags)
            .with_defined_tags(**(legacy_fine_tuned_model.defined_tags or {}))
            .with_custom_metadata_list(legacy_fine_tuned_model.custom_metadata_list)
            .with_defined_metadata_list(legacy_fine_tuned_model.defined_metadata_list)
            .with_provenance_metadata(legacy_fine_tuned_model.provenance_metadata)
            .create(model_by_reference=True)
        )

        logger.info(
            f"Successfully created version '{AQUA_FINE_TUNE_MODEL_VERSION}' fine-tuned model: '{fine_tune_model_v2.id}' "
            f"based on legacy model '{model_id}'. This new model is now ready for deployment."
        )

        if delete_model:
            logger.info(
                f"Deleting legacy model {model_id}. To keep both models next time, set 'delete_model' as 'False'."
            )
            legacy_fine_tuned_model.delete()

        return fine_tune_model_v2

    def _fetch_metric_from_metadata(
        self,
        custom_metadata_list: ModelCustomMetadata,
        target: str,
        category: str,
        metric_name: str,
    ) -> AquaFineTuningMetric:
        """Gets target metric from `ads.model.model_metadata.ModelCustomMetadata`."""
        try:
            scores = []
            for custom_metadata in custom_metadata_list._items:
                # We use description to group metrics
                if custom_metadata.description == target:
                    scores.append(custom_metadata.value)
                    if metric_name.endswith("final"):
                        break

            return AquaFineTuningMetric(
                name=metric_name,
                category=category,
                scores=scores,
            )
        except Exception:
            return AquaFineTuningMetric(name=metric_name, category=category, scores=[])

    def _build_ft_metrics(
        self, custom_metadata_list: ModelCustomMetadata
    ) -> List[AquaFineTuningMetric]:
        """Builds Fine Tuning metrics."""

        validation_metrics = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.VALIDATION_METRICS_EPOCH,
            category=FineTuningMetricCategories.VALIDATION,
            metric_name=VALIDATION_METRICS,
        )

        training_metrics = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_EPOCH,
            category=FineTuningMetricCategories.TRAINING,
            metric_name=TRINING_METRICS,
        )

        validation_final = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.VALIDATION_METRICS_FINAL,
            category=FineTuningMetricCategories.VALIDATION,
            metric_name=VALIDATION_METRICS_FINAL,
        )

        training_final = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_FINAL,
            category=FineTuningMetricCategories.TRAINING,
            metric_name=TRAINING_METRICS_FINAL,
        )

        return [
            validation_metrics,
            training_metrics,
            validation_final,
            training_final,
        ]

    def get_hf_tokenizer_config(self, model_id):
        """
        Gets the default model tokenizer config for the given Aqua model.
        Returns the content of tokenizer_config.json stored in model artifact.

        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.

        Returns
        -------
        Dict:
            Model tokenizer config.
        """
        config = self.get_config(
            model_id, AQUA_MODEL_TOKENIZER_CONFIG, ConfigFolder.ARTIFACT
        ).config
        if not config:
            logger.debug(
                f"{AQUA_MODEL_TOKENIZER_CONFIG} is not available for the model: {model_id}. "
                f"Check if the custom metadata has the artifact path set."
            )
            return config

        return config

    @staticmethod
    def to_aqua_model(
        model: Union[
            DataScienceModel,
            oci.data_science.models.model.Model,
            oci.data_science.models.ModelSummary,
            oci.resource_search.models.ResourceSummary,
        ],
        region: str,
    ) -> AquaModel:
        """Converts a model to an Aqua model."""
        return AquaModel(**AquaModelApp._process_model(model, region))

    @staticmethod
    def _process_model(
        model: Union[
            DataScienceModel,
            oci.data_science.models.model.Model,
            oci.data_science.models.ModelSummary,
            oci.resource_search.models.ResourceSummary,
        ],
        region: str,
        inference_containers: Optional[List[Any]] = None,
    ) -> dict:
        """Constructs required fields for AquaModelSummary."""

        # todo: revisit icon generation code
        # icon = self._load_icon(model.display_name)
        icon = ""

        tags = {}
        tags.update(model.defined_tags or {})
        tags.update(model.freeform_tags or {})

        model_id = (
            model.identifier
            if isinstance(model, oci.resource_search.models.ResourceSummary)
            else model.id
        )

        console_link = get_console_link(
            resource="models",
            ocid=model_id,
            region=region,
        )

        description = ""
        if isinstance(model, (DataScienceModel, oci.data_science.models.model.Model)):
            description = model.description
        elif isinstance(model, oci.resource_search.models.ResourceSummary):
            description = model.additional_details.get("description")

        search_text = (
            AquaModelApp._build_search_text(tags=tags, description=description)
            if tags
            else UNKNOWN
        )

        freeform_tags = model.freeform_tags or {}
        is_fine_tuned_model = Tags.AQUA_FINE_TUNED_MODEL_TAG in freeform_tags
        ready_to_deploy = (
            freeform_tags.get(Tags.AQUA_TAG, "").upper() == READY_TO_DEPLOY_STATUS
        )

        ready_to_finetune = (
            freeform_tags.get(Tags.READY_TO_FINE_TUNE, "").upper()
            == READY_TO_FINE_TUNE_STATUS
        )
        ready_to_import = (
            freeform_tags.get(Tags.READY_TO_IMPORT, "").upper()
            == READY_TO_IMPORT_STATUS
        )

        try:
            model_file = model.custom_metadata_list.get(AQUA_MODEL_ARTIFACT_FILE).value
        except Exception:
            model_file = UNKNOWN

        if not inference_containers:
            inference_containers = (
                AquaApp().get_container_config().to_dict().get("inference")
            )

        model_formats_str = freeform_tags.get(
            Tags.MODEL_FORMAT, ModelFormat.SAFETENSORS
        ).upper()
        model_formats = model_formats_str.split(",")

        supported_platform: Set[str] = set()

        for container in inference_containers:
            for model_format in model_formats:
                if model_format in container.model_formats:
                    supported_platform.update(container.platforms)

        nvidia_gpu_supported = Platform.NVIDIA_GPU in supported_platform
        arm_cpu_supported = Platform.ARM_CPU in supported_platform

        return {
            "compartment_id": model.compartment_id,
            "icon": icon or UNKNOWN,
            "id": model_id,
            "license": freeform_tags.get(Tags.LICENSE, UNKNOWN),
            "name": model.display_name,
            "organization": freeform_tags.get(Tags.ORGANIZATION, UNKNOWN),
            "task": freeform_tags.get(Tags.TASK, UNKNOWN),
            "time_created": str(model.time_created),
            "is_fine_tuned_model": is_fine_tuned_model,
            "tags": tags,
            "console_link": console_link,
            "search_text": search_text,
            "ready_to_deploy": ready_to_deploy,
            "ready_to_finetune": ready_to_finetune,
            "ready_to_import": ready_to_import,
            "nvidia_gpu_supported": nvidia_gpu_supported,
            "arm_cpu_supported": arm_cpu_supported,
            "model_file": model_file,
            "model_formats": model_formats,
        }

    @telemetry(entry_point="plugin=model&action=list", name="aqua")
    def list(
        self,
        compartment_id: str = None,
        category: str = None,
        project_id: str = None,
        model_type: str = None,
        **kwargs,
    ) -> List["AquaModelSummary"]:
        """Lists all Aqua models within a specified compartment and/or project.
        If `category` is not specified, the method defaults to returning
        the service models within the pre-configured default compartment. By default, the list
        of models in the service compartment are cached. Use clear_model_list_cache() to invalidate
        the cache.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        category: (str,optional). Defaults to `SERVICE`
            The category of the models to fetch. Can be either `USER` or `SERVICE`
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        model_type: (str, optional). Defaults to `None`.
            Model type represents the type of model in the user compartment, can be either FT or BASE.
        **kwargs:
            Additional keyword arguments that can be used to filter the results.

        Returns
        -------
        List[AquaModelSummary]:
            The list of the `ads.aqua.model.AquaModelSummary`.
        """

        category = category or kwargs.pop("category", SERVICE)
        compartment_id = compartment_id or COMPARTMENT_OCID
        if category == USER:
            # tracks number of times custom model listing was called
            self.telemetry.record_event_async(
                category="aqua/custom/model", action="list"
            )

            logger.info(f"Fetching custom models from compartment_id={compartment_id}.")
            model_type = model_type.upper() if model_type else ModelType.FT
            models = self._rqs(compartment_id, model_type=model_type)
            logger.info(
                f"Fetched {len(models)} models from {compartment_id or COMPARTMENT_OCID}."
            )
        else:
            # tracks number of times service model listing was called
            self.telemetry.record_event_async(
                category="aqua/service/model", action="list"
            )

            if AQUA_SERVICE_MODELS in self._service_models_cache:
                logger.info("Returning service models list from cache.")
                return self._service_models_cache.get(AQUA_SERVICE_MODELS)
            lifecycle_state = kwargs.pop(
                "lifecycle_state", Model.LIFECYCLE_STATE_ACTIVE
            )

            models = self.list_resource(
                self.ds_client.list_models,
                compartment_id=compartment_id,
                lifecycle_state=lifecycle_state,
                category=category,
                **kwargs,
            )
            logger.info(f"Fetched {len(models)} service models.")

        aqua_models = []
        inference_containers = self.get_container_config().to_dict().get("inference")
        for model in models:
            # Skip models without required tags early
            freeform_tags = model.freeform_tags or {}
            if Tags.AQUA_TAG.lower() not in {tag.lower() for tag in freeform_tags}:
                continue

            aqua_models.append(
                AquaModelSummary(
                    **self._process_model(
                        model=model,
                        region=self.region,
                        inference_containers=inference_containers,
                    ),
                    project_id=project_id or UNKNOWN,
                )
            )

        # Adds service models to cache
        if category == SERVICE:
            self._service_models_cache.__setitem__(
                key=AQUA_SERVICE_MODELS, value=aqua_models
            )

        return aqua_models

    def clear_model_list_cache(
        self,
    ):
        """
        Allows user to clear list model cache items from the service models compartment.
        Returns
        -------
            dict with the key used, and True if cache has the key that needs to be deleted.
        """
        res = {}
        with self._cache_lock:
            if AQUA_SERVICE_MODELS in self._service_models_cache:
                self._service_models_cache.pop(key=AQUA_SERVICE_MODELS)
                logger.info("Cleared models cache for service compartment.")
                res = {
                    "cache_deleted": True,
                }
        return res

    def clear_model_details_cache(self, model_id):
        """
        Allows user to clear model details cache item
        Returns
        -------
            dict with the key used, and True if cache has the key that needs to be deleted.
        """
        res = {}
        with self._cache_lock:
            if model_id in self._service_model_details_cache:
                self._service_model_details_cache.pop(key=model_id)
                logger.info(f"Clearing model details cache for model {model_id}.")
                res = {"key": {"model_id": model_id}, "cache_deleted": True}

        return res

    @staticmethod
    def list_valid_inference_containers():
        containers = AquaApp().get_container_config().to_dict().get("inference")
        family_values = [item.family for item in containers]
        return family_values

    @telemetry(
        entry_point="plugin=model&action=get_defined_metadata_artifact_content",
        name="aqua",
    )
    def get_defined_metadata_artifact_content(self, model_id: str, metadata_key: str):
        """
        Gets the defined metadata artifact content for the given model

        Args:
            model_id: str
                model ocid for which defined metadata artifact needs to be created
            metadata_key: str
                defined metadata key  like Readme , License , DeploymentConfiguration , FinetuningConfiguration
        Returns:
            The model defined metadata artifact content. Can be either str or Dict

        """

        content = self.get_config(model_id, metadata_key)
        if not content:
            logger.debug(
                f"Defined metadata artifact {metadata_key} for model: {model_id} is not available."
            )
        return content

    @telemetry(
        entry_point="plugin=model&action=create_defined_metadata_artifact", name="aqua"
    )
    def create_defined_metadata_artifact(
        self,
        model_id: str,
        metadata_key: str,
        path_type: MetadataArtifactPathType,
        artifact_path_or_content: str,
    ) -> None:
        """
        Creates defined metadata artifact for the registered unverified model

        Args:
            model_id: str
                model ocid for which defined metadata artifact needs to be created
            metadata_key: str
                defined metadata key  like Readme , License , DeploymentConfiguration , FinetuningConfiguration
            path_type: str
                path type of the given defined metadata can be local , oss or the content itself
            artifact_path_or_content: str
                It can be local path or oss path or the actual content itself
        Returns:
            None
        """

        ds_model = DataScienceModel.from_id(model_id)
        oci_aqua = ds_model.freeform_tags.get(Tags.AQUA_TAG, None)
        if not oci_aqua:
            raise AquaRuntimeError(f"Target model {model_id} is not an Aqua model.")
        is_registered_model = ds_model.freeform_tags.get(Tags.BASE_MODEL_CUSTOM, None)
        is_verified_model = ds_model.freeform_tags.get(
            Tags.AQUA_SERVICE_MODEL_TAG, None
        )
        if is_registered_model and not is_verified_model:
            try:
                ds_model.create_defined_metadata_artifact(
                    metadata_key_name=metadata_key,
                    artifact_path_or_content=artifact_path_or_content,
                    path_type=path_type,
                )
            except Exception as ex:
                raise AquaRuntimeError(
                    f"Error occurred in creating defined metadata artifact for model {model_id}: {ex}"
                ) from ex
        else:
            raise AquaRuntimeError(
                f"Cannot create defined metadata artifact for model {model_id}"
            )

    def _create_model_catalog_entry(
        self,
        os_path: str,
        model_name: str,
        inference_container: str,
        finetuning_container: str,
        verified_model: DataScienceModel,
        validation_result: ModelValidationResult,
        compartment_id: Optional[str],
        project_id: Optional[str],
        inference_container_uri: Optional[str],
        freeform_tags: Optional[dict] = None,
        defined_tags: Optional[dict] = None,
    ) -> DataScienceModel:
        """Create model by reference from the object storage path

        Args:
            os_path (str): OCI  where the model is uploaded - oci://bucket@namespace/prefix
            model_name (str): name of the model
            inference_container (str): selects service defaults
            finetuning_container (str): selects service defaults
            verified_model (DataScienceModel): If set, then copies all the tags and custom metadata information from the service verified model
            compartment_id (Optional[str]): Compartment Id of the compartment where the model has to be created
            project_id (Optional[str]): Project id of the project where the model has to be created
            inference_container_uri (Optional[str]): Inference container uri for BYOC
            freeform_tags (dict): Freeform tags for the model
            defined_tags (dict): Defined tags for the model

        Returns:
            DataScienceModel: Returns Datascience model instance.
        """
        model = DataScienceModel()
        tags: Dict[str, str] = (
            {
                **verified_model.freeform_tags,
                Tags.AQUA_SERVICE_MODEL_TAG: verified_model.id,
            }
            if verified_model
            else {
                Tags.AQUA_TAG: "active",
                Tags.BASE_MODEL_CUSTOM: "true",
            }
        )
        tags.update({Tags.BASE_MODEL_CUSTOM: "true"})

        if validation_result and validation_result.model_formats:
            tags.update(
                {
                    Tags.MODEL_FORMAT: ",".join(
                        model_format for model_format in validation_result.model_formats
                    )
                }
            )

        # Remove `ready_to_import` tag that might get copied from service model.
        tags.pop(Tags.READY_TO_IMPORT, None)
        defined_metadata_dict = {}
        readme_file_path = os_path.rstrip("/") + "/" + README
        license_file_path = os_path.rstrip("/") + "/" + LICENSE
        if verified_model:
            # Verified model is a model in the service catalog that either has no artifacts but contains all the necessary metadata for deploying and fine tuning.
            # If set, then we copy all the model metadata.
            metadata = verified_model.custom_metadata_list
            if verified_model.model_file_description:
                model = model.with_model_file_description(
                    json_dict=verified_model.model_file_description
                )
            defined_metadata_list = (
                verified_model.defined_metadata_list._to_oci_metadata()
            )
            for defined_metadata in defined_metadata_list:
                if defined_metadata.has_artifact:
                    content = (
                        self.ds_client.get_model_defined_metadatum_artifact_content(
                            verified_model.id, defined_metadata.key
                        ).data.content
                    )
                    defined_metadata_dict[defined_metadata.key] = content
        else:
            metadata = ModelCustomMetadata()
            if not inference_container:
                raise AquaRuntimeError(
                    f"Require Inference container information. Model: {model_name} does not have associated inference "
                    f"container defaults. Check docs for more information on how to pass inference container."
                )
            metadata.add(
                key=AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
                value=inference_container,
                description=f"Inference container mapping for {model_name}",
                category="Other",
            )
            if inference_container_uri:
                metadata.add(
                    key=AQUA_DEPLOYMENT_CONTAINER_URI_METADATA_NAME,
                    value=inference_container_uri,
                    description=f"Inference container URI for {model_name}",
                    category="Other",
                )

            inference_containers = (
                AquaContainerConfig.from_service_config(
                    service_containers=self.list_service_containers()
                )
                .to_dict()
                .get("inference")
            )
            smc_container_set = {container.family for container in inference_containers}
            # only add cmd vars if inference container is not an SMC
            if (
                inference_container not in smc_container_set
                and inference_container in CustomInferenceContainerTypeFamily.values()
            ):
                cmd_vars = generate_tei_cmd_var(os_path)
                metadata.add(
                    key=AQUA_DEPLOYMENT_CONTAINER_CMD_VAR_METADATA_NAME,
                    value=" ".join(cmd_vars),
                    description=f"Inference container cmd vars for {model_name}",
                    category="Other",
                )

            if finetuning_container:
                tags[Tags.READY_TO_FINE_TUNE] = "true"
                metadata.add(
                    key=AQUA_FINETUNING_CONTAINER_METADATA_NAME,
                    value=finetuning_container,
                    description=f"Fine-tuning container mapping for {model_name}",
                    category="Other",
                )
            else:
                logger.warn(
                    "Proceeding with model registration without the fine-tuning container information. "
                    "This model will not be available for fine tuning."
                )
            if validation_result and validation_result.model_file:
                metadata.add(
                    key=AQUA_MODEL_ARTIFACT_FILE,
                    value=validation_result.model_file,
                    description=f"The model file for {model_name}",
                    category="Other",
                )

            metadata.add(
                key=AQUA_EVALUATION_CONTAINER_METADATA_NAME,
                value="odsc-llm-evaluate",
                description="Evaluation container mapping for SMC",
                category="Other",
            )

            if validation_result and validation_result.tags:
                tags[Tags.TASK] = validation_result.tags.get(Tags.TASK, UNKNOWN)
                tags[Tags.ORGANIZATION] = validation_result.tags.get(
                    Tags.ORGANIZATION, UNKNOWN
                )
                tags[Tags.LICENSE] = validation_result.tags.get(Tags.LICENSE, UNKNOWN)

        # Set artifact location to user bucket, and replace existing key if present.
        metadata.add(
            key=MODEL_BY_REFERENCE_OSS_PATH_KEY,
            value=os_path,
            description="artifact location",
            category="Other",
            replace=True,
        )
        # override tags with freeform tags if set
        tags = {**tags, **(freeform_tags or {})}
        model = (
            model.with_custom_metadata_list(metadata)
            .with_compartment_id(compartment_id or COMPARTMENT_OCID)
            .with_project_id(project_id or PROJECT_OCID)
            .with_artifact(os_path)
            .with_display_name(model_name)
            .with_freeform_tags(**tags)
            .with_defined_tags(**(defined_tags or {}))
        ).create(model_by_reference=True)
        logger.debug(f"Created model catalog entry for the model:\n{model}")
        for key, value in defined_metadata_dict.items():
            model.create_defined_metadata_artifact(
                key, value, MetadataArtifactPathType.CONTENT
            )

        if is_path_exists(readme_file_path):
            try:
                model.create_defined_metadata_artifact(
                    AquaModelMetadataKeys.README,
                    readme_file_path,
                    MetadataArtifactPathType.OSS,
                )
            except Exception as ex:
                logger.error(
                    f"Error Uploading Readme in defined metadata for model: {model.id} : {str(ex)}"
                )
        if not verified_model and is_path_exists(license_file_path):
            try:
                model.create_defined_metadata_artifact(
                    AquaModelMetadataKeys.LICENSE,
                    license_file_path,
                    MetadataArtifactPathType.OSS,
                )
            except Exception as ex:
                logger.error(
                    f"Error Uploading License in defined metadata for model: {model.id} : {str(ex)}"
                )
        return model

    @staticmethod
    def get_model_files(os_path: str, model_format: str) -> List[str]:
        """
        Get a list of model files based on the given OS path and model format.

        Args:
            os_path (str): The OS path where the model files are located.
            model_format (str): The format of the model files.

        Returns:
            List[str]: A list of model file names.

        """
        model_files: List[str] = []
        # todo: revisit this logic to account for .bin files. In the current state, .bin and .safetensor models
        #   are grouped in one category and validation checks for config.json files only.
        if model_format == ModelFormat.SAFETENSORS:
            model_files.extend(
                list_os_files_with_extension(oss_path=os_path, extension=".safetensors")
            )
            try:
                load_config(
                    file_path=os_path,
                    config_file_name=AQUA_MODEL_ARTIFACT_CONFIG,
                )
            except Exception as ex:
                message = (
                    f"The model path {os_path} does not contain the file config.json. "
                    f"Please check if the path is correct or the model artifacts are available at this location."
                )
                logger.warning(
                    f"{message}\n"
                    f"Details: {ex.reason if isinstance(ex, AquaFileNotFoundError) else str(ex)}\n"
                )
            else:
                model_files.append(AQUA_MODEL_ARTIFACT_CONFIG)

        if model_format == ModelFormat.GGUF:
            model_files.extend(
                list_os_files_with_extension(oss_path=os_path, extension=".gguf")
            )
        logger.debug(
            f"Fetched {len(model_files)} model files from {os_path} for model format {model_format}."
        )
        return model_files

    @staticmethod
    def get_hf_model_files(model_name: str, model_format: str) -> List[str]:
        """
        Get a list of model files based on the given OS path and model format.

        Args:
            model_name (str): The huggingface model name.
            model_format (str): The format of the model files.

        Returns:
            List[str]: A list of model file names.

        """
        model_files: List[str] = []

        # todo: revisit this logic to account for .bin files. In the current state, .bin and .safetensor models
        #   are grouped in one category and returns config.json file only.

        try:
            model_siblings = get_hf_model_info(repo_id=model_name).siblings
        except Exception as e:
            huggingface_err_message = str(e)
            raise AquaValueError(
                f"Could not get the model files of {model_name} from https://huggingface.co. "
                f"Error: {huggingface_err_message}."
            ) from e

        if not model_siblings:
            raise AquaValueError(
                f"Failed to fetch the model files of {model_name} from https://huggingface.co."
            )

        for model_sibling in model_siblings:
            extension = pathlib.Path(model_sibling.rfilename).suffix[1:].upper()
            if (
                model_format == ModelFormat.SAFETENSORS
                and model_sibling.rfilename == AQUA_MODEL_ARTIFACT_CONFIG
            ):
                model_files.append(model_sibling.rfilename)
            if extension == model_format:
                model_files.append(model_sibling.rfilename)

        logger.debug(
            f"Fetched {len(model_files)} model files for the model {model_name} for model format {model_format}."
        )
        return model_files

    def _validate_model(
        self,
        import_model_details: ImportModelDetails = None,
        model_name: str = None,
        verified_model: DataScienceModel = None,
    ) -> ModelValidationResult:
        """
        Validates the model configuration and returns the model format telemetry model name.

        Args:
            import_model_details (ImportModelDetails): Model details for importing the model.
            model_name (str): name of the model
            verified_model (DataScienceModel): If set, then copies all the tags and custom metadata information from
                the service verified model

        Returns:
            ModelValidationResult: The result of the model validation.

        Raises:
            AquaRuntimeError: If there is an error while loading the config file or if the model path is incorrect.
            AquaValueError: If the model format is not supported by AQUA.
        """
        model_formats = []
        validation_result: ModelValidationResult = ModelValidationResult()

        hf_download_config_present = False

        if import_model_details.download_from_hf:
            safetensors_model_files = self.get_hf_model_files(
                model_name, ModelFormat.SAFETENSORS
            )
            if (
                safetensors_model_files
                and AQUA_MODEL_ARTIFACT_CONFIG in safetensors_model_files
            ):
                hf_download_config_present = True
            gguf_model_files = self.get_hf_model_files(model_name, ModelFormat.GGUF)
        else:
            safetensors_model_files = self.get_model_files(
                import_model_details.os_path, ModelFormat.SAFETENSORS
            )
            gguf_model_files = self.get_model_files(
                import_model_details.os_path, ModelFormat.GGUF
            )

        if not (safetensors_model_files or gguf_model_files):
            raise AquaRuntimeError(
                f"The model {model_name} does not contain either {ModelFormat.SAFETENSORS} "
                f"or {ModelFormat.GGUF} files in {import_model_details.os_path} or Hugging Face repository. "
                f"Please check if the path is correct or the model artifacts are available at this location."
            )

        if verified_model:
            aqua_model = self.to_aqua_model(verified_model, self.region)
            model_formats = aqua_model.model_formats
        else:
            if safetensors_model_files:
                model_formats.append(ModelFormat.SAFETENSORS)
            if gguf_model_files:
                model_formats.append(ModelFormat.GGUF)

            # get tags for models from hf
            if import_model_details.download_from_hf:
                model_info = get_hf_model_info(repo_id=model_name)

                try:
                    license_value = UNKNOWN
                    if model_info.tags:
                        license_tag = next(
                            (
                                tag
                                for tag in model_info.tags
                                if tag.startswith("license:")
                            ),
                            UNKNOWN,
                        )
                        license_value = (
                            license_tag.split(":")[1] if license_tag else UNKNOWN
                        )

                    hf_tags = {
                        Tags.TASK: (model_info and model_info.pipeline_tag) or UNKNOWN,
                        Tags.ORGANIZATION: (
                            model_info.author
                            if model_info and hasattr(model_info, "author")
                            else UNKNOWN
                        ),
                        Tags.LICENSE: license_value,
                    }
                    validation_result.tags = hf_tags
                except Exception as ex:
                    logger.debug(
                        f"An error occurred while getting tag information for model {model_name}. "
                        f"Error: {str(ex)}"
                    )

        validation_result.model_formats = model_formats

        # now as we know that at least one type of model files exist, validate the content of oss path.
        # for safetensors, we check if config.json files exist, and for gguf format we check if files with
        # gguf extension exist.
        if {ModelFormat.SAFETENSORS, ModelFormat.GGUF}.issubset(set(model_formats)):
            if (
                import_model_details.inference_container.lower()
                == InferenceContainerTypeFamily.AQUA_LLAMA_CPP_CONTAINER_FAMILY
            ):
                self._validate_gguf_format(
                    import_model_details=import_model_details,
                    verified_model=verified_model,
                    gguf_model_files=gguf_model_files,
                    validation_result=validation_result,
                    model_name=model_name,
                )
            else:
                self._validate_safetensor_format(
                    import_model_details=import_model_details,
                    verified_model=verified_model,
                    validation_result=validation_result,
                    hf_download_config_present=hf_download_config_present,
                    model_name=model_name,
                )
        elif ModelFormat.SAFETENSORS in model_formats:
            self._validate_safetensor_format(
                import_model_details=import_model_details,
                verified_model=verified_model,
                validation_result=validation_result,
                hf_download_config_present=hf_download_config_present,
                model_name=model_name,
            )
        elif ModelFormat.GGUF in model_formats:
            self._validate_gguf_format(
                import_model_details=import_model_details,
                verified_model=verified_model,
                gguf_model_files=gguf_model_files,
                validation_result=validation_result,
                model_name=model_name,
            )

        return validation_result

    @staticmethod
    def _validate_safetensor_format(
        import_model_details: ImportModelDetails = None,
        verified_model: DataScienceModel = None,
        validation_result: ModelValidationResult = None,
        hf_download_config_present: bool = None,
        model_name: str = None,
    ):
        if import_model_details.download_from_hf:
            # validates config.json exists for safetensors model from huggingface
            if not (
                hf_download_config_present
                or import_model_details.ignore_model_artifact_check
            ):
                raise AquaRuntimeError(
                    f"The model {model_name} does not contain {AQUA_MODEL_ARTIFACT_CONFIG} file as required "
                    f"by {ModelFormat.SAFETENSORS} format model."
                    f" Please check if the model name is correct in Hugging Face repository."
                )
            validation_result.telemetry_model_name = model_name
        else:
            # validate if config.json is available from object storage, and get model name for telemetry
            model_config = None
            try:
                model_config = load_config(
                    file_path=import_model_details.os_path,
                    config_file_name=AQUA_MODEL_ARTIFACT_CONFIG,
                )
            except Exception as ex:
                message = (
                    f"The model path {import_model_details.os_path} does not contain the file config.json. "
                    f"Please check if the path is correct or the model artifacts are available at this location."
                )
                if not import_model_details.ignore_model_artifact_check:
                    logger.error(
                        f"{message}\n"
                        f"Details: {ex.reason if isinstance(ex, AquaFileNotFoundError) else str(ex)}"
                    )
                    raise AquaRuntimeError(message) from ex
                else:
                    logger.warning(
                        f"{message}\n"
                        f"Proceeding with model registration as ignore_model_artifact_check field is set."
                    )

            if verified_model:
                # model_type validation, log message if metadata field doesn't match.
                try:
                    metadata_model_type = verified_model.custom_metadata_list.get(
                        AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE
                    ).value
                    if metadata_model_type and model_config is not None:
                        if AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE in model_config:
                            if (
                                model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE]
                                != metadata_model_type
                            ):
                                logger.debug(
                                    f"The {AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE} attribute in {AQUA_MODEL_ARTIFACT_CONFIG}"
                                    f" at {import_model_details.os_path} is invalid, expected {metadata_model_type} for "
                                    f"the model {model_name}. Please check if the path is correct or "
                                    f"the correct model artifacts are available at this location."
                                    f""
                                )
                        else:
                            logger.debug(
                                f"Could not find {AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE} attribute in "
                                f"{AQUA_MODEL_ARTIFACT_CONFIG}. Proceeding with model registration."
                            )
                except Exception as ex:
                    # todo: raise exception if model_type doesn't match. Currently log message and pass since service
                    #   models do not have this metadata.
                    logger.debug(
                        f"Error occurred while processing metadata for model {model_name}. "
                        f"Exception: {str(ex)}"
                    )
                validation_result.telemetry_model_name = verified_model.display_name
            elif (
                model_config is not None
                and AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME in model_config
            ):
                validation_result.telemetry_model_name = f"{AQUA_MODEL_TYPE_CUSTOM}_{model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME]}"
            elif (
                model_config is not None
                and AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE in model_config
            ):
                validation_result.telemetry_model_name = f"{AQUA_MODEL_TYPE_CUSTOM}_{model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE]}"
            else:
                validation_result.telemetry_model_name = AQUA_MODEL_TYPE_CUSTOM

    @staticmethod
    def _validate_gguf_format(
        import_model_details: ImportModelDetails = None,
        verified_model: DataScienceModel = None,
        gguf_model_files: List[str] = None,
        validation_result: ModelValidationResult = None,
        model_name: str = None,
    ):
        if import_model_details.finetuning_container:
            raise AquaValueError(
                "Fine-tuning is currently not supported with GGUF model format."
            )
        if verified_model:
            try:
                model_file = verified_model.custom_metadata_list.get(
                    AQUA_MODEL_ARTIFACT_FILE
                ).value
            except ValueError as err:
                raise AquaRuntimeError(
                    f"The model {verified_model.display_name} does not contain the custom metadata {AQUA_MODEL_ARTIFACT_FILE}. "
                    f"Please check if the model has the valid metadata."
                ) from err
        else:
            model_file = import_model_details.model_file

        model_files = gguf_model_files
        # todo: have a separate error validation class for different type of error messages.
        if model_file:
            if model_file not in model_files:
                raise AquaRuntimeError(
                    f"The model path {import_model_details.os_path} or the Hugging Face "
                    f"model repository for {model_name} does not contain the file "
                    f"{model_file}. Please check if the path is correct or the model "
                    f"artifacts are available at this location."
                )
            else:
                validation_result.model_file = model_file
        elif len(model_files) == 0:
            raise AquaRuntimeError(
                f"The model path {import_model_details.os_path} or the Hugging Face model "
                f"repository for {model_name} does not contain any GGUF format files. "
                f"Please check if the path is correct or the model artifacts are available "
                f"at this location."
            )
        elif len(model_files) > 1:
            raise AquaRuntimeError(
                f"The model path {import_model_details.os_path} or the Hugging Face model "
                f"repository for {model_name} contains multiple GGUF format files. "
                f"Please specify the file that needs to be deployed using the model_file "
                f"parameter."
            )
        else:
            validation_result.model_file = model_files[0]

        if verified_model:
            validation_result.telemetry_model_name = verified_model.display_name
        elif import_model_details.download_from_hf:
            validation_result.telemetry_model_name = model_name
        else:
            validation_result.telemetry_model_name = AQUA_MODEL_TYPE_CUSTOM

    @staticmethod
    def _download_model_from_hf(
        model_name: str,
        os_path: str,
        local_dir: str = None,
        allow_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
    ) -> str:
        """This helper function downloads the model artifact from Hugging Face to a local folder, then uploads
        to object storage location.

        Parameters
        ----------
        model_name (str): The huggingface model name.
        os_path (str): The OS path where the model files are located.
        local_dir (str): The local temp dir to store the huggingface model.
        allow_patterns (list): Model files matching at least one pattern are downloaded.
            Example: ["*.json"] will download all .json files. ["folder/*"] will download all files under `folder`.
            Patterns are Standard Wildcards (globbing patterns) and rules can be found here: https://docs.python.org/3/library/fnmatch.html
        ignore_patterns (list): Model files matching any of the patterns are not downloaded.
            Example: ["*.json"] will ignore all .json files. ["folder/*"] will ignore all files under `folder`.
            Patterns are Standard Wildcards (globbing patterns) and rules can be found here: https://docs.python.org/3/library/fnmatch.html

        Returns
        -------
        model_artifact_path (str): Location where the model artifacts are downloaded.
        """
        # Download the model from hub
        if local_dir:
            local_dir = os.path.join(local_dir, model_name)
            os.makedirs(local_dir, exist_ok=True)

        # if local_dir is not set, the return value points to the cached data folder
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        # Upload to object storage and skip .cache/huggingface/ folder
        logger.debug(
            f"Uploading local artifacts from local directory {local_dir} to {os_path}."
        )
        # Upload to object storage
        model_artifact_path = upload_folder(
            os_path=os_path,
            local_dir=local_dir,
            model_name=model_name,
            exclude_pattern=f"{HF_METADATA_FOLDER}*",
        )

        return model_artifact_path

    def register(
        self, import_model_details: ImportModelDetails = None, **kwargs
    ) -> AquaModel:
        """Loads the model from object storage and registers as Model in Data Science Model catalog
        The inference container and finetuning container could be of type Service Managed Container(SMC) or custom.
        If it is custom, full container URI is expected. If it of type SMC, only the container family name is expected.\n
        For detailed information about CLI flags see: https://github.com/oracle-samples/oci-data-science-ai-samples/blob/main/ai-quick-actions/cli-tips.md#register-model

        Args:
            import_model_details (ImportModelDetails): Model details for importing the model.
            kwargs:
                model (str): name of the model or OCID of the service model that has inference and finetuning information
                os_path (str): Object storage destination URI to store the downloaded model. Format: oci://bucket-name@namespace/prefix
                inference_container (str): selects service defaults
                finetuning_container (str): selects service defaults
                allow_patterns (list): Model files matching at least one pattern are downloaded.
                    Example: ["*.json"] will download all .json files. ["folder/*"] will download all files under `folder`.
                    Patterns are Standard Wildcards (globbing patterns) and rules can be found here: https://docs.python.org/3/library/fnmatch.html
                ignore_patterns (list): Model files matching any of the patterns are not downloaded.
                    Example: ["*.json"] will ignore all .json files. ["folder/*"] will ignore all files under `folder`.
                    Patterns are Standard Wildcards (globbing patterns) and rules can be found here: https://docs.python.org/3/library/fnmatch.html
                cleanup_model_cache (bool): Deletes downloaded files from local machine after model is successfully
                registered. Set to True by default.

        Returns:
            AquaModel:
                The registered model as a AquaModel object.
        """
        if not import_model_details:
            import_model_details = ImportModelDetails(**kwargs)

        # If OCID of a model is passed, we need to copy the defaults for Tags and metadata from the service model.
        verified_model: Optional[DataScienceModel] = None
        if (
            import_model_details.model.startswith("ocid")
            and "datasciencemodel" in import_model_details.model
        ):
            logger.info(f"Fetching details for model {import_model_details.model}.")
            verified_model = DataScienceModel.from_id(import_model_details.model)
        else:
            # If users passes model name, check if there is model with the same name in the service model catalog. If it is there, then use that model
            model_service_id = self._find_matching_aqua_model(
                import_model_details.model
            )
            if model_service_id:
                logger.info(
                    f"Found service model for {import_model_details.model}: {model_service_id}"
                )
                verified_model = DataScienceModel.from_id(model_service_id)

        # Copy the model name from the service model if `model` is ocid
        model_name = (
            verified_model.display_name
            if verified_model
            else import_model_details.model
        )

        # validate model and artifact
        validation_result = self._validate_model(
            import_model_details=import_model_details,
            model_name=model_name,
            verified_model=verified_model,
        )

        # download model from hugginface if indicates
        if import_model_details.download_from_hf:
            artifact_path = self._download_model_from_hf(
                model_name=model_name,
                os_path=import_model_details.os_path,
                local_dir=import_model_details.local_dir,
                allow_patterns=import_model_details.allow_patterns,
                ignore_patterns=import_model_details.ignore_patterns,
            ).rstrip("/")
        else:
            artifact_path = import_model_details.os_path.rstrip("/")

        # Create Model catalog entry with pass by reference
        ds_model = self._create_model_catalog_entry(
            os_path=artifact_path,
            model_name=model_name,
            inference_container=import_model_details.inference_container,
            finetuning_container=import_model_details.finetuning_container,
            verified_model=verified_model,
            validation_result=validation_result,
            compartment_id=import_model_details.compartment_id,
            project_id=import_model_details.project_id,
            inference_container_uri=import_model_details.inference_container_uri,
            freeform_tags=import_model_details.freeform_tags,
            defined_tags=import_model_details.defined_tags,
        )
        # registered model will always have inference and evaluation container, but
        # fine-tuning container may be not set
        inference_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER),
        ).value
        inference_container_uri = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER_URI,
            ModelCustomMetadataItem(
                key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER_URI
            ),
        ).value
        evaluation_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.EVALUATION_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.EVALUATION_CONTAINER),
        ).value
        finetuning_container: str = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.FINETUNE_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.FINETUNE_CONTAINER),
        ).value

        aqua_model_attributes = dict(
            **self._process_model(ds_model, self.region),
            project_id=ds_model.project_id,
            inference_container=inference_container,
            inference_container_uri=inference_container_uri,
            finetuning_container=finetuning_container,
            evaluation_container=evaluation_container,
            artifact_location=artifact_path,
        )

        self.telemetry.record_event_async(
            category="aqua/model",
            action="register",
            detail=validation_result.telemetry_model_name,
        )

        if (
            import_model_details.download_from_hf
            and import_model_details.cleanup_model_cache
        ):
            cleanup_local_hf_model_artifact(
                model_name=model_name, local_dir=import_model_details.local_dir
            )

        return AquaModel(**aqua_model_attributes)

    def _if_show(self, model: DataScienceModel) -> bool:
        """Determine if the given model should be return by `list`."""
        if model.freeform_tags is None:
            return False

        TARGET_TAGS = model.freeform_tags.keys()
        return Tags.AQUA_TAG in TARGET_TAGS or Tags.AQUA_TAG.lower() in TARGET_TAGS

    def _load_icon(self, model_name: str) -> str:
        """Loads icon."""

        # TODO: switch to the official logo
        try:
            return create_word_icon(model_name, return_as_datauri=True)
        except Exception as e:
            logger.debug(f"Failed to load icon for the model={model_name}: {str(e)}.")
            return None

    def _rqs(self, compartment_id: str, model_type="FT", **kwargs):
        """Use RQS to fetch models in the user tenancy."""
        if model_type == ModelType.FT:
            filter_tag = Tags.AQUA_FINE_TUNED_MODEL_TAG
        elif model_type == ModelType.BASE:
            filter_tag = Tags.BASE_MODEL_CUSTOM
        # elif model_type == ModelType.MULTIMODEL:
        #     filter_tag = Tags.MULTIMODEL_TYPE_TAG
        else:
            raise AquaValueError(
                f"Model of type {model_type} is unknown. The values should be in {ModelType.values()}"
            )

        condition_tags = f"&& (freeformTags.key = '{Tags.AQUA_TAG}' && freeformTags.key = '{filter_tag}')"
        condition_lifecycle = "&& lifecycleState = 'ACTIVE'"
        query = f"query datasciencemodel resources where (compartmentId = '{compartment_id}' {condition_lifecycle} {condition_tags})"
        logger.info(query)
        logger.info(f"tenant_id={TENANCY_OCID}")
        return OCIResource.search(
            query, type=SEARCH_TYPE.STRUCTURED, tenant_id=TENANCY_OCID, **kwargs
        )

    @staticmethod
    def _build_search_text(tags: dict, description: str = None) -> str:
        """Constructs search_text field in response."""
        description = description or ""
        tags_text = (
            ",".join(str(v) for v in tags.values()) if isinstance(tags, dict) else ""
        )
        separator = " " if description else ""
        return f"{description}{separator}{tags_text}"

    @telemetry(entry_point="plugin=model&action=load_readme", name="aqua")
    def load_readme(self, model_id: str) -> AquaModelReadme:
        """Loads the readme or the model card for the given model.

        Parameters
        ----------
        model_id: str
            The model id.

        Returns
        -------
        AquaModelReadme:
            The instance of AquaModelReadme.
        """
        oci_model = self.ds_client.get_model(model_id).data
        artifact_path = get_artifact_path(oci_model.custom_metadata_list)
        if not artifact_path:
            raise AquaRuntimeError(
                f"Readme could not be loaded. Failed to get artifact path from custom metadata for"
                f"the model {model_id}."
            )

        content = ""
        try:
            content = self.ds_client.get_model_defined_metadatum_artifact_content(
                model_id, AquaModelMetadataKeys.README
            ).data.content.decode("utf-8", errors="ignore")
            logger.info(f"Fetched {README} from defined metadata for model: {model_id}")
        except Exception as ex:
            logger.error(
                f"Readme could not be found for model: {model_id} in defined metadata : {str(ex)}"
            )
            artifact_path = get_artifact_path(oci_model.custom_metadata_list)
            readme_path = os.path.join(os.path.dirname(artifact_path), "artifact")
            if not is_path_exists(readme_path):
                readme_path = os.path.join(artifact_path.rstrip("/"), "artifact")
                if not is_path_exists(readme_path):
                    readme_path = f"{artifact_path.rstrip('/')}/"

            readme_file_path = os.path.join(readme_path, README)
            logger.info(f"Fetching {README} from {readme_file_path}")
            if is_path_exists(readme_file_path):
                try:
                    content = str(read_file(readme_file_path, auth=default_signer()))
                except Exception as e:
                    logger.debug(
                        f"Error occurred while fetching config {README} at path {readme_file_path} : {str(e)}"
                    )
        return AquaModelReadme(id=model_id, model_card=content)

    @telemetry(entry_point="plugin=model&action=load_license", name="aqua")
    def load_license(self, model_id: str) -> AquaModelLicense:
        """Loads the license full text for the given model.

        Parameters
        ----------
        model_id: str
            The model id.

        Returns
        -------
        AquaModelLicense:
            The instance of AquaModelLicense.
        """
        oci_model = self.ds_client.get_model(model_id).data
        artifact_path = get_artifact_path(oci_model.custom_metadata_list)
        if not artifact_path:
            raise AquaRuntimeError(
                f"License could not be loaded. Failed to get artifact path from custom metadata for"
                f"the model {model_id}."
            )

        content = ""
        try:
            content = self.ds_client.get_model_defined_metadatum_artifact_content(
                model_id, AquaModelMetadataKeys.LICENSE
            ).data.content.decode("utf-8", errors="ignore")
            logger.info(
                f"Fetched {LICENSE} from defined metadata for model: {model_id}"
            )
        except Exception as ex:
            logger.error(
                f"License could not be found for model: {model_id} in defined metadata : {str(ex)}"
            )
            artifact_path = get_artifact_path(oci_model.custom_metadata_list)
            license_path = os.path.join(os.path.dirname(artifact_path), "config")
            if not is_path_exists(license_path):
                license_path = os.path.join(artifact_path.rstrip("/"), "config")
                if not is_path_exists(license_path):
                    license_path = f"{artifact_path.rstrip('/')}/"

            license_file_path = os.path.join(license_path, LICENSE)
            logger.info(f"Fetching {LICENSE} from {license_file_path}")
            if is_path_exists(license_file_path):
                try:
                    content = str(read_file(license_file_path, auth=default_signer()))
                except Exception as e:
                    logger.debug(
                        f"Error occurred while fetching config {LICENSE} at path {license_path} : {str(e)}"
                    )
        return AquaModelLicense(id=model_id, license=content)

    def _find_matching_aqua_model(self, model_id: str) -> Optional[str]:
        """
        Finds a matching model in AQUA based on the model ID from list of verified models.

        Parameters
        ----------
        model_id (str): Verified model ID to match.

        Returns
        -------
        Optional[str]
            Returns model ocid that matches the model in the service catalog else returns None.
        """
        # Convert the model ID to lowercase once
        model_id_lower = model_id.lower()

        aqua_model_list = self.list()

        for aqua_model_summary in aqua_model_list:
            if aqua_model_summary.name.lower() == model_id_lower:
                logger.info(
                    f"Found matching verified model id {aqua_model_summary.id} for the model {model_id}"
                )
                return aqua_model_summary.id

        return None

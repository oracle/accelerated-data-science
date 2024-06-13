#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
from datetime import datetime, timedelta
from threading import Lock
from typing import List, Optional, Union

from cachetools import TTLCache
from oci.data_science.models import JobRun, Model

from ads.aqua import ODSC_MODEL_COMPARTMENT_OCID
from ads.aqua.app import AquaApp
from ads.aqua.common.enums import Tags
from ads.aqua.common.errors import AquaRuntimeError
from ads.aqua.common.utils import (
    create_word_icon,
    get_artifact_path,
    read_file,
    copy_model_config,
    load_config,
)
from ads.aqua.constants import (
    LICENSE_TXT,
    MODEL_BY_REFERENCE_OSS_PATH_KEY,
    README,
    READY_TO_DEPLOY_STATUS,
    READY_TO_FINE_TUNE_STATUS,
    READY_TO_IMPORT_STATUS,
    TRAINING_METRICS_FINAL,
    TRINING_METRICS,
    UNKNOWN,
    VALIDATION_METRICS,
    VALIDATION_METRICS_FINAL,
    AQUA_MODEL_ARTIFACT_CONFIG,
    AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME,
    AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE,
    AQUA_MODEL_TYPE_CUSTOM,
)
from ads.aqua.model.constants import *
from ads.aqua.model.entities import *
from ads.common.auth import default_signer
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.utils import get_console_link
from ads.config import (
    AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
    AQUA_EVALUATION_CONTAINER_METADATA_NAME,
    AQUA_FINETUNING_CONTAINER_METADATA_NAME,
    COMPARTMENT_OCID,
    PROJECT_OCID,
    TENANCY_OCID,
)
from ads.model import DataScienceModel
from ads.model.model_metadata import ModelCustomMetadata, ModelCustomMetadataItem
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
        self, model_id: str, project_id: str, compartment_id: str = None, **kwargs
    ) -> DataScienceModel:
        """Creates custom aqua model from service model.

        Parameters
        ----------
        model_id: str
            The service model id.
        project_id: str
            The project id for custom model.
        compartment_id: str
            The compartment id for custom model. Defaults to None.
            If not provided, compartment id will be fetched from environment variables.

        Returns
        -------
        DataScienceModel:
            The instance of DataScienceModel.
        """
        service_model = DataScienceModel.from_id(model_id)
        target_project = project_id or PROJECT_OCID
        target_compartment = compartment_id or COMPARTMENT_OCID

        if service_model.compartment_id != ODSC_MODEL_COMPARTMENT_OCID:
            logger.debug(
                f"Aqua Model {model_id} already exists in user's compartment."
                "Skipped copying."
            )
            return service_model

        custom_model = (
            DataScienceModel()
            .with_compartment_id(target_compartment)
            .with_project_id(target_project)
            .with_model_file_description(json_dict=service_model.model_file_description)
            .with_display_name(service_model.display_name)
            .with_description(service_model.description)
            .with_freeform_tags(**(service_model.freeform_tags or {}))
            .with_defined_tags(**(service_model.defined_tags or {}))
            .with_custom_metadata_list(service_model.custom_metadata_list)
            .with_defined_metadata_list(service_model.defined_metadata_list)
            .with_provenance_metadata(service_model.provenance_metadata)
            # TODO: decide what kwargs will be needed.
            .create(model_by_reference=True, **kwargs)
        )
        logger.debug(
            f"Aqua Model {custom_model.id} created with the service model {model_id}"
        )

        # tracks unique models that were created in the user compartment
        self.telemetry.record_event_async(
            category="aqua/service/model",
            action="create",
            detail=service_model.display_name,
        )

        return custom_model

    @telemetry(entry_point="plugin=model&action=get", name="aqua")
    def get(self, model_id: str, load_model_card: Optional[bool] = True) -> "AquaModel":
        """Gets the information of an Aqua model.

        Parameters
        ----------
        model_id: str
            The model OCID.
        load_model_card: (bool, optional). Defaults to `True`.
            Whether to load model card from artifacts or not.

        Returns
        -------
        AquaModel:
            The instance of AquaModel.
        """

        cached_item = self._service_model_details_cache.get(model_id)
        if cached_item:
            return cached_item

        ds_model = DataScienceModel.from_id(model_id)
        if not self._if_show(ds_model):
            raise AquaRuntimeError(f"Target model `{ds_model.id} `is not Aqua model.")

        is_fine_tuned_model = (
            True
            if ds_model.freeform_tags
            and ds_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG)
            else False
        )

        # todo: consolidate this logic in utils for model and deployment use
        is_verified_type = (
            ds_model.freeform_tags.get(Tags.READY_TO_IMPORT, "false").upper()
            == READY_TO_IMPORT_STATUS
        )

        model_card = ""
        if load_model_card:
            artifact_path = get_artifact_path(
                ds_model.custom_metadata_list._to_oci_metadata()
            )
            if artifact_path != UNKNOWN:
                model_card = str(
                    read_file(
                        file_path=(
                            f"{artifact_path.rstrip('/')}/config/{README}"
                            if is_verified_type
                            else f"{artifact_path.rstrip('/')}/{README}"
                        ),
                        auth=default_signer(),
                    )
                )

        inference_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER,
            ModelCustomMetadataItem(key=ModelCustomMetadataFields.DEPLOYMENT_CONTAINER),
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
            model_card=model_card,
            inference_container=inference_container,
            finetuning_container=finetuning_container,
            evaluation_container=evaluation_container,
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
                )
                jobrun = None

            try:
                source_id = ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE
                ).value
            except ValueError as e:
                logger.debug(str(e))
                source_id = UNKNOWN

            try:
                source_name = ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE_NAME
                ).value
            except ValueError as e:
                logger.debug(str(e))
                source_name = UNKNOWN

            source_identifier = utils._build_resource_identifier(
                id=source_id,
                name=source_name,
                region=self.region,
            )

            ft_metrics = self._build_ft_metrics(ds_model.custom_metadata_list)

            job_run_status = (
                jobrun.lifecycle_state
                if jobrun
                and not jobrun.lifecycle_state == JobRun.LIFECYCLE_STATE_DELETED
                else (
                    JobRun.LIFECYCLE_STATE_SUCCEEDED
                    if self.if_artifact_exist(ds_model.id)
                    else JobRun.LIFECYCLE_STATE_FAILED
                )
            )
            # TODO: change the argument's name.
            lifecycle_state = utils.LifecycleStatus.get_status(
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
        except:
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

    def _process_model(
        self,
        model: Union[
            DataScienceModel,
            oci.data_science.models.model.Model,
            oci.data_science.models.ModelSummary,
            oci.resource_search.models.ResourceSummary,
        ],
        region: str,
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

        console_link = (
            get_console_link(
                resource="models",
                ocid=model_id,
                region=region,
            ),
        )

        description = ""
        if isinstance(model, DataScienceModel) or isinstance(
            model, oci.data_science.models.model.Model
        ):
            description = model.description
        elif isinstance(model, oci.resource_search.models.ResourceSummary):
            description = model.additional_details.get("description")

        search_text = (
            self._build_search_text(tags=tags, description=description)
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

        return dict(
            compartment_id=model.compartment_id,
            icon=icon or UNKNOWN,
            id=model_id,
            license=freeform_tags.get(Tags.LICENSE, UNKNOWN),
            name=model.display_name,
            organization=freeform_tags.get(Tags.ORGANIZATION, UNKNOWN),
            task=freeform_tags.get(Tags.TASK, UNKNOWN),
            time_created=model.time_created,
            is_fine_tuned_model=is_fine_tuned_model,
            tags=tags,
            console_link=console_link,
            search_text=search_text,
            ready_to_deploy=ready_to_deploy,
            ready_to_finetune=ready_to_finetune,
            ready_to_import=ready_to_import,
        )

    @telemetry(entry_point="plugin=model&action=list", name="aqua")
    def list(
        self,
        compartment_id: str = None,
        project_id: str = None,
        model_type: str = None,
        **kwargs,
    ) -> List["AquaModelSummary"]:
        """Lists all Aqua models within a specified compartment and/or project.
        If `compartment_id` is not specified, the method defaults to returning
        the service models within the pre-configured default compartment. By default, the list
        of models in the service compartment are cached. Use clear_model_list_cache() to invalidate
        the cache.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
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

        models = []
        if compartment_id:
            # tracks number of times custom model listing was called
            self.telemetry.record_event_async(
                category="aqua/custom/model", action="list"
            )

            logger.info(f"Fetching custom models from compartment_id={compartment_id}.")
            model_type = model_type.upper() if model_type else ModelType.FT
            models = self._rqs(compartment_id, model_type=model_type)
        else:
            # tracks number of times service model listing was called
            self.telemetry.record_event_async(
                category="aqua/service/model", action="list"
            )

            if ODSC_MODEL_COMPARTMENT_OCID in self._service_models_cache.keys():
                logger.info(
                    f"Returning service models list in {ODSC_MODEL_COMPARTMENT_OCID} from cache."
                )
                return self._service_models_cache.get(ODSC_MODEL_COMPARTMENT_OCID)
            logger.info(
                f"Fetching service models from compartment_id={ODSC_MODEL_COMPARTMENT_OCID}"
            )
            lifecycle_state = kwargs.pop(
                "lifecycle_state", Model.LIFECYCLE_STATE_ACTIVE
            )

            models = self.list_resource(
                self.ds_client.list_models,
                compartment_id=ODSC_MODEL_COMPARTMENT_OCID,
                lifecycle_state=lifecycle_state,
                **kwargs,
            )

        logger.info(
            f"Fetch {len(models)} model in compartment_id={compartment_id or ODSC_MODEL_COMPARTMENT_OCID}."
        )

        aqua_models = []

        for model in models:
            aqua_models.append(
                AquaModelSummary(
                    **self._process_model(model=model, region=self.region),
                    project_id=project_id or UNKNOWN,
                )
            )

        if not compartment_id:
            self._service_models_cache.__setitem__(
                key=ODSC_MODEL_COMPARTMENT_OCID, value=aqua_models
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
        logger.info(f"Clearing _service_models_cache")
        with self._cache_lock:
            if ODSC_MODEL_COMPARTMENT_OCID in self._service_models_cache.keys():
                self._service_models_cache.pop(key=ODSC_MODEL_COMPARTMENT_OCID)
                res = {
                    "key": {
                        "compartment_id": ODSC_MODEL_COMPARTMENT_OCID,
                    },
                    "cache_deleted": True,
                }
        return res

    def _create_model_catalog_entry(
        self,
        os_path: str,
        model_name: str,
        inference_container: str,
        finetuning_container: str,
        verified_model: DataScienceModel,
        compartment_id: Optional[str],
        project_id: Optional[str],
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

        Returns:
            DataScienceModel: Returns Datascience model instance.
        """
        model = DataScienceModel()
        tags = (
            {
                **verified_model.freeform_tags,
                Tags.AQUA_SERVICE_MODEL_TAG: verified_model.id,
            }
            if verified_model
            else {Tags.AQUA_TAG: "active", Tags.BASE_MODEL_CUSTOM: "true"}
        )
        tags.update({Tags.BASE_MODEL_CUSTOM: "true"})

        # Remove `ready_to_import` tag that might get copied from service model.
        tags.pop(Tags.READY_TO_IMPORT, None)
        metadata = None
        if verified_model:
            # Verified model is a model in the service catalog that either has no artifacts but contains all the necessary metadata for deploying and fine tuning.
            # If set, then we copy all the model metadata.
            metadata = verified_model.custom_metadata_list
            if verified_model.model_file_description:
                model = model.with_model_file_description(
                    json_dict=verified_model.model_file_description
                )

        else:
            metadata = ModelCustomMetadata()
            if not inference_container:
                raise AquaRuntimeError(
                    f"Require Inference container information. Model: {model_name} does not have associated inference container defaults. Check docs for more information on how to pass inference container."
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
                    f"Proceeding with model registration without the fine-tuning container information. "
                    f"This model will not be available for fine tuning."
                )

            metadata.add(
                key=AQUA_DEPLOYMENT_CONTAINER_METADATA_NAME,
                value=inference_container,
                description=f"Inference container mapping for {model_name}",
                category="Other",
            )
            metadata.add(
                key=AQUA_EVALUATION_CONTAINER_METADATA_NAME,
                value="odsc-llm-evaluate",
                description="Evaluation container mapping for SMC",
                category="Other",
            )
            # TODO: either get task and organization from user or a config file
            # tags["task"] = "UNKNOWN"
            # tags["organization"] = "UNKNOWN"

        try:
            # If verified model already has a artifact json, use that.
            artifact_path = metadata.get(MODEL_BY_REFERENCE_OSS_PATH_KEY).value
            logger.info(
                f"Found model artifact in the service bucket. "
                f"Using artifact from service bucket instead of {os_path}"
            )

            # todo: implement generic copy_folder method
            # copy model config from artifact path to user bucket
            copy_model_config(
                artifact_path=artifact_path, os_path=os_path, auth=default_signer()
            )
        except:
            logger.debug(
                f"Proceeding with model registration without copying model config files at {os_path}. "
                f"Default configuration will be used for deployment and fine-tuning."
            )
        # Set artifact location to user bucket, and replace existing key if present.
        metadata.add(
            key=MODEL_BY_REFERENCE_OSS_PATH_KEY,
            value=os_path,
            description="artifact location",
            category="Other",
            replace=True,
        )

        model = (
            model.with_custom_metadata_list(metadata)
            .with_compartment_id(compartment_id or COMPARTMENT_OCID)
            .with_project_id(project_id or PROJECT_OCID)
            .with_artifact(os_path)
            .with_display_name(model_name)
            .with_freeform_tags(**tags)
        ).create(model_by_reference=True)
        logger.debug(model)
        return model

    def register(
        self, import_model_details: ImportModelDetails = None, **kwargs
    ) -> AquaModel:
        """Loads the model from object storage and registers as Model in Data Science Model catalog
        The inference container and finetuning container could be of type Service Manged Container(SMC) or custom.
        If it is custom, full container URI is expected. If it of type SMC, only the container family name is expected.

        Args:
            import_model_details (ImportModelDetails): Model details for importing the model.
            kwargs:
                model (str): name of the model or OCID of the service model that has inference and finetuning information
                os_path (str): Object storage destination URI to store the downloaded model. Format: oci://bucket-name@namespace/prefix
                inference_container (str): selects service defaults
                finetuning_container (str): selects service defaults

        Returns:
            AquaModel:
                The registered model as a AquaModel object.
        """
        verified_model_details: DataScienceModel = None

        if not import_model_details:
            import_model_details = ImportModelDetails(**kwargs)

        try:
            model_config = load_config(
                file_path=import_model_details.os_path,
                config_file_name=AQUA_MODEL_ARTIFACT_CONFIG,
            )
        except Exception as ex:
            logger.error(
                f"Exception occurred while loading config file from {import_model_details.os_path}"
                f"Exception message: {ex}"
            )
            raise AquaRuntimeError(
                f"The model path {import_model_details.os_path} does not contain the file config.json. "
                f"Please check if the path is correct or the model artifacts are available at this location."
            )

        model_service_id = None
        # If OCID of a model is passed, we need to copy the defaults for Tags and metadata from the service model.
        if (
            import_model_details.model.startswith("ocid")
            and "datasciencemodel" in import_model_details.model
        ):
            model_service_id = import_model_details.model
        else:
            # If users passes model name, check if there is model with the same name in the service model catalog. If it is there, then use that model
            model_service_id = self._find_matching_aqua_model(
                import_model_details.model
            )
            logger.info(
                f"Found service model for {import_model_details.model}: {model_service_id}"
            )
        if model_service_id:
            verified_model_details = DataScienceModel.from_id(model_service_id)
            try:
                metadata_model_type = verified_model_details.custom_metadata_list.get(
                    AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE
                ).value
                if metadata_model_type:
                    if AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE in model_config:
                        if (
                            model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE]
                            != metadata_model_type
                        ):
                            raise AquaRuntimeError(
                                f"The {AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE} attribute in {AQUA_MODEL_ARTIFACT_CONFIG}"
                                f" at {import_model_details.os_path} is invalid, expected {metadata_model_type} for "
                                f"the model {import_model_details.model}. Please check if the path is correct or "
                                f"the correct model artifacts are available at this location."
                                f""
                            )
                    else:
                        logger.debug(
                            f"Could not find {AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE} attribute in "
                            f"{AQUA_MODEL_ARTIFACT_CONFIG}. Proceeding with model registration."
                        )
            except:
                pass

        # Copy the model name from the service model if `model` is ocid
        model_name = (
            verified_model_details.display_name
            if verified_model_details
            else import_model_details.model
        )

        # Create Model catalog entry with pass by reference
        ds_model = self._create_model_catalog_entry(
            os_path=import_model_details.os_path,
            model_name=model_name,
            inference_container=import_model_details.inference_container,
            finetuning_container=import_model_details.finetuning_container,
            verified_model=verified_model_details,
            compartment_id=import_model_details.compartment_id,
            project_id=import_model_details.project_id,
        )
        # registered model will always have inference and evaluation container, but
        # fine-tuning container may be not set
        inference_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.DEPLOYMENT_CONTAINER
        ).value
        evaluation_container = ds_model.custom_metadata_list.get(
            ModelCustomMetadataFields.EVALUATION_CONTAINER,
        ).value
        try:
            finetuning_container = ds_model.custom_metadata_list.get(
                ModelCustomMetadataFields.FINETUNE_CONTAINER,
            ).value
        except:
            finetuning_container = None

        aqua_model_attributes = dict(
            **self._process_model(ds_model, self.region),
            project_id=ds_model.project_id,
            model_card=str(
                read_file(
                    file_path=f"{import_model_details.os_path.rstrip('/')}/{README}",
                    auth=default_signer(),
                )
            ),
            inference_container=inference_container,
            finetuning_container=finetuning_container,
            evaluation_container=evaluation_container,
        )

        if verified_model_details:
            telemetry_model_name = model_name
        else:
            if AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME in model_config:
                telemetry_model_name = f"{AQUA_MODEL_TYPE_CUSTOM}_{model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_NAME]}"
            elif AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE in model_config:
                telemetry_model_name = f"{AQUA_MODEL_TYPE_CUSTOM}_{model_config[AQUA_MODEL_ARTIFACT_CONFIG_MODEL_TYPE]}"
            else:
                telemetry_model_name = AQUA_MODEL_TYPE_CUSTOM

        self.telemetry.record_event_async(
            category="aqua/model",
            action="register",
            detail=telemetry_model_name,
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
        else:
            raise ValueError(
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

    def _build_search_text(self, tags: dict, description: str = None) -> str:
        """Constructs search_text field in response."""
        description = description or ""
        tags_text = (
            ",".join(str(v) for v in tags.values()) if isinstance(tags, dict) else ""
        )
        separator = " " if description else ""
        return f"{description}{separator}{tags_text}"

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
            raise AquaRuntimeError("Failed to get artifact path from custom metadata.")

        content = str(
            read_file(
                file_path=f"{os.path.dirname(artifact_path)}/{LICENSE_TXT}",
                auth=default_signer(),
            )
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
                return aqua_model_summary.id

        return None

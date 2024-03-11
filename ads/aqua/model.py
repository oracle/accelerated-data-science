#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import json
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import List, Union

import oci
from cachetools import TTLCache
from oci.data_science.models import JobRun, Model

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
from ads.aqua.constants import (
    TRAINING_METRICS_FINAL,
    TRINING_METRICS,
    UNKNOWN_VALUE,
    VALIDATION_METRICS,
    VALIDATION_METRICS_FINAL,
    FineTuningDefinedMetadata,
)
from ads.aqua.data import AquaResourceIdentifier, Tags
from ads.aqua.exception import AquaRuntimeError
from ads.aqua.utils import README, UNKNOWN, create_word_icon, read_file
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link
from ads.config import (
    COMPARTMENT_OCID,
    ODSC_MODEL_COMPARTMENT_OCID,
    PROJECT_OCID,
    TENANCY_OCID,
)
from ads.model import DataScienceModel
from ads.model.model_metadata import MetadataTaxonomyKeys, ModelCustomMetadata


class FineTuningMetricCategories(Enum):
    VALIDATION = "validation"
    TRAINING = "training"


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
class AquaFineTuningMetric(DataClassSerializable):
    name: str = field(default_factory=str)
    category: str = field(default_factory=str)
    scores: list[dict] = field(default_factory=list)


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


@dataclass(repr=False)
class AquaModel(AquaModelSummary, DataClassSerializable):
    """Represents an Aqua model."""

    model_card: str = None


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

        loggroup_url = (
            f"https://cloud.oracle.com/logging/log-groups/{loggroup_id}?region={region}"
            if loggroup_id
            else ""
        )

        log_url = (
            f"https://cloud.oracle.com/logging/log-groups/{loggroup_id}/logs/{log_id}?region={region}"
            if (loggroup_id and log_id)
            else ""
        )
        log_name = None
        loggroup_name = None

        if log_id:
            try:
                log = utils.query_resource(log_id, return_all=False)
                log_name = log.display_name if log else ""
            except:
                pass

        if loggroup_id:
            try:
                loggroup = utils.query_resource(loggroup_id, return_all=False)
                loggroup_name = loggroup.display_name if loggroup else ""
            except:
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
            utils.LIFECYCLE_DETAILS_MISSING_JOBRUN
            if not jobrun
            else jobrun.lifecycle_details
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
                f"Failed to extract model hyperparameters from {model.id}:" f"{str(e)}"
            )
            model_hyperparameters = {}

        self.dataset = model_hyperparameters.get(
            FineTuningDefinedMetadata.TRAINING_DATA.value
        )
        if not self.dataset:
            logger.debug(
                f"Key={FineTuningDefinedMetadata.TRAINING_DATA.value} not found in model hyperparameters."
            )

        self.validation = AquaFineTuneValidation(
            value=model_hyperparameters.get(
                FineTuningDefinedMetadata.VAL_SET_SIZE.value
            )
        )
        if not self.validation:
            logger.debug(
                f"Key={FineTuningDefinedMetadata.VAL_SET_SIZE.value} not found in model hyperparameters."
            )


# TODO: merge metadata key used in create FT


class FineTuningCustomMetadata(Enum):
    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "train_metrics_final"
    VALIDATION_METRICS_FINAL = "val_metrics_final"
    TRAINING_METRICS_EPOCH = "train_metrics_epoch"
    VALIDATION_METRICS_EPOCH = "val_metrics_epoch"


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
            .with_artifact(service_model.artifact)
            .with_display_name(service_model.display_name)
            .with_description(service_model.description)
            .with_freeform_tags(**(service_model.freeform_tags or {}))
            .with_defined_tags(**(service_model.defined_tags or {}))
            .with_model_version_set_id(service_model.model_version_set_id)
            .with_version_label(service_model.version_label)
            .with_custom_metadata_list(service_model.custom_metadata_list)
            .with_defined_metadata_list(service_model.defined_metadata_list)
            .with_provenance_metadata(service_model.provenance_metadata)
            # TODO: decide what kwargs will be needed.
            .create(model_by_reference=True, **kwargs)
        )
        logger.debug(
            f"Aqua Model {custom_model.id} created with the service model {model_id}"
        )
        return custom_model

    def get(self, model_id) -> "AquaModel":
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
            return cached_item

        ds_model = DataScienceModel.from_id(model_id)

        if not self._if_show(ds_model):
            raise AquaRuntimeError(f"Target model `{ds_model.id} `is not Aqua model.")

        is_fine_tuned_model = (
            True
            if ds_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
            else False
        )

        try:
            artifact_path = ds_model.custom_metadata_list.get(
                utils.MODEL_BY_REFERENCE_OSS_PATH_KEY
            ).value
        except ValueError:
            artifact_path = utils.UNKNOWN

        if not artifact_path:
            logger.debug("Failed to get artifact path from custom metadata.")

        aqua_model_atttributes = dict(
            **self._process_model(ds_model, self.region),
            project_id=ds_model.project_id,
            model_card=str(
                read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
            ),
        )

        if not is_fine_tuned_model:
            model_details = AquaModel(**aqua_model_atttributes)
            self._service_model_details_cache.__setitem__(
                key=model_id, value=model_details
            )

        else:
            jobrun_ocid = ds_model.provenance_metadata.training_id
            jobrun = self.ds_client.get_job_run(jobrun_ocid).data

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
                **aqua_model_atttributes,
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
            for custom_metadata in custom_metadata_list:
                # We use description to group metrics
                if custom_metadata.description == target:
                    scores.append(json.loads(custom_metadata.value))
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
            category=FineTuningMetricCategories.VALIDATION.value,
            metric_name=VALIDATION_METRICS,
        )

        training_metrics = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_EPOCH,
            category=FineTuningMetricCategories.TRAINING.value,
            metric_name=TRINING_METRICS,
        )

        validation_final = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.VALIDATION_METRICS_FINAL,
            category=FineTuningMetricCategories.VALIDATION.value,
            metric_name=VALIDATION_METRICS_FINAL,
        )

        training_final = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_FINAL,
            category=FineTuningMetricCategories.TRAINING.value,
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

        return dict(
            compartment_id=model.compartment_id,
            icon=icon or UNKNOWN,
            id=model_id,
            license=model.freeform_tags.get(Tags.LICENSE.value, UNKNOWN),
            name=model.display_name,
            organization=model.freeform_tags.get(Tags.ORGANIZATION.value, UNKNOWN),
            task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
            time_created=model.time_created,
            is_fine_tuned_model=(
                True
                if model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
                else False
            ),
            tags=tags,
            console_link=console_link,
            search_text=search_text,
        )

    def list(
        self, compartment_id: str = None, project_id: str = None, **kwargs
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
        **kwargs:
            Additional keyword arguments that can be used to filter the results.

        Returns
        -------
        List[AquaModelSummary]:
            The list of the `ads.aqua.model.AquaModelSummary`.
        """
        models = []
        if compartment_id:
            logger.info(f"Fetching custom models from compartment_id={compartment_id}.")
            models = self._rqs(compartment_id)
        else:
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

    def ModelCustomMetadata_process_model(
        self, model: Union["ModelSummary", "Model", "ResourceSummary"], region: str
    ) -> dict:
        """Constructs required fields for AquaModelSummary."""

        # todo: revisit icon generation code
        # icon = self._load_icon(model.display_name)
        icon = ""
        tags = {}
        tags.update(model.defined_tags or {})
        tags.update(model.freeform_tags or {})

        model_id = (
            model.id
            if (
                isinstance(model, oci.data_science.models.ModelSummary)
                or isinstance(model, oci.data_science.models.model.Model)
            )
            else model.identifier
        )
        console_link = (
            get_console_link(
                resource="models",
                ocid=model_id,
                region=region,
            ),
        )
        # TODO: build search_text with description
        search_text = self._build_search_text(tags) if tags else UNKNOWN

        return dict(
            compartment_id=model.compartment_id,
            icon=icon or UNKNOWN,
            id=model_id,
            license=model.freeform_tags.get(Tags.LICENSE.value, UNKNOWN),
            name=model.display_name,
            organization=model.freeform_tags.get(Tags.ORGANIZATION.value, UNKNOWN),
            task=model.freeform_tags.get(Tags.TASK.value, UNKNOWN),
            time_created=model.time_created,
            is_fine_tuned_model=(
                True
                if model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
                else False
            ),
            tags=tags,
            console_link=console_link,
            search_text=search_text,
        )

    def _if_show(self, model: "AquaModel") -> bool:
        """Determine if the given model should be return by `list`."""
        TARGET_TAGS = model.freeform_tags.keys()
        return (
            Tags.AQUA_TAG.value in TARGET_TAGS
            or Tags.AQUA_TAG.value.lower() in TARGET_TAGS
        )

    def _load_icon(self, model_name: str) -> str:
        """Loads icon."""

        # TODO: switch to the official logo
        try:
            return create_word_icon(model_name, return_as_datauri=True)
        except Exception as e:
            logger.debug(f"Failed to load icon for the model={model_name}: {str(e)}.")
            return None

    def _rqs(self, compartment_id: str, **kwargs):
        """Use RQS to fetch models in the user tenancy."""

        condition_tags = f"&& (freeformTags.key = '{Tags.AQUA_TAG.value}' && freeformTags.key = '{Tags.AQUA_FINE_TUNED_MODEL_TAG.value}')"
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import List, Union

import oci
from cachetools import TTLCache

from ads.aqua import logger, utils
from ads.aqua.base import AquaApp
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
from ads.model.model_metadata import ModelCustomMetadata


class FineTuningMetricCategories(Enum):
    VALIDATION = "validation"
    TRAINING = "training"


@dataclass(repr=False)
class FineTuningShapeInfo(DataClassSerializable):
    instance_shape: str = field(default_factory=str)
    replica: int = field(default_factory=int)


class AquaFineTuningMetricScore(DataClassSerializable):
    epoch: float = ""
    step: int = ""
    score: float = field(default_factory=float)


@dataclass(repr=False)
class AquaFineTuningMetric(DataClassSerializable):
    name: str = field(default_factory=str)
    category: str = field(default_factory=str)
    scores: list[AquaFineTuningMetricScore] = field(default_factory=list)


@dataclass(repr=False)
class AquaModelSummary(DataClassSerializable):
    """Represents a summary of Aqua model."""

    compartment_id: str
    icon: str
    id: str
    is_fine_tuned_model: bool
    license: str
    name: str
    organization: str
    project_id: str
    tags: dict
    task: str
    time_created: str
    console_link: str
    search_text: str


@dataclass(repr=False)
class AquaModel(AquaModelSummary, DataClassSerializable):
    """Represents an Aqua model."""

    model_card: str


@dataclass(repr=False)
class AquaFineTuneModel(AquaModel, DataClassSerializable):
    """Represents an Aqua Fine Tuned Model."""

    job: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    source: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    experiment: AquaResourceIdentifier = field(default_factory=AquaResourceIdentifier)
    shape_info: FineTuningShapeInfo = field(default_factory=FineTuningShapeInfo)
    metrics: List[AquaFineTuningMetric] = field(default_factory=list)


# TODO: merge metadata key used in create FT


class FineTuningCustomMetadata(Enum):
    FT_SOURCE = "fine_tune_source"
    FT_SOURCE_NAME = "fine_tune_source_name"
    FT_OUTPUT_PATH = "fine_tune_output_path"
    FT_JOB_ID = "fine_tune_job_id"
    FT_JOB_RUN_ID = "fine_tune_jobrun_id"
    TRAINING_METRICS_FINAL = "training_metrics_final"
    VALIDATION_METRICS_FINAL = "validation_metrics_final"

    TRAINING_METRICS_EPOCH = "training_metrics_"
    VALIDATION_METRICS_EPOCH = "validation_metrics_"


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
        # TODO: add cache for service model details
        ds_model = DataScienceModel.from_id(model_id)

        if not self._if_show(ds_model):
            raise AquaRuntimeError(f"Target model {ds_model.id} is not Aqua model.")

        artifact_path = ds_model.custom_metadata_list.get(
            utils.MODEL_BY_REFERENCE_OSS_PATH_KEY, utils.UNKNOWN
        )
        if not artifact_path:
            logger.debug("Failed to get artifact path from custom metadata.")

        aqua_model_atttributes = dict(
            **self._process_model(ds_model, self.region),
            project_id=ds_model.project_id,
            model_card=str(
                read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
            ),
        )

        is_fine_tuned_model = (
            True
            if ds_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
            else False
        )

        if not is_fine_tuned_model:
            model_details = AquaModel(**aqua_model_atttributes)
        else:
            jobrun_ocid = ds_model.provenance_metadata.training_id
            jobrun = self.ds_client.get_job_run(jobrun_ocid).data
            # shape info
            shape_info = FineTuningShapeInfo(
                instance_shape=jobrun.job_infrastructure_configuration_details.shape_name,
                # TODO: use variable for `NODE_COUNT` in ads/jobs/builders/runtimes/base.py
                replica=jobrun.job_configuration_override_details.environment_variables.get(
                    "NODE_COUNT"
                ),
            )

            jobrun_identifier = utils._build_resource_identifier(
                id=jobrun.id, name=jobrun.display_name, region=self.region
            )
            source_identifier = utils._build_resource_identifier(
                id=ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE
                ),
                name=ds_model.custom_metadata_list.get(
                    FineTuningCustomMetadata.FT_SOURCE_NAME
                ),
                region=self.region,
            )

            experiment_identifier = utils._build_resource_identifier(
                id=ds_model.model_version_set_id,
                name=ds_model.model_version_set_name,
                region=self.region,
            )
            ft_metrics = self._build_ft_metrics(ds_model.custom_metadata_list)

            model_details = AquaFineTuneModel(
                **aqua_model_atttributes,
                job=jobrun_identifier,
                source=source_identifier,
                experiment=experiment_identifier,
                shape_info=shape_info,
                metrics=ft_metrics,
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
                    # final loss/accuracy
                    metrics = json.loads(custom_metadata.value)
                    scores.append(
                        AquaFineTuningMetricScore(
                            epoch=metrics.get("epoch", ""),
                            step=metrics.get("step", ""),
                            score=(
                                metrics.get("rouge", [""])[0]
                                if metric_name == "validation_accuracy"
                                else metrics.get("loss", "")
                            ),
                        )
                    )
                    if (
                        metric_name == "validation_accuracy"
                        or metric_name == "final_training_loss"
                    ):
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

        validation_loss = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.VALIDATION_METRICS_EPOCH,
            category=FineTuningMetricCategories.VALIDATION.value,
            metric_name="validation_loss",
        )

        training_loss = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_EPOCH,
            category=FineTuningMetricCategories.TRAINING.value,
            metric_name="training_loss",
        )
        validation_accuracy = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.VALIDATION_METRICS_FINAL,
            category=FineTuningMetricCategories.VALIDATION.value,
            metric_name="validation_accuracy",
        )
        final_training_loss = self._fetch_metric_from_metadata(
            custom_metadata_list=custom_metadata_list,
            target=FineTuningCustomMetadata.TRAINING_METRICS_FINAL,
            category=FineTuningMetricCategories.TRAINING.value,
            metric_name="final_training_loss",
        )

        return [
            validation_loss,
            training_loss,
            validation_accuracy,
            final_training_loss,
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
            models = self.list_resource(
                self.ds_client.list_models,
                compartment_id=ODSC_MODEL_COMPARTMENT_OCID,
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

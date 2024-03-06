#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import List, Union

import oci
from cachetools import TTLCache

from ads.aqua import logger
from ads.aqua.base import AquaApp
from ads.aqua.data import AquaResourceIdentifier, Resource, Tags
from ads.aqua.exception import AquaRuntimeError
from ads.aqua.utils import (
    README,
    UNKNOWN,
    create_word_icon,
    get_artifact_path,
    read_file,
)
from ads.common.oci_resource import SEARCH_TYPE, OCIResource
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link
from ads.config import (
    COMPARTMENT_OCID,
    ODSC_MODEL_COMPARTMENT_OCID,
    PROJECT_OCID,
    TENANCY_OCID,
)
from ads.model.datascience_model import DataScienceModel
from oci.data_science.models import Model


@dataclass(repr=False)
class AquaFineTuningMetric(DataClassSerializable):
    name: str
    category: str
    scores: list


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
    shape_info: dict = field(default_factory=dict)
    metrics: List[AquaFineTuningMetric] = field(default_factory=list)


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
        oci_model = self.ds_client.get_model(model_id).data

        if not self._if_show(oci_model):
            raise AquaRuntimeError(f"Target model {oci_model.id} is not Aqua model.")

        artifact_path = get_artifact_path(oci_model.custom_metadata_list)

        is_fine_tuned_model = (
            True
            if oci_model.freeform_tags.get(Tags.AQUA_FINE_TUNED_MODEL_TAG.value)
            else False
        )

        return (
            AquaModel(
                **self._process_model(oci_model, self.region),
                project_id=oci_model.project_id,
                model_card=str(
                    read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
                ),
            )
            if not is_fine_tuned_model
            else AquaFineTuneModel(
                **self._process_model(oci_model, self.region),
                project_id=oci_model.project_id,
                model_card=str(
                    read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
                ),
                # mock data for fine tuned model details
                # TODO: fetch real value from custom metadata
                job=AquaResourceIdentifier(
                    id="ocid1.datasciencejobrun.oc1.iad.xxxx",
                    name="Job Run Name",
                    url=get_console_link(
                        resource=Resource.JOBRUN.value,
                        ocid="ocid1.datasciencejobrun.oc1.iad.xxxx",
                        region=self.region,
                    ),
                ),
                source=AquaResourceIdentifier(
                    id="ocid1.datasciencemodel.oc1.iad.xxxx",
                    name="Base Model Name",
                    url=get_console_link(
                        resource=Resource.MODEL.value,
                        ocid="ocid1.datasciencemodel.oc1.iad.xxxx",
                        region=self.region,
                    ),
                ),
                experiment=AquaResourceIdentifier(
                    id="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
                    name="Model Version Set Name",
                    url=get_console_link(
                        resource=Resource.MODEL_VERSION_SET.value,
                        ocid="ocid1.datasciencemodelversionset.oc1.iad.xxxx",
                        region=self.region,
                    ),
                ),
                shape_info={"instance_shape": "VM.Standard.E4.Flex", "replica": 1},
                metrics=[
                    AquaFineTuningMetric(
                        **{
                            "name": "validation_loss",
                            "category": "validation",
                            "scores": [
                                {"epoch": 2.5, "step": 12, "score": 1.1149},
                                {"epoch": 3.5, "step": 20, "score": 1.1067},
                                # ...
                            ],
                        }
                    ),
                    AquaFineTuningMetric(
                        **{
                            "name": "training_loss",
                            "category": "training",
                            "scores": [
                                {"epoch": 1.0, "step": 4, "score": 1.3856},
                                {"epoch": 1.5, "step": 8, "score": 1.0992},
                                {"epoch": 3.0, "step": 3, "score": 0.9193},
                                {"epoch": 3.5, "step": 20, "score": 0.853},
                            ],
                        }
                    ),
                    AquaFineTuningMetric(
                        **{
                            "name": "validation_accuracy",
                            "category": "validation",
                            "scores": [
                                # accuracy will be stored in "val_metrics_final"
                                # Before we finalize the accuracy, we can use the rouge1 score as an example.
                                # There will be only one number without epoch/step
                                {"score": 29.1849}
                            ],
                        }
                    ),
                    AquaFineTuningMetric(
                        **{
                            "name": "final_training_loss",
                            "category": "training",
                            "scores": [{"score": 1.0474}],
                        }
                    ),
                ],
            )
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

    def _process_model(
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

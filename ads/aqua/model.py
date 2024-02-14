#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime, timedelta
from threading import Lock

import cachetools
from cachetools import TTLCache
from cachetools import cached
from cachetools.keys import hashkey

import oci
from ads.aqua import logger
from ads.aqua.base import AquaApp
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
from ads.config import COMPARTMENT_OCID, ODSC_MODEL_COMPARTMENT_OCID, TENANCY_OCID
from ads.model.datascience_model import DataScienceModel


class Tags(Enum):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"


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

    @staticmethod
    def load_icon(model_name) -> str:
        """Loads icon."""
        # TODO: switch to the official logo
        try:
            return create_word_icon(model_name, return_as_datauri=True)
        except Exception as e:
            logger.debug(f"Failed to load icon for the model={model_name}.")
            return None


@dataclass(repr=False)
class AquaModel(AquaModelSummary, DataClassSerializable):
    """Represents an Aqua model."""

    model_card: str


class AquaModelApp(AquaApp):
    """Contains APIs for Aqua model.

    Attributes
    ----------

    Methods
    -------
    create(self, **kwargs) -> "AquaModel"
        Creates an instance of Aqua model.
    get(..., **kwargs)
        Gets the information of an Aqua model.
    list(...) -> List["AquaModelSummary"]
        List existing models created via Aqua.
    """

    def create(
        self, model_id: str, project_id: str, comparment_id: str = None, **kwargs
    ) -> "AquaModel":
        """Creates custom aqua model from service model.

        Parameters
        ----------
        model_id: str
            The service model id.
        project_id: str
            The project id for custom model.
        comparment_id: str
            The compartment id for custom model. Defaults to None.
            If not provided, compartment id will be fetched from environment variables.

        Returns
        -------
        AquaModel:
            The instance of AquaModel.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                service_model = DataScienceModel.from_id(model_id)
                service_model.download_artifact(target_dir=temp_dir)
                custom_model = (
                    DataScienceModel()
                    .with_compartment_id(comparment_id or COMPARTMENT_OCID)
                    .with_project_id(project_id)
                    .with_artifact(temp_dir)
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
                    .create(**kwargs)
                )
            except Exception as e:
                # TODO: adjust error raising
                logger.error(f"Failed to create model from the given id {model_id}.")
                raise e

            artifact_path = get_artifact_path(
                custom_model.dsc_model.custom_metadata_list
            )
            return AquaModel(
                **AquaModelApp.process_model(custom_model.dsc_model, self.region),
                project_id=custom_model.project_id,
                model_card=str(
                    read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
                ),
            )

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

        return AquaModel(
            **AquaModelApp.process_model(oci_model, self.region),
            project_id=oci_model.project_id,
            model_card=str(
                read_file(file_path=f"{artifact_path}/{README}", auth=self._auth)
            ),
        )

    @cached(
        cache=TTLCache(maxsize=10, ttl=timedelta(minutes=1), timer=datetime.now),
        key=lambda _, compartment_id, project_id: hashkey(compartment_id, project_id),
        lock=Lock(),
        info=True,
    )
    def list(
        self,
        compartment_id: str = None,
        project_id: str = None,
    ) -> List["AquaModelSummary"]:
        """List Aqua models in a given compartment and under certain project.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
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
            logger.info(
                f"Fetching service model from compartment_id={ODSC_MODEL_COMPARTMENT_OCID}"
            )
            models = self.list_resource(
                self.ds_client.list_models, compartment_id=ODSC_MODEL_COMPARTMENT_OCID
            )

        logger.info(
            f"Fetch {len(models)} model in compartment_id={compartment_id or ODSC_MODEL_COMPARTMENT_OCID}."
        )

        aqua_models = []
        # TODO: build index.json for service model as caching if needed.

        for model in models:
            aqua_models.append(
                AquaModelSummary(
                    **AquaModelApp.process_model(model=model, region=self.region),
                    project_id=project_id or UNKNOWN,
                )
            )

        return aqua_models

    def clear_model_list_cache(self):
        self.list.cache_clear()
        logger.info(f"Cache usage: {self.list.cache_info()}")
        return {"cache_deleted": True}

    @classmethod
    def process_model(cls, model, region) -> dict:
        icon = AquaModelSummary.load_icon(model.display_name)
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
        )

    def _if_show(self, model: "AquaModel") -> bool:
        """Determine if the given model should be return by `list`."""
        TARGET_TAGS = model.freeform_tags.keys()
        return (
            Tags.AQUA_TAG.value in TARGET_TAGS
            or Tags.AQUA_TAG.value.lower() in TARGET_TAGS
        )

    def _rqs(self, compartment_id):
        """Use RQS to fetch models in the user tenancy."""
        condition_tags = f"&& (freeformTags.key = '{Tags.AQUA_TAG.value}' && freeformTags.key = '{Tags.AQUA_FINE_TUNED_MODEL_TAG.value}')"
        condition_lifecycle = "&& lifecycleState = 'ACTIVE'"
        query = f"query datasciencemodel resources where (compartmentId = '{compartment_id}' {condition_lifecycle} {condition_tags})"
        logger.info(query)
        logger.info(f"tenant_id={TENANCY_OCID}")
        return OCIResource.search(
            query,
            type=SEARCH_TYPE.STRUCTURED,
            tenant_id=TENANCY_OCID,
        )

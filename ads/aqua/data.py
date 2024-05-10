#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from dataclasses import InitVar, dataclass
from enum import Enum

from ads.aqua import logger
from ads.aqua.utils import CONSOLE_LINK_RESOURCE_TYPE_MAPPING, get_resource_type
from ads.common.extended_enum import ExtendedEnum
from ads.common.serializer import DataClassSerializable
from ads.common.utils import get_console_link


class Resource(Enum):
    JOB = "jobs"
    JOBRUN = "jobruns"
    MODEL = "models"
    MODEL_DEPLOYMENT = "modeldeployments"
    MODEL_VERSION_SET = "model-version-sets"


class DataScienceResource(Enum):
    MODEL_DEPLOYMENT = "datasciencemodeldeployment"
    MODEL = "datasciencemodel"


class Tags(Enum):
    TASK = "task"
    LICENSE = "license"
    ORGANIZATION = "organization"
    AQUA_TAG = "OCI_AQUA"
    AQUA_SERVICE_MODEL_TAG = "aqua_service_model"
    AQUA_FINE_TUNED_MODEL_TAG = "aqua_fine_tuned_model"
    AQUA_MODEL_NAME_TAG = "aqua_model_name"
    AQUA_EVALUATION = "aqua_evaluation"
    AQUA_FINE_TUNING = "aqua_finetuning"
    READY_TO_FINE_TUNE = "ready_to_fine_tune"
    READY_TO_IMPORT = "ready_to_import"
    BASE_MODEL_CUSTOM = "aqua_custom_base_model"


class InferenceContainerType(ExtendedEnum):
    CONTAINER_TYPE_VLLM = "vllm"
    CONTAINER_TYPE_TGI = "tgi"


class InferenceContainerTypeKey(ExtendedEnum):
    AQUA_VLLM_CONTAINER_KEY = "odsc-vllm-serving"
    AQUA_TGI_CONTAINER_KEY = "odsc-tgi-serving"


class InferenceContainerParamType(ExtendedEnum):
    PARAM_TYPE_VLLM = "VLLM_PARAMS"
    PARAM_TYPE_TGI = "TGI_PARAMS"


@dataclass(repr=False)
class AquaResourceIdentifier(DataClassSerializable):
    """
    Data class representing a resource identifier.

    Attributes
    ----------
    id: str
        Unique identifier of the resource.
    name: str
        The display name of the resource.
    url: str
        The link to view the resource in console.
    """

    id: str = ""
    name: str = ""
    url: str = ""

    region: InitVar = None

    def __post_init__(self, region):
        if self.id and not self.url:
            try:
                resource_type = CONSOLE_LINK_RESOURCE_TYPE_MAPPING.get(
                    get_resource_type(self.id)
                )
                self.url = get_console_link(
                    resource=resource_type,
                    ocid=self.id,
                    region=region,
                )
            except Exception as ex:
                logger.debug(
                    f"Failed to construct console url for the resource: id=`{id}`. "
                    f"DEBUG INFO: {str(ex)}"
                )

    @classmethod
    def from_data(cls, data: dict) -> "AquaResourceIdentifier":
        """
        Creates AquaResourceIdentifier instance from given data.

        Parameters
        ----------
        data: dict
            The data dict contains variable needed for create instance.

        Returns
        -------
        AquaResourceIdentifier
            The instance of AquaResourceIdentifier.
        """
        try:
            obj = cls(**data)
            return obj
        except Exception as ex:
            logger.debug(
                f"Failed to construct AquaResourceIdentifier for the resource from the given data=`{data}`. "
                f"DEBUG INFO: {str(ex)}"
            )
            return cls()

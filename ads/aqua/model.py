#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from typing import List

from oci.data_science.data_science_client import DataScienceClient

from ads.config import COMPARTMENT_OCID
from ads.jobs.builders.base import Builder
from ads.common import utils

logger = logging.getLogger(__name__)


# Freeform-tag/Define-tag
# key=OCI_AQUA (100 chars max), val = (256 chars max)

AQUA_TAG = "OCI_AQUA"
AQUA_SERVICE_MODEL_TAG = "aqua_service_model"


class AquaApp:
    @property
    def oci_client(self):
        """Gets OCI client."""
        if not hasattr(self, "_oci_client") or not self._oci_client:
            self._oci_client = DataScienceClient(**self.auth)
        return self._oci_client

    def list(self):
        pass


class AquaModelApp(AquaApp):
    @classmethod
    def list(
        cls, compartment_id: str = None, project_id: str = None, **kwargs
    ) -> List["AquaModelSummary"]:
        """List Aqua models in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        project_id: (str, optional). Defaults to `None`.
            The project OCID.
        kwargs
            Additional keyword arguments for `list_models <https://docs.oracle.com/en-us/iaas/tools/python/2.118.1/api/data_science/client/oci.data_science.DataScienceClient.html#oci.data_science.DataScienceClient.list_models>`_

        Returns
        -------
        List[AquaModelSummary]
            The list of the Aqua models.
        """
        compartment_id = compartment_id or COMPARTMENT_OCID
        kwargs.update({"compartment_id": compartment_id, "project_id": project_id})

        try:
            # https://docs.oracle.com/en-us/iaas/tools/python-sdk-examples/2.118.1/datascience/list_models.py.html
            # list_call_get_all_results
            response = cls.oci_client.list_models(**kwargs)
            models = response.data
        except Exception as e:
            # show opc-request-id and status code
            logger.error(f"Failing to retreive models in the given compartment. {e}")
            return []

        aqua_models = []
        for model in models:  # ModelSummary
            if model.freeform_tags.contains(
                AQUA_TAG_KEY
            ) and not model.freeform_tags.contains(AQUA_SERVICE_MODEL_TAG_KEY):
                aqua_models.append(cls()._update_from_dsc_model_summary(model))
        return aqua_models


@dataclass
class AquaModelSummary:
    """
    No custom metadata in this model
    """

    name: str
    ocid: str
    time_created: int

    # From ADS or somewhere else
    icon: str

    # Following are embedded in tags
    # text-classification
    task: str
    license: str
    organization: str
    is_fine_tuned_model: bool
    # real freeform tags?
    tags: list


class AquaModel(AquaModelSummary):
    # content from readme.md
    model_card: str
    model_path: str
    metadata: dict

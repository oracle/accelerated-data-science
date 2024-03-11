#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

import oci

from ads import set_auth
from ads.aqua import logger
from ads.aqua.data import Tags
from ads.aqua.exception import AquaRuntimeError, AquaValueError
from ads.aqua.utils import load_config, logger
from ads.common import oci_client as oc
from ads.common.auth import default_signer
from ads.common.utils import extract_region
from ads.config import (
    AQUA_CONFIG_FOLDER,
    OCI_ODSC_SERVICE_ENDPOINT,
    OCI_RESOURCE_PRINCIPAL_VERSION,
)


class AquaApp:
    """Base Aqua App to contain common components."""

    def __init__(self) -> None:
        if OCI_RESOURCE_PRINCIPAL_VERSION:
            set_auth("resource_principal")
        self._auth = default_signer({"service_endpoint": OCI_ODSC_SERVICE_ENDPOINT})
        self.ds_client = oc.OCIClientFactory(**self._auth).data_science
        self.logging_client = oc.OCIClientFactory(**default_signer()).logging_management
        self.identity_client = oc.OCIClientFactory(**default_signer()).identity
        self.region = extract_region(self._auth)

    def list_resource(
        self,
        list_func_ref,
        **kwargs,
    ) -> list:
        """Generic method to list OCI Data Science resources.

        Parameters
        ----------
        list_func_ref : function
            A reference to the list operation which will be called.
        **kwargs :
            Additional keyword arguments to filter the resource.
            The kwargs are passed into OCI API.

        Returns
        -------
        list
            A list of OCI Data Science resources.
        """
        return oci.pagination.list_call_get_all_results(
            list_func_ref,
            **kwargs,
        ).data

    def if_artifact_exist(self, model_id: str, **kwargs) -> bool:
        """Checks if the artifact exists.

        Parameters
        ----------
        model_id : str
            The model OCID.
        **kwargs :
            Additional keyword arguments passed in head_model_artifact.

        Returns
        -------
        bool
            Whether the artifact exists.
        """

        try:
            response = self.ds_client.head_model_artifact(model_id=model_id, **kwargs)
            return True if response.status == 200 else False
        except oci.exceptions.ServiceError as ex:
            if ex.status == 404:
                logger.info(f"Artifact not found in model {model_id}.")
                return False

    def get_config(self, model_id: str, config_file_name: str) -> Dict:
        """Gets the config for the given Aqua model.
        Parameters
        ----------
        model_id: str
            The OCID of the Aqua model.
        config_file_name: str
            name of the config file

        Returns
        -------
        Dict:
            A dict of allowed configs.
        """
        oci_model = self.ds_client.get_model(model_id).data
        model_name = oci_model.display_name

        oci_aqua = (
            (
                Tags.AQUA_TAG.value in oci_model.freeform_tags
                or Tags.AQUA_TAG.value.lower() in oci_model.freeform_tags
            )
            if oci_model.freeform_tags
            else False
        )

        if not oci_aqua:
            raise AquaRuntimeError(f"Target model {oci_model.id} is not Aqua model.")

        # todo: currently loads config within ads, artifact_path will be an external bucket
        artifact_path = AQUA_CONFIG_FOLDER
        config = load_config(
            artifact_path,
            config_file_name=config_file_name,
        )

        if model_name not in config:
            logger.error(
                f"{config_file_name} does not have config details for model: {model_name}"
            )
            return {}

        return config[model_name]

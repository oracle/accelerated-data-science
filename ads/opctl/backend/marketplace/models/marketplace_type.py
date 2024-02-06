#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC
from enum import Enum
from typing import List
from ads.common.serializer import DataClassSerializable
from ads.opctl.backend.marketplace.models.ocir_details import OCIRDetails


class MarketplaceListingDetails(DataClassSerializable, ABC):
    def __init__(
        self,
        listing_id: str,
        compartment_id: str,
        version: str,
    ):
        self.listing_id = listing_id
        self.compartment_id = compartment_id
        self.version = version


class SecretStrategy(Enum):
    PROMPT = "prompt"
    AUTOMATIC = "automatic"


class HelmMarketplaceListingDetails(MarketplaceListingDetails):
    def __init__(
        self,
        listing_id: str,
        compartment_id: str,
        marketplace_version: str,
        ocir_details: OCIRDetails,
        helm_chart_tag: str,
        image_tag_pattern: List[str],
        helm_values: dict,
        helm_app_name: str,
        namespace: str,
        docker_registry_secret: str,
        secret_strategy: SecretStrategy,
    ):
        super().__init__(listing_id, compartment_id, marketplace_version)
        self.ocir_details = ocir_details
        self.compartment_id = compartment_id
        self.helm_values = helm_values
        self.helm_chart_tag = helm_chart_tag
        self.container_tag_pattern = image_tag_pattern
        self.helm_app_name = helm_app_name
        self.namespace = namespace
        self.docker_registry_secret = docker_registry_secret
        self.secret_strategy = secret_strategy

    @property
    def ocir_url(self):
        return self.ocir_details.ocir_url

    @property
    def helm_fully_qualified_url(self):
        return f"oci://{self.ocir_url}"

    @property
    def ocir_registry(self):
        return self.ocir_details.ocir_region_url

    @property
    def ocir_image_path(self):
        return self.ocir_details.path_in_tenancy

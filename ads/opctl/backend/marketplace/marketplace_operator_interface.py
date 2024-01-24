#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import ABC, abstractmethod
from enum import Enum

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from kubernetes.client import V1ServiceList

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
)


# TODO: Handle generic listings properly


class Status(Enum):
    FAILURE = 0
    SUCCESS = 1


class MarketplaceInterface(ABC):
    @abstractmethod
    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        pass

    @abstractmethod
    def get_oci_meta(self, operator_config: str, tags_map: Dict[str, str]) -> dict:
        pass

    @abstractmethod
    def finalise_installation(
        self,
        operator_config: str,
        status: Status,
        tags_map: Dict[str, str],
        kubernetes_service_list: "V1ServiceList",
    ):
        pass

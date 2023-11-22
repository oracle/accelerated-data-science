from abc import ABC
from enum import Enum
from typing import Any, Dict

from typing import List

from ads.common.extended_enum import ExtendedEnumMeta

from ads.common.serializer import DataClassSerializable


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
        ocir_repo: str,
        helm_chart_tag: str,
        container_tag_pattern: List[str],
        helm_values: dict,
        helm_app_name: str,
        namespace: str,
        docker_registry_secret: str,
        secret_strategy: SecretStrategy,
    ):
        super().__init__(listing_id, compartment_id, marketplace_version)
        self._ocir_repo_ = ocir_repo
        self.compartment_id = compartment_id
        self.helm_values = helm_values
        self.helm_chart_tag = helm_chart_tag
        self.container_tag_pattern = container_tag_pattern
        self.helm_app_name = helm_app_name
        self.namespace = namespace
        self.docker_registry_secret = docker_registry_secret
        self.secret_strategy = secret_strategy

    @property
    def ocir_fully_qualified_url(self):
        return self._ocir_repo_.rstrip("/")

    @property
    def helm_fully_qualified_url(self):
        return f"oci://{self.ocir_fully_qualified_url}"

    @property
    def ocir_registry(self):
        return self._ocir_repo_.split("/")[0]

    @property
    def ocir_image(self):
        return self.ocir_fully_qualified_url.split("/")[-1]

    @property
    def ocir_image_path(self):
        return "/".join(self.ocir_fully_qualified_url.split("/")[1:])

    @property
    def helm_fully_qualified_path(self):
        return f"{self.ocir_fully_qualified_url}:{self.helm_chart_tag}"

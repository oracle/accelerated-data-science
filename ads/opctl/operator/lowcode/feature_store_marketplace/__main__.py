import json
import sys

from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
)
from typing import Dict


class FeatureStoreOperatorRunner(MarketplaceOperatorRunner):
    VERSION = "v0.10.2"
    LISTING_ID = "ocid1.mktpublisting.oc1.iad.amaaaaaaclen5bqas6lyd5xler6fewsri5uascfhybhrtwjnh3fwyzbzaora"

    @staticmethod
    def __add_docker_registry_secret__(
        helm_values: dict, operator_config_spec: dict
    ) -> None:
        secret_name = operator_config_spec["clusterDetails"]["dockerRegistrySecretName"]
        helm_values["imagePullSecrets"] = [{"name": f"{secret_name}"}]

    @staticmethod
    def __get_spec_from_config__(operator_config: str):
        operator_config_json = json.loads(operator_config)
        return operator_config_json["spec"]

    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        helm_values = operator_config_spec["helmValues"]
        self.__add_docker_registry_secret__(helm_values, operator_config_spec)
        return HelmMarketplaceListingDetails(
            listing_id=self.LISTING_ID,
            helm_chart_name="feature-store-dp-api",
            container_name_pattern=["feature-store-api"],
            version=self.VERSION,
            helm_values=operator_config_spec["helmValues"],
            namespace=operator_config_spec["clusterDetails"]["namespace"],
            ocir_repo=operator_config_spec["ocirRepo"],
            compartment_id=operator_config_spec["marketplaceCompartmentId"],
            helm_app_name=operator_config_spec["helmAppName"],
        )

    def get_oci_meta(self, container_map: Dict[str, str], operator_config: str) -> dict:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        oci_meta = {
            "repo": operator_config_spec["ocirRepo"] + "/",
            "images": {
                "api": {
                    "image": container_map["feature-store-api"],
                    # TODO: fix after listing
                    # "tag": self.VERSION,
                    "tag": "1.0.2",
                }
            },
        }
        return oci_meta


if __name__ == "__main__":
    # print(sys.argv)
    FeatureStoreOperatorRunner().run(sys.argv)

import json
import sys

from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
    SecretStrategy,
)
from typing import Dict

from ads.opctl.operator.lowcode.feature_store_marketplace.const import LISTING_ID


class FeatureStoreOperatorRunner(MarketplaceOperatorRunner):
    @staticmethod
    def __get_spec_from_config__(operator_config: str):
        operator_config_json = json.loads(operator_config)
        return operator_config_json["spec"]

    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        helm_values = operator_config_spec["helm"]["values"]
        secret_name: str = operator_config_spec["clusterDetails"].get(
            "dockerRegistrySecretName", ""
        )
        secret_strategy = SecretStrategy.PROMPT
        helm_values["imagePullSecrets"] = [{"name": f"{secret_name}"}]

        return HelmMarketplaceListingDetails(
            listing_id=LISTING_ID,
            helm_chart_tag=operator_config_spec["version"],
            container_tag_pattern=[
                f"feature-store-api-{operator_config_spec['version']}"
            ],
            marketplace_version=operator_config_spec["version"],
            helm_values=helm_values,
            ocir_repo=operator_config_spec["ocirURL"],
            compartment_id=operator_config_spec["compartmentId"],
            helm_app_name=operator_config_spec["helm"]["appName"],
            docker_registry_secret=secret_name,
            namespace=operator_config_spec["clusterDetails"]["namespace"],
            secret_strategy=secret_strategy.PROMPT,
        )

    def get_oci_meta(self, container_map: Dict[str, str], operator_config: str) -> dict:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        repo = operator_config_spec["ocirURL"].split("/")[0]
        container_path = container_map[
            f"feature-store-api-{operator_config_spec['version']}"
        ]
        namespace = self.__get_spec_from_config__(operator_config)["ocirURL"].split(
            "/"
        )[1]
        oci_meta = {
            "repo": repo,
            "images": {
                "api": {
                    "image": "/" + namespace + "/" + container_path.split(":")[0],
                    "tag": container_path.split(":")[1],
                },
                "authoriser": {"image": "dummy", "tag": "dummy"},
            },
        }
        return oci_meta


if __name__ == "__main__":
    FeatureStoreOperatorRunner().run(sys.argv)

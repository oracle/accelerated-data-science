import json
import random
import string
import sys

from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
    SecretStrategy,
)
from typing import Dict


# helm install fs-dp-api-test oci://iad.ocir.io/idogsu2ylimg/test-listing   --version 1.0 --namespace feature-store  --values /home/hvrai/projects/feature-store-dataplane/feature-store-terraform/k8/example_values/values_custom.yaml
class FeatureStoreOperatorRunner(MarketplaceOperatorRunner):
    LISTING_ID = "ocid1.mktpublisting.oc1.iad.amaaaaaabiudgxyazaterzjaubwdvhf5r55zie7wg6ujfnuryuhuje3y5tkq"

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
        if not secret_name:
            secret_name = (
                "ocir-secret-"
                + "".join(random.choices(string.ascii_letters, k=4)).lower()
            )
            secret_strategy = SecretStrategy.AUTOMATIC
        helm_values["imagePullSecrets"] = [{"name": f"{secret_name}"}]
        # TODO: Revert after helidon
        return HelmMarketplaceListingDetails(
            listing_id=self.LISTING_ID,
            # helm_chart_name="feature-store-dp-api",
            helm_chart_tag="1.0",
            container_tag_pattern=["feature-store-dataplane-api"],
            marketplace_version="0.1",
            helm_values=helm_values,
            ocir_repo=operator_config_spec["ocirRepo"],
            compartment_id=operator_config_spec["compartmentId"],
            helm_app_name=operator_config_spec["helm"]["appName"],
            docker_registry_secret=secret_name,
            namespace=operator_config_spec["clusterDetails"]["namespace"],
            secret_strategy=secret_strategy.AUTOMATIC,
        )

    def get_oci_meta(self, container_map: Dict[str, str], operator_config: str) -> dict:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        oci_meta = {
            "repo": operator_config_spec["ocirRepo"].rstrip("/"),
            "images": {
                "api": {
                    "image": "",
                    # TODO: fix after listing
                    # "tag": self.VERSION,
                    "tag": container_map["feature-store-dataplane-api"].split(":")[1],
                }
            },
        }
        return oci_meta


if __name__ == "__main__":
    # print(sys.argv)
    FeatureStoreOperatorRunner().run(sys.argv)

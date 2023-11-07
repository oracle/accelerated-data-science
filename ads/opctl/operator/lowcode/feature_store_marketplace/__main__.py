import json
import sys

from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
)


class FeatureStoreOperatorRunner(MarketplaceOperatorRunner):
    def get_listing_details(self, operator_config: str) -> MarketplaceListingDetails:
        operator_config_json = json.loads(operator_config)
        operator_config_spec = operator_config_json['spec']
        return HelmMarketplaceListingDetails(
            listing_id="<listing_id>",
            name='fs-dp-api-test',
            chart='oci://iad.ocir.io/idogsu2ylimg/feature-store-dataplane-api/helm-chart/feature-store-dp-api',
            version="1.0",
            helm_values=operator_config_spec['helmValues'],
            cluster_id="<cluster id>",
            namespace=operator_config_spec['clusterDetails']['namespace'],
            ocir_repo="<ocir_repo>",

        )


if __name__ == "__main__":
    print(sys.argv)
    FeatureStoreOperatorRunner().run(sys.argv)

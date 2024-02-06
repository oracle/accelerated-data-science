#!/usr/bin/env python
# -*- coding: utf-8 -*--
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import sys

from ads.opctl.backend.marketplace.marketplace_utils import Color, StatusIcons
from ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils import (
    get_nlb_id_from_service,
    update_resource_manager_stack,
    apply_stack,
    prompt_security_rules,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.apigw_config import (
    APIGatewayConfig,
)
from kubernetes.client import (
    V1ServiceList,
)

from ads.opctl.backend.marketplace.marketplace_operator_interface import Status
from ads.opctl.backend.marketplace.marketplace_operator_runner import (
    MarketplaceOperatorRunner,
)

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
    SecretStrategy,
)
from typing import Dict

from ads.opctl.backend.marketplace.models.ocir_details import OCIRDetails
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
            image_tag_pattern=[
                f"feature-store-api-{operator_config_spec['version']}",
                f"feature-store-authoriser-{operator_config_spec['version']}",
            ],
            marketplace_version=operator_config_spec["version"],
            helm_values=helm_values,
            ocir_details=OCIRDetails(operator_config_spec["ocirURL"]),
            compartment_id=operator_config_spec["compartmentId"],
            helm_app_name=operator_config_spec["helm"]["appName"],
            docker_registry_secret=secret_name,
            namespace=operator_config_spec["clusterDetails"]["namespace"],
            secret_strategy=secret_strategy.PROMPT,
        )

    def get_oci_meta(self, operator_config: str, tags_map: Dict[str, str]) -> dict:
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        ocir_details = OCIRDetails(operator_config_spec["ocirURL"])
        image_tag = tags_map[f"feature-store-api-{operator_config_spec['version']}"]
        oci_meta = {
            "repo": ocir_details.repository_url,
            "images": {
                "api": {"image": f"/{ocir_details.image}", "tag": image_tag},
                "authoriser": {"image": "dummy", "tag": "dummy"},
            },
        }
        return oci_meta

    def finalise_installation(
        self,
        operator_config: str,
        status: Status,
        tags_map: Dict[str, str],
        kubernetes_service_list: V1ServiceList,
    ):
        if status == Status.FAILURE:
            return
        operator_config_spec = self.__get_spec_from_config__(operator_config)
        ocir_details = OCIRDetails(operator_config_spec["ocirURL"])
        apigw_config: APIGatewayConfig = APIGatewayConfig.from_dict(
            operator_config_spec["apiGatewayDeploymentDetails"]
        )
        if not apigw_config.enabled:
            return
        fn_tag = tags_map[f"feature-store-authoriser-{operator_config_spec['version']}"]
        nlb_id = get_nlb_id_from_service(kubernetes_service_list.items[0], apigw_config)
        update_resource_manager_stack(
            apigw_config,
            nlb_id,
            f"{ocir_details.ocir_url}:{fn_tag}",
        )
        apply_stack(apigw_config.stack_id)
        prompt_security_rules(apigw_config.stack_id)
        print(
            f"{Color.GREEN}Successfully completed API Gateway deployment. {Color.END}{StatusIcons.TADA}"
        )


if __name__ == "__main__":
    FeatureStoreOperatorRunner().run(sys.argv)

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

import click
from ads.opctl.operator.lowcode.feature_store_marketplace.models.apigw_config import (
    APIGatewayConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.operator_utils import (
    get_latest_listing_version,
    get_db_details,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.db_config import (
    DBConfig,
)
from ads.opctl.backend.marketplace.marketplace_utils import Color, print_heading
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator


def init(**kwargs: Dict) -> dict:
    """
    Generates operator config by the schema.

    Properties
    ----------
    kwargs: (Dict, optional).
        Additional key value arguments.

        - type: str
            The type of the operator.

    Returns
    -------
    str
        The YAML specification generated based on the schema.
    """
    print(
        "========= Feature store marketplace kubernetes cluster configuration ========="
    )

    print_heading("OCIR Configuration", colors=[Color.BOLD, Color.BLUE])
    compartment_id = click.prompt("Compartment Id")
    ocir_url: str = click.prompt(
        "URL of the OCIR repository where the images will be cloned from marketplace \n"
        "(format: {region}.ocir.io/{tenancy_namespace}/{repository})",
    )
    db_config: DBConfig = get_db_details()

    print_heading(
        f"Cluster configuration",
        colors=[Color.BOLD, Color.BLUE],
        prefix_newline_count=2,
    )
    helm_app_name = click.prompt("Helm app name", default="feature-store-api")
    kubernetes_namespace = click.prompt("Kubernetes namespace", default="default")
    version = click.prompt(
        "Version of feature store stack to install",
        default=get_latest_listing_version(compartment_id),
    )
    # api_gw_config = get_api_gw_details(compartment_id)
    api_gw_config = APIGatewayConfig()
    api_gw_config.enabled = False
    yaml_dict: Dict = YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example_dict(
        values={
            "helm.values.db": db_config.to_dict(),
            "helm.appName": helm_app_name,
            "clusterDetails.namespace": kubernetes_namespace,
            "compartmentId": compartment_id,
            "ocirURL": f"{ocir_url.rstrip('/')}",
            "spec.version": version,
            "apiGatewayDeploymentDetails": api_gw_config.to_dict(),
        },
        required_keys=[],
    )

    return yaml_dict

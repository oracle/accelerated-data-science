#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

import click
from ads.common.auth import AuthContext
from ads.opctl.operator.lowcode.feature_store_marketplace.const import LISTING_ID

from ads.opctl.operator.lowcode.feature_store_marketplace.models.db_config import (
    DBConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.prompts import get_db_details

from ads.opctl.backend.marketplace.marketplace_utils import Color, print_heading
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.common.operator_yaml_generator import YamlGenerator
import oci.marketplace

from ads.common.oci_client import OCIClientFactory
from ads.common import auth as authutil


def init(**kwargs: Dict) -> dict:
    def get_latest_version() -> str:
        marketplace_client = OCIClientFactory(
            **authutil.default_signer()
        ).create_client(oci.marketplace.MarketplaceClient)
        listing: oci.marketplace.models.Listing = marketplace_client.get_listing(
            LISTING_ID, compartment_id=compartment_id
        ).data
        return listing.default_package_version

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
    ocir_image = click.prompt("OCIR image name", default="feature-store-api")

    db_config: DBConfig = get_db_details()

    print_heading(
        f"Cluster configuration",
        colors=[Color.BOLD, Color.BLUE],
        prefix_newline_count=2,
    )
    helm_app_name = click.prompt("Helm app name", default="feature-store-api")
    kubernetes_namespace = click.prompt("Kubernetes namespace", default="feature-store")
    version = click.prompt(
        "Version of feature store stack to install", default=get_latest_version()
    )
    yaml_dict: Dict = YamlGenerator(
        schema=_load_yaml_from_uri(__file__.replace("cmd.py", "schema.yaml"))
    ).generate_example_dict(
        values={
            "helm.values.db": db_config.to_dict(),
            "helm.appName": helm_app_name,
            "clusterDetails.namespace": kubernetes_namespace,
            "compartmentId": compartment_id,
            "ocirURL": f"{ocir_url.rstrip('/')}/{ocir_image}",
            "version": version,
        }
    )
    return yaml_dict

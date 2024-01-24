#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import base64
import json
import re
import subprocess
import time
import webbrowser
from typing import List

import click
from ads.opctl.backend.marketplace.models.bearer_token import BEARER_TOKEN_USERNAME

from ads.opctl import logger

from ads.opctl.backend.marketplace.models.marketplace_type import (
    MarketplaceListingDetails,
    HelmMarketplaceListingDetails,
    SecretStrategy,
)
import oci.marketplace.models as models

from ads.opctl.backend.marketplace.marketplace_utils import (
    get_marketplace_client,
    Color,
    StatusIcons,
    get_docker_bearer_token,
    print_heading,
    WARNING,
)

DOCKER_SECRET_TYPE = "kubernetes.io/dockerconfigjson"


class BinaryValidation:
    def __init__(self, binary: str, installation_path: str):
        self.binary = binary
        self.installation_path = installation_path


def get_highlighted_text(text: str, colors: List[Color] = (Color.BLUE,)) -> str:
    colors = [f"{color}" for color in colors]
    colors = " ".join(colors)
    return f"{colors}'{text}'{Color.END}"


def check_prerequisites(listing_details: MarketplaceListingDetails):
    print_heading(f"Checking Prerequisites", colors=[Color.BLUE, Color.BOLD])
    _check_license_for_listing_(listing_details)
    if isinstance(listing_details, HelmMarketplaceListingDetails):
        _check_binaries_(
            [
                BinaryValidation("helm", "https://helm.sh/docs/intro/install/"),
                BinaryValidation("kubectl", "https://kubernetes.io/docs/tasks/tools/"),
            ]
        )
        _prompt_kubernetes_confirmation_()
        _check_kubernetes_secret_(listing_details)
    print_heading(
        f"Completed prerequisites check {StatusIcons.TADA}",
        colors=[Color.BLUE, Color.BOLD],
    )


def _prompt_kubernetes_confirmation_():
    print(
        f"{WARNING}{Color.RED}Please ensure that kubectl is connected to the exact cluster where the listing needs to be deployed. Failure to do so can lead to unintended consequences.{Color.END}"
    )
    click.confirm(
        text="Is it safe to proceed?",
        default=False,
        abort=True,
    )


def _check_kubernetes_secret_(listing_details: HelmMarketplaceListingDetails):
    from kubernetes.client import V1SecretList, V1Secret, V1ObjectMeta

    import kubernetes

    print(
        f"Starting docker registry secret verification for secret: {get_highlighted_text(listing_details.docker_registry_secret)} in namespace: {get_highlighted_text(listing_details.namespace)}"
    )
    create_secret = True
    kubernetes.config.load_kube_config()
    v1 = kubernetes.client.CoreV1Api()
    secrets: V1SecretList = v1.list_namespaced_secret(
        namespace=listing_details.namespace
    )
    v1.list_namespaced_service(
        namespace="feature-store",
    )
    secret_strategy: SecretStrategy = listing_details.secret_strategy
    for secret in secrets.items:
        secret: V1Secret = secret
        metadata: V1ObjectMeta = secret.metadata
        if (
            secret.type == DOCKER_SECRET_TYPE
            and metadata.name == listing_details.docker_registry_secret
        ):
            annotations: dict = metadata.annotations
            # Check expiry date with 3 minutes buffer to accommodate for pull time
            if int(annotations.get("expiry")) > (time.time() + 60 * 3):
                print(
                    f"Secret {get_highlighted_text(listing_details.docker_registry_secret)} exists in namespace {get_highlighted_text(listing_details.namespace)} {StatusIcons.CHECK}"
                )
                return
            else:
                print(
                    f"Operator generated docker secret {get_highlighted_text(listing_details.docker_registry_secret)} has expired.\nRecreating secret using bearer token"
                )
                v1.delete_namespaced_secret(
                    name=metadata.name, namespace=listing_details.namespace
                )
                secret_strategy = SecretStrategy.AUTOMATIC

    if secret_strategy == SecretStrategy.PROMPT:
        create_secret = click.confirm(
            text=f"Docker registry secret: {listing_details.docker_registry_secret} wasn't found in namespace: {listing_details.namespace}. Do you want to allow operator to automatically try to fix the issue by setting up bearer token authentication?",
            default=True,
            abort=True,
        )
    if create_secret:
        username = BEARER_TOKEN_USERNAME
        token = get_docker_bearer_token(ocir_repo=listing_details.ocir_registry)
        auth = base64.b64encode(f"{username}:{token.token}".encode("utf-8")).decode(
            "utf-8"
        )
        docker_config_dict = {
            "auths": {
                listing_details.ocir_registry: {
                    "username": username,
                    "password": token.token,
                    "auth": auth,
                }
            }
        }
        docker_config = base64.b64encode(
            json.dumps(docker_config_dict).encode("utf-8")
        ).decode("utf-8")
        secret = kubernetes.client.models.V1Secret(
            type=DOCKER_SECRET_TYPE,
            metadata=kubernetes.client.models.V1ObjectMeta(
                name=listing_details.docker_registry_secret,
                namespace=listing_details.namespace,
                annotations={"expiry": str(token.expires_in + int(time.time()))},
            ),
            data={".dockerconfigjson": docker_config},
        )
        response = v1.create_namespaced_secret(
            namespace=listing_details.namespace, body=secret
        )
        print(
            f"Successfully created secret {get_highlighted_text(listing_details.docker_registry_secret)} in namespace {get_highlighted_text(listing_details.namespace)} {StatusIcons.CHECK}"
        )


def _check_binaries_(binaries: List[BinaryValidation]):
    for binary in binaries:
        result = subprocess.run(["which", f"{binary.binary}"], capture_output=True)
        if result.returncode == 0:
            print(f"{binary.binary.capitalize()} is present {StatusIcons.CHECK}")
        else:
            print(
                f"{binary.binary} is not present in PATH {StatusIcons.CROSS}.\nFor installation instructions please check: {binary.installation_path}"
            )
            raise Exception


def _check_license_for_listing_(listing_details: MarketplaceListingDetails):
    compartment_id = listing_details.compartment_id
    package_version = listing_details.version
    listing_id = listing_details.listing_id
    marketplace = get_marketplace_client()
    listing: models.Listing = marketplace.get_listing(
        listing_id=listing_id, compartment_id=compartment_id
    ).data
    logger.debug(f"Checking license agreements for listing: {listing.name}")
    accepted_agreements: List[
        models.AcceptedAgreementSummary
    ] = marketplace.list_accepted_agreements(
        listing_id=listing_id,
        compartment_id=compartment_id,
    ).data

    agreement_summaries: List[models.AgreementSummary] = marketplace.list_agreements(
        listing_id=listing_id,
        package_version=package_version,
        compartment_id=listing_details.compartment_id,
    ).data

    accepted_agreement_ids: List[str] = []
    for accepted_agreement in accepted_agreements:
        accepted_agreement_ids.append(accepted_agreement.agreement_id)

    for agreement_summary in agreement_summaries:
        if agreement_summary.id not in accepted_agreement_ids:
            agreement: models.Agreement = marketplace.get_agreement(
                listing_id=listing_id,
                package_version=package_version,
                agreement_id=agreement_summary.id,
                compartment_id=listing_details.compartment_id,
            ).data
            print(
                f"Agreement from {get_highlighted_text(agreement.author)} with id: {agreement_summary.id} is not accepted. \
                Opening terms and conditions in default browser {StatusIcons.LOADING}"
            )
            webbrowser.open(agreement.content_url)
            time.sleep(1)
            answer = click.confirm(
                f'{re.sub("<.*?>", "", agreement.prompt)}',
                default=False,
            )
            if not answer:
                print_heading(
                    f"Agreement from author: {get_highlighted_text(agreement_summary.author)} with id: {agreement_summary.id} "
                    f"is rejected {StatusIcons.CROSS} "
                )
                raise Exception
            else:
                create_accepted_agreement_details: models.CreateAcceptedAgreementDetails = (
                    models.CreateAcceptedAgreementDetails()
                )
                create_accepted_agreement_details.agreement_id = agreement_summary.id
                create_accepted_agreement_details.listing_id = listing_id
                create_accepted_agreement_details.compartment_id = compartment_id
                create_accepted_agreement_details.signature = agreement.signature
                create_accepted_agreement_details.package_version = package_version
                marketplace.create_accepted_agreement(create_accepted_agreement_details)

        print(
            f"Agreement from author: {get_highlighted_text(agreement_summary.author)} with id: {agreement_summary.id} is accepted "
            f"{StatusIcons.CHECK} "
        )

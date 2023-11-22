import base64
import json
import re
import subprocess
import time
import webbrowser
from typing import List

import click
from kubernetes.client import V1SecretList, V1Secret, V1ObjectMeta

from ads.opctl.backend.marketplace.helm_helper import (
    check_helm_pull,
    run_helm_login,
    HelmPullStatus,
)
import kubernetes
from ads.opctl import logger

from ads.opctl.backend.marketplace.marketplace_type import (
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
)

DOCKER_SECRET_TYPE = "kubernetes.io/dockerconfigjson"


def check_prerequisites(listing_details: MarketplaceListingDetails):
    print("\n\n", "*" * 30, f"{Color.BOLD}Checking prerequisites{Color.END}", "*" * 30)
    _check_license_for_listing_(listing_details)
    if isinstance(listing_details, HelmMarketplaceListingDetails):
        _check_binaries_(["helm", "kubectl"])
        _prompt_kubernetes_confirmation_()
        _check_kubernetes_secret_(listing_details)
        _check_helm_login_(listing_details)
    print(
        "*" * 30,
        f"{Color.BOLD}Completed prerequisites check {StatusIcons.TADA}{Color.END}",
        "*" * 30,
        "\n\n",
    )


# def _check_docker():
#     docker.client.


def _prompt_kubernetes_confirmation_():
    print(
        f"{Color.RED}{Color.BOLD}WARNING:{Color.END}{Color.RED} Please ensure that kubectl is connected to the exact cluster where the listing needs to be deployed. Failure to do so can lead to unintended consequences.{Color.END}"
    )
    click.confirm(
        text="Is it safe to proceed?",
        default=False,
        abort=True,
    )


def _check_helm_login_(listing_details: HelmMarketplaceListingDetails):
    status = check_helm_pull(
        helm_chart_url=listing_details.helm_fully_qualified_url,
        version=listing_details.helm_chart_tag,
    )
    if status == HelmPullStatus.UNKNOWN_FAILURE:
        # Todo throw correct exception
        raise Exception
    elif status == HelmPullStatus.SUCCESS:
        print(f"Helm client is authenticated {StatusIcons.CHECK}")
        return
    elif status == HelmPullStatus.AUTHENTICATION_FAILURE:
        response = click.confirm(
            text="Helm is unable to access OCIR due to authentication failure. Do you want to allow operator to automatically try to fix the issue by setting up bearer token authentication?",
            abort=True,
            default=True,
        )
        if response:
            token = get_docker_bearer_token(listing_details.ocir_registry)
            run_helm_login(ocir_repo=listing_details.ocir_registry, token=token)
            status = check_helm_pull(
                helm_chart_url=listing_details.helm_fully_qualified_url,
                version=listing_details.helm_chart_tag,
            )
            if status != HelmPullStatus.SUCCESS:
                print(f"Unable to setup helm authentication. {StatusIcons.CROSS}")
                # Todo throw correct exception
                raise Exception
            else:
                print(f"Successfully setup authentication for Helm {StatusIcons.CHECK}")


def _check_kubernetes_secret_(listing_details: HelmMarketplaceListingDetails):
    create_secret = True
    kubernetes.config.load_kube_config()
    v1 = kubernetes.client.CoreV1Api()
    secrets: V1SecretList = v1.list_namespaced_secret(
        namespace=listing_details.namespace
    )
    for secret in secrets.items:
        secret: V1Secret = secret
        metadata: V1ObjectMeta = secret.metadata
        if (
            secret.type == DOCKER_SECRET_TYPE
            and metadata.name == listing_details.docker_registry_secret
        ):
            print(
                f"Secret with name {listing_details.docker_registry_secret} exists in namespace {listing_details.namespace} {StatusIcons.CHECK}"
            )
            return
    if listing_details.secret_strategy == SecretStrategy.PROMPT:
        create_secret = click.confirm(
            text=f"Docker registry secret: {listing_details.docker_registry_secret} wasn't found in namespace: {listing_details.namespace}. Do you want to allow operator to automatically try to fix the issue by setting up bearer token authentication?",
            default=True,
            abort=True,
        )
    if create_secret:
        username = "BEARER_TOKEN"
        password = get_docker_bearer_token(ocir_repo=listing_details.ocir_registry)
        auth = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
            "utf-8"
        )
        docker_config_dict = {
            "auths": {
                listing_details.ocir_registry: {
                    "username": username,
                    "password": password,
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
            ),
            data={".dockerconfigjson": docker_config},
        )

        v1.create_namespaced_secret(namespace=listing_details.namespace, body=secret)
        print(
            f"Successfully created secret {listing_details.docker_registry_secret} in namespace {listing_details.namespace} {StatusIcons.CHECK}"
        )


def _check_binaries_(binaries: List[str]):
    for binary in binaries:
        result = subprocess.run(["which", f"{binary}"], capture_output=True)
        if result.returncode == 0:
            print(f"{binary.capitalize()} is present {StatusIcons.CHECK}")
        else:
            print(f"{binary} is not present in PATH {StatusIcons.CROSS}")


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
                f"Agreement from {agreement.author} with id: {agreement_summary.id} is not accepted. \
                Opening terms and conditions in default browser {StatusIcons.LOADING}"
            )
            webbrowser.open(agreement.content_url)
            time.sleep(1)
            answer = click.confirm(
                f'{re.sub("<.*?>", "", agreement.prompt)}',
                default=False,
            )
            if not answer:
                print(
                    "*" * 30,
                    f"Agreement from author: {agreement_summary.author} with id: {agreement_summary.id} "
                    f"is rejected {StatusIcons.CROSS} "
                    "*" * 30,
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
            f"Agreement from author: {agreement_summary.author} with id: {agreement_summary.id} is accepted "
            f"{StatusIcons.CHECK} "
        )

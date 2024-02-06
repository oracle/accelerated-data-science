#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import oci
from typing import List, Dict
from ads.opctl.backend.marketplace.models.bearer_token import BearerToken
from ads.common.oci_client import OCIClientFactory
from ads.common import auth as authutil
from ads.opctl.backend.marketplace.models.marketplace_type import (
    HelmMarketplaceListingDetails,
)


class Color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


WARNING = f"{Color.RED}{Color.BOLD}WARNING: {Color.END}"


class StatusIcons:
    CHECK = "\u2705 "
    CROSS = "\u274C "
    LOADING = "\u274d "
    TADA = "\u2728 "


def print_ticker(text: str, iteration: int):
    ticker = ["/", "-", "\\"]
    text = f"{text} [{ticker[iteration % len(ticker)]}]"
    end = "\n"
    try:
        size = os.get_terminal_size()
        backspaces_needed = int((len(text) + 1) / size.columns) + 1
    except OSError:
        end = "\r"
    else:
        if iteration != 0:
            print(f"\033[{backspaces_needed}F", end="", flush=True)

    print(text, end=end, flush=True)


def print_heading(
    heading: str,
    prefix_newline_count: int = 0,
    suffix_newline_count: int = 0,
    colors: List[str] = (Color.BOLD,),
) -> None:
    colors = [f"{color}" for color in colors]
    colors = " ".join(colors)
    print(
        f"{colors}"
        + "\n" * prefix_newline_count
        + "*" * 30
        + heading
        + "*" * 30
        + "\n" * suffix_newline_count
        + f"{Color.END}"
    )


def set_kubernetes_session_token_env(profile: str = "DEFAULT") -> None:
    os.environ["OCI_CLI_AUTH"] = "security_token"
    os.environ["OCI_CLI_PROFILE"] = profile


def get_marketplace_client() -> oci.marketplace.MarketplaceClient:
    return OCIClientFactory(**authutil.default_signer()).marketplace


def get_marketplace_composite_client() -> (
    oci.marketplace.MarketplaceClientCompositeOperations
):
    return oci.marketplace.MarketplaceClientCompositeOperations(
        client=get_marketplace_client()
    )


def get_docker_bearer_token(ocir_repo: str) -> BearerToken:
    def get_ocir_url(repo: str):
        repo = repo.lstrip("https://")
        repo = repo.rstrip("/")
        repo = f"https://{repo}/20180419"
        return repo

    token_client: oci.BaseClient = OCIClientFactory(
        **authutil.default_signer(
            client_kwargs={
                "service": "docker",
                "service_endpoint": get_ocir_url(ocir_repo),
                "type_mapping": {"SecurityToken": BearerToken},
            },
        )
    ).create_client(oci.BaseClient)
    token: BearerToken = token_client.call_api(
        resource_path="/docker/token", method="GET", response_type="SecurityToken"
    ).data
    return token


def _export_helm_chart_(listing_details: HelmMarketplaceListingDetails):
    client = get_marketplace_client()
    export_listing_work_request: oci.marketplace.models.WorkRequest = (
        client.export_listing(
            listing_id=listing_details.listing_id,
            package_version=listing_details.version,
            export_package_details=oci.marketplace.models.ExportPackageDetails(
                compartment_id=listing_details.compartment_id,
                container_repository_path=listing_details.ocir_image_path,
            ),
        ).data
    )

    export_listing_work_request = oci.wait_until(
        client,
        client.get_work_request(export_listing_work_request.id),
        evaluate_response=lambda r: getattr(r.data, "status")
        and getattr(r.data, "status").lower() in ["succeeded", "failed"],
        wait_callback=lambda times_checked, _: print_ticker(
            "Waiting for marketplace export to finish", iteration=times_checked - 1
        ),
        max_interval_seconds=1,
    ).data
    if export_listing_work_request.status == "FAILED":
        print(f"Couldn't export images from marketplace to OCIR {StatusIcons.CROSS}")
        # TODO: Raise proper exception
        raise Exception
    else:
        print(
            f"Images were successfully exported to OCIR from marketplace {StatusIcons.CHECK}"
        )
    # Get the data from response


def list_container_images(
    compartment_id: str, ocir_image_path: str
) -> oci.artifacts.models.ContainerImageCollection:
    artifact_client = OCIClientFactory(**authutil.default_signer()).artifacts
    list_container_images_response = artifact_client.list_container_images(
        compartment_id=compartment_id,
        sort_by="TIMECREATED",
        repository_name=ocir_image_path,
    )
    return list_container_images_response.data


def export_if_tags_not_exist(
    listing_details: HelmMarketplaceListingDetails,
) -> Dict[str, str]:
    class ImageTagPatternNotFound(Exception):
        def __init__(self, pattern: List[str]):
            self.pattern = pattern

        def __repr__(self):
            print_heading(
                f"Couldn't find images with tags: {listing_details.container_tag_pattern} requested by the operator.",
                colors=[Color.RED],
            )

    tags_map = _get_tags_map_(listing_details)

    if not tags_map:
        _export_helm_chart_(listing_details)
        tags_map = _get_tags_map_(listing_details)
        if not tags_map:
            raise ImageTagPatternNotFound(pattern=listing_details.container_tag_pattern)
    else:
        print(
            f"Images already exist in the path. Continuing without export.{StatusIcons.CHECK}"
        )
    return tags_map


def get_kubernetes_service(listings_details: HelmMarketplaceListingDetails):
    import kubernetes

    kubernetes.config.load_kube_config()
    k8 = kubernetes.client.CoreV1Api()
    return k8.list_namespaced_service(
        namespace=listings_details.namespace,
        label_selector=f"app.kubernetes.io/instance={listings_details.helm_app_name}",
    )


def _get_tags_map_(
    listing_details: HelmMarketplaceListingDetails,
) -> Dict[str, str]:
    images = list_container_images(
        compartment_id=listing_details.compartment_id,
        ocir_image_path=listing_details.ocir_image_path,
    )
    tags_map = {}
    for image in images.items:
        for container_tag_pattern in listing_details.container_tag_pattern:
            if container_tag_pattern in image.display_name:
                tags_map[container_tag_pattern] = image.display_name.split(":")[1]
                break
    return tags_map

import os
import time

import oci
from ads.common.oci_client import OCIClientFactory

from ads.common import auth as authutil
from ads.opctl.backend.marketplace.marketplace_type import HelmMarketplaceListingDetails


def get_oci_signer():
    PROFILE_NAME = "SESSION_PROFILE"
    config = oci.config.from_file(profile_name=PROFILE_NAME)
    if "OCI_RESOURCE_PRINCIPAL_VERSION" not in os.environ:
        token_file = config["security_token_file"]
        token = None
        with open(token_file, "r") as f:
            token = f.read()

        private_key = oci.signer.load_private_key_from_file(config["key_file"])
        signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
        return {"region": config["region"]}, signer
    else:
        signer = oci.auth.signers.get_resource_principals_signer()
        return {}, signer


def set_kubernetes_session_token_env() -> None:
    os.environ["OCI_CLI_AUTH"] = "security_token"
    os.environ["OCI_CLI_PROFILE"] = "SESSION_PROFILE"


def get_marketplace_client():
    return OCIClientFactory(**authutil.default_signer()).marketplace


def export_helm_chart(listing_details: HelmMarketplaceListingDetails):
    export_listing_response = get_marketplace_client().export_listing(
        listing_id=listing_details.listing_id,
        package_version=listing_details.version,
        export_package_details=oci.marketplace.models.ExportPackageDetails(
            compartment_id=listing_details.compartment_id,
            container_repository_path=listing_details.ocir_repo.rstrip("/"),
        ),
    )

    # Get the data from response
    workflow_id = export_listing_response.data.id
    return workflow_id


def get_marketplace_request_status(work_request_id):
    get_work_request_response = get_marketplace_client().get_work_request(
        work_request_id=work_request_id
    )
    return get_work_request_response.data


def wait_for_marketplace_export(work_request_id):
    # Configs can be set in Configuration class directly or using helper utility
    start_time = time.time()
    timeout_seconds = 10 * 60
    sleep_time = 10
    while True:
        response = get_marketplace_request_status(work_request_id)
        if response.status == "SUCCEEDED":
            print(f"Export Successful")
            return 0
        if response.status == "FAILED":
            return -1
        if time.time() - start_time >= timeout_seconds:
            print(
                f"Timed out waiting for work request. Current status: {response.status}"
            )
            break
        print(
            f"Waiting to complete marketplace export workflow. Current status: {response.status}"
        )
        time.sleep(sleep_time)
    return -1


def list_container_images(
    listing_details: HelmMarketplaceListingDetails,
) -> oci.artifacts.models.ContainerImageCollection:
    artifact_client = OCIClientFactory(**authutil.default_signer()).artifacts
    list_container_images_response = artifact_client.list_container_images(
        compartment_id=listing_details.compartment_id,
        compartment_id_in_subtree=True,
        sort_by="TIMECREATED",
        repository_name=listing_details.ocir_repo.rstrip("/").split("/")[-1],
    )
    return list_container_images_response.data

#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ast
import base64
from typing import Optional, List, Dict

import oci
import requests
from typing import TYPE_CHECKING

try:
    from kubernetes.client import (
        V1ServiceStatus,
        V1Service,
        V1LoadBalancerStatus,
        V1LoadBalancerIngress,
    )
except ImportError:
    if TYPE_CHECKING:
        from kubernetes.client import (
            V1ServiceStatus,
            V1Service,
            V1LoadBalancerStatus,
            V1LoadBalancerIngress,
        )

from oci.resource_manager.models import StackSummary, AssociatedResourceSummary

from ads.opctl.operator.lowcode.feature_store_marketplace.models.apigw_config import (
    APIGatewayConfig,
)

from ads.common.oci_client import OCIClientFactory
from ads.opctl.operator.lowcode.feature_store_marketplace.const import (
    LISTING_ID,
    APIGW_STACK_NAME,
    STACK_URL,
    NLB_RULES_ADDRESS,
    NODES_RULES_ADDRESS,
)
from ads import logger
import click
from ads.opctl import logger

from ads.opctl.backend.marketplace.marketplace_utils import (
    Color,
    print_heading,
    print_ticker,
)
from ads.opctl.operator.lowcode.feature_store_marketplace.models.mysql_config import (
    MySqlConfig,
)

from ads.opctl.operator.lowcode.feature_store_marketplace.models.db_config import (
    DBConfig,
)
from ads.common import auth as authutil


def get_db_details() -> DBConfig:
    jdbc_url = "jdbc:mysql://{}/{}?createDatabaseIfNotExist=true"
    mysql_db_config = MySqlConfig()
    print_heading(
        f"MySQL database configuration",
        colors=[Color.BOLD, Color.BLUE],
        prefix_newline_count=2,
    )
    mysql_db_config.username = click.prompt("Username", default="admin")

    mysql_db_config.auth_type = MySqlConfig.MySQLAuthType(
        click.prompt(
            "Is password provided as plain-text or via a Vault secret?\n"
            "(https://docs.oracle.com/en-us/iaas/Content/KeyManagement/Concepts/keyoverview.htm)",
            type=click.Choice(MySqlConfig.MySQLAuthType.values()),
            default=MySqlConfig.MySQLAuthType.BASIC.value,
        )
    )
    if mysql_db_config.auth_type == MySqlConfig.MySQLAuthType.BASIC:
        basic_auth_config = MySqlConfig.BasicConfig()
        basic_auth_config.password = click.prompt(f"Password", hide_input=True)
        mysql_db_config.basic_config = basic_auth_config

    elif mysql_db_config.auth_type == MySqlConfig.MySQLAuthType.VAULT:
        vault_auth_config = MySqlConfig.VaultConfig()
        vault_auth_config.vault_ocid = click.prompt("Vault OCID")
        vault_auth_config.secret_name = click.prompt(
            "Name of the secret having password"
        )
        mysql_db_config.vault_config = vault_auth_config

    mysql_jdbc_ip = click.prompt(
        "IP address using which the database can be access inside the Kubernetes cluster"
        " (example: 10.0.0.1:3306)"
    )
    db_name = click.prompt(
        "Database name (will be auto created if it doesn't already exist)",
        default="FeatureStore",
    )
    mysql_db_config.url = jdbc_url.format(mysql_jdbc_ip, db_name)
    logger.debug(f"MySQL jdbc url generated is: {mysql_db_config.url}")
    db_config = DBConfig()
    db_config.mysql_config = mysql_db_config
    return db_config


def get_latest_listing_version(compartment_id: str) -> str:
    try:
        marketplace_client = OCIClientFactory(
            **authutil.default_signer()
        ).create_client(oci.marketplace.MarketplaceClient)
        listing: oci.marketplace.models.Listing = marketplace_client.get_listing(
            LISTING_ID, compartment_id=compartment_id
        ).data
        return listing.default_package_version
    except Exception as e:
        logger.error(f"Couldn't get marketplace listing.\n{e}")
        raise e


def get_region() -> Optional[str]:
    try:
        return authutil.default_signer()["config"]["region"]
    except Exception as e:
        logger.debug(f"Unable to get region with error: {e}")
        return None


def _create_new_stack(apigw_config: APIGatewayConfig):
    resource_manager_client: oci.resource_manager.ResourceManagerClient = (
        OCIClientFactory(**authutil.default_signer()).create_client(
            oci.resource_manager.ResourceManagerClient
        )
    )
    print("Creating new api gateway stack...")
    response = requests.get(STACK_URL)
    source_details = oci.resource_manager.models.CreateZipUploadConfigSourceDetails()
    source_details.zip_file_base64_encoded = base64.b64encode(response.content).decode()
    stack_details = oci.resource_manager.models.CreateStackDetails()
    stack_details.compartment_id = apigw_config.root_compartment_id
    stack_details.display_name = APIGW_STACK_NAME
    stack_details.config_source = source_details
    stack_details.variables = {
        "nlb_id": "",
        "tenancy_ocid": apigw_config.root_compartment_id,
        "function_img_ocir_url": "",
        "authorized_user_groups": apigw_config.authorized_user_groups,
        "region": apigw_config.region,
    }
    stack: oci.resource_manager.models.Stack = resource_manager_client.create_stack(
        stack_details
    ).data
    print(f"Created stack {stack.display_name} with id {stack.id}")
    return stack.id


def _print_stack_detail(stack: StackSummary):
    print(f"Detected stack :'{stack.display_name}' created on: '{stack.time_created}'")


def detect_or_create_stack(apigw_config: APIGatewayConfig):
    resource_manager_client: oci.resource_manager.ResourceManagerClient = (
        OCIClientFactory(**authutil.default_signer()).create_client(
            oci.resource_manager.ResourceManagerClient
        )
    )
    stacks: List[StackSummary] = resource_manager_client.list_stacks(
        compartment_id=apigw_config.root_compartment_id,
        display_name=APIGW_STACK_NAME,
        lifecycle_state="ACTIVE",
        sort_by="TIMECREATED",
        sort_order="DESC",
    ).data

    if len(stacks) >= 1:
        print(f"Auto-detected feature store stack(s) in tenancy:")
        for stack in stacks:
            _print_stack_detail(stack)
    choices = {"1": "new", "2": "existing"}
    stack_provision_method = click.prompt(
        f"Select stack provisioning method:\n1.Create new stack\n2.Existing stack\n",
        type=click.Choice(list(choices.keys())),
        show_choices=False,
    )
    if choices[stack_provision_method] == "new":
        return _create_new_stack(apigw_config)
    else:
        return click.prompt(
            "Enter the resource manager stack OCID of the stack to use",
            show_choices=False,
        )


def get_admin_group(tenant_id: str) -> Optional[str]:
    identity_client = OCIClientFactory(**authutil.default_signer()).create_client(
        oci.identity.IdentityClient
    )
    groups: List[oci.identity.models.Group] = identity_client.list_groups(
        compartment_id=tenant_id,
        name="Administrators",
        sort_order="ASC",
        sort_by="TIMECREATED",
    ).data
    if len(groups) > 0:
        return groups[0].id
    else:
        return None


def get_api_gw_details(compartment_id: str) -> APIGatewayConfig:
    apigw_config = APIGatewayConfig()
    apigw_config.enabled = click.confirm(
        "Do you want to setup API gateway for feature store to enable secure access over the internet with authN/Z?",
        default=True,
    )
    if not apigw_config.enabled:
        return apigw_config
    if "tenancy" in compartment_id:
        apigw_config.root_compartment_id = compartment_id
    else:
        apigw_config.root_compartment_id = click.prompt(
            "Please enter the tenancy id where the stack will be deployed", type=str
        )
    apigw_config.region = get_region()
    if apigw_config.region is None:
        apigw_config.region = click.prompt(
            "Please enter the region where the stack is being deployed", type=str
        )

    apigw_config.authorized_user_groups = click.prompt(
        "Please enter the user group ids authorized to access feature store (separate each ocid by a comma)",
        type=str,
        default=get_admin_group(apigw_config.root_compartment_id),
    )
    apigw_config.stack_id = detect_or_create_stack(apigw_config)
    return apigw_config


def get_nlb_id_from_service(service: "V1Service", apigw_config: APIGatewayConfig):
    status: "V1ServiceStatus" = service.status
    lb_status: "V1LoadBalancerStatus" = status.load_balancer
    lb_ingress: "V1LoadBalancerIngress" = lb_status.ingress[0]
    resource_client = OCIClientFactory(**authutil.default_signer()).create_client(
        oci.resource_search.ResourceSearchClient
    )
    search_details = oci.resource_search.models.FreeTextSearchDetails()
    search_details.matching_context_type = "NONE"
    search_details.text = lb_ingress.ip
    resources: List[
        oci.resource_search.models.ResourceSummary
    ] = resource_client.search_resources(
        search_details, tenant_id=apigw_config.root_compartment_id
    ).data.items
    private_ips = list(filter(lambda obj: obj.resource_type == "PrivateIp", resources))
    if len(private_ips) != 1:
        return click.prompt(
            f"Please enter OCID of load balancer associated with ip: {lb_ingress.ip}"
        )
    else:
        nlb_private_ip = private_ips[0]
        nlb_client = OCIClientFactory(**authutil.default_signer()).create_client(
            oci.network_load_balancer.NetworkLoadBalancerClient
        )
        nlbs: List[
            oci.network_load_balancer.models.NetworkLoadBalancerSummary
        ] = nlb_client.list_network_load_balancers(
            compartment_id=nlb_private_ip.compartment_id,
            display_name=nlb_private_ip.display_name,
        ).data.items
        if len(nlbs) != 1:
            return click.prompt(
                f"Please enter OCID of load balancer associated with ip: {lb_ingress.ip}"
            )
        else:
            return nlbs[0].id


def update_resource_manager_stack(
    apigw_config: APIGatewayConfig, nlb_id: str, authoriser_image: str
):
    print("Updating resource manager stack")
    resource_manager_client: oci.resource_manager.ResourceManagerClient = (
        OCIClientFactory(**authutil.default_signer()).create_client(
            oci.resource_manager.ResourceManagerClient
        )
    )
    update_stack_details = oci.resource_manager.models.UpdateStackDetails()
    groups = ",".join(apigw_config.authorized_user_groups.split(","))
    update_stack_details.variables = {
        "nlb_id": nlb_id,
        "tenancy_ocid": apigw_config.root_compartment_id,
        "function_img_ocir_url": authoriser_image,
        "authorized_user_groups": f'["{groups}"]',
        "region": apigw_config.region,
    }
    response = requests.get(STACK_URL)
    source_details = oci.resource_manager.models.UpdateZipUploadConfigSourceDetails()
    source_details.zip_file_base64_encoded = base64.b64encode(response.content).decode()
    update_stack_details.config_source = source_details
    resource_manager_client.update_stack(apigw_config.stack_id, update_stack_details)


def apply_stack(stack_id: str):
    print("Applying stack")
    resource_manager_client: oci.resource_manager.ResourceManagerClient = (
        OCIClientFactory(**authutil.default_signer()).create_client(
            oci.resource_manager.ResourceManagerClient
        )
    )
    resource_manager_composite_client = (
        oci.resource_manager.ResourceManagerClientCompositeOperations(
            resource_manager_client
        )
    )
    job_details = oci.resource_manager.models.CreateJobDetails()
    job_details.stack_id = stack_id
    job_operation_details = oci.resource_manager.models.CreateApplyJobOperationDetails()
    job_operation_details.operation = "APPLY"
    job_operation_details.is_provider_upgrade_required = False
    job_operation_details.execution_plan_strategy = "AUTO_APPROVED"
    job_details.operation = "APPLY"
    job_details.job_operation_details = job_operation_details
    job: oci.resource_manager.models.Job = (
        resource_manager_composite_client.create_job_and_wait_for_state(
            create_job_details=job_details,
            wait_for_states=["FAILED", "SUCCEEDED"],
            waiter_kwargs={
                "wait_callback": lambda times_checked, _: print_ticker(
                    "Waiting for stack to apply", iteration=times_checked - 1
                ),
                "max_interval_seconds": 1,
            },
        )
    ).data
    if job.lifecycle_state == "FAILED":
        print(
            f"{Color.RED}{Color.BOLD}Couldn't apply feature store apigw stack. Please check the error logs to debug issues. Re run the operator once the issues are resolved to complete deployment. For more help refer documentation: https://feature-store-accelerated-data-science.readthedocs.io/{Color.END}"
        )
        exit(1)


def prompt_security_rules(stack_id: str):
    if not click.confirm(
        "Do you want to attach required security rules to subnets automatically?",
        default=True,
    ):
        return

    resource_manager_client: oci.resource_manager.ResourceManagerClient = (
        OCIClientFactory(**authutil.default_signer()).create_client(
            oci.resource_manager.ResourceManagerClient
        )
    )
    associated_resources: List[AssociatedResourceSummary] = (
        resource_manager_client.list_stack_associated_resources(stack_id=stack_id)
    ).data.items
    resource_map: Dict[str, AssociatedResourceSummary] = {}
    for resource in associated_resources:
        resource_map[resource.resource_address] = resource
    nlb_subnet_id = ast.literal_eval(
        resource_map[NLB_RULES_ADDRESS].attributes["freeform_tags"]
    )["subnet_id"]
    nodes_subnet_id = ast.literal_eval(
        resource_map[NODES_RULES_ADDRESS].attributes["freeform_tags"]
    )["subnet_id"]
    add_security_list_to_subnet(
        resource_map[NLB_RULES_ADDRESS].resource_id, nlb_subnet_id
    )
    add_security_list_to_subnet(
        resource_map[NODES_RULES_ADDRESS].resource_id, nodes_subnet_id
    )


def add_security_list_to_subnet(sec_list_id: str, subnet_id: str):
    network_client = OCIClientFactory(**authutil.default_signer()).create_client(
        oci.core.VirtualNetworkClient
    )
    subnet: oci.core.models.Subnet = network_client.get_subnet(subnet_id=subnet_id).data
    print(
        f"Adding security list ({sec_list_id}) to subnet: '{subnet.display_name}'({subnet.id})"
    )
    if sec_list_id not in subnet.security_list_ids:
        update_subnet_details = oci.core.models.UpdateSubnetDetails()
        subnet.security_list_ids.append(sec_list_id)
        update_subnet_details.security_list_ids = subnet.security_list_ids
        network_client.update_subnet(
            subnet_id=subnet_id, update_subnet_details=update_subnet_details
        )

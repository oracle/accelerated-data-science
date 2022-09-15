#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import ads
from ads.common import oci_client as oc
import fsspec
import uuid
import os
from ads.jobs import Job
import oci
from collections import defaultdict
import ipaddress
from ads.opctl.distributed.common.abstract_cluster_provider import ClusterProvider
from ads.opctl.diagnostics.requirement_exception import RequirementException
import urllib.parse
import requests

cp = ClusterProvider(mode="MAIN")
auth = cp.get_oci_auth()
cf = oc.OCIClientFactory(**auth)


class PortsChecker:
    def __init__(self, subnet_id=None, auth=None, **kwargs):
        if subnet_id:
            self.subnet_id = subnet_id
        else:
            job_ocid = os.environ["JOB_OCID"]
            jobDef = Job.from_datascience_job(job_ocid)
            self.subnet_id = jobDef.infrastructure.subnet_id
        self.authinfo = auth or ads.common.auth.default_signer()

    def _fetch_security_list(self):
        core_client = oci.core.VirtualNetworkClient(**self.authinfo)
        subnet = core_client.get_subnet(self.subnet_id).data
        cidr = subnet.cidr_block
        security_list_ids = subnet.security_list_ids
        print(f"{cidr=}")
        # print(type(security_list_ids))
        for id in security_list_ids:
            print(f"{id=}")

        port_open_details = defaultdict(set)
        for id in security_list_ids:
            security_list_info = core_client.get_security_list(id).data
            # print(f"Egress: {security_list_info.egress_security_rules}")
            ingress_security_rules = security_list_info.ingress_security_rules
            for rule in ingress_security_rules:
                tcp_options = (
                    rule.tcp_options.destination_port_range if rule.tcp_options else {}
                )
                source = rule.source
                if tcp_options:
                    port_open_details[source].add((tcp_options.min, tcp_options.max))
                if (
                    rule.protocol == "all" or rule.protocol.lower() == "6"
                ) and rule.tcp_options is None:
                    port_open_details[source].add((0,))

        for key in port_open_details:
            port_open_details[key] = sorted(port_open_details[key])
        return port_open_details

    def _flatten_candidate_port_range(self, port_list):
        """
        Ports are opened by providing the CIDR block of the source host. We need to check only within the hosts that intersects the IP of the job instance
        """
        self_ip = cp.find_self_ip(self.authinfo)
        flattened_port_list = []
        for cidr in port_list:
            if ipaddress.ip_address(self_ip) in ipaddress.ip_network(cidr):
                flattened_port_list += port_list[cidr]
        return sorted(flattened_port_list)

    def check_ports(self, ports, *kwargs):
        port_list = self._flatten_candidate_port_range(self._fetch_security_list())
        unsatisfied_ports = []
        print(f"Checking if ports are open", flush=True)
        print(f"Open port information: {port_list}", flush=True)
        if port_list[0] == (0,):
            print(f"Subnet has all ports open.")
            return True
        for port in ports:
            print(f"Checking if port {port} is open", flush=True)
            if str(port).lower() == "all":
                if port_list[0] != (0,):
                    print(
                        f"Requires ALL ports for this subnet to be open. Open ports are: {'None' if len(port_list) == 0 else port_list}"
                    )
                    unsatisfied_ports.append(port)
                    break
                else:
                    return True
            elif port in port_list:
                print(f"found port in open port list: {port_list}", flush=True)
                continue
            else:
                if len(port.split("-")) == 1:
                    found = False
                    for port_range in port_list:
                        if int(port) >= port_range[0] and int(port) <= port_range[1]:
                            print(
                                f"found port: {port} in range {port_range}", flush=True
                            )
                            found = True
                            break
                    if not found:
                        print(
                            f"Not found port: {port} in {port_list}!!! Check security list",
                            flush=True,
                        )
                        unsatisfied_ports.append(port)
                else:
                    minport = int(port.split("-")[0])
                    maxport = int(port.split("-")[1])
                    found = False
                    curr_minport = minport
                    # It might be better to normalize the range in port_list
                    # to remove any possible overlaps between the ranges. That
                    # will make search much easier than below.
                    for port_range in port_list:
                        if curr_minport >= port_range[1]:
                            continue

                        if curr_minport >= port_range[0]:
                            if maxport <= port_range[1]:
                                print(f"found port open in range {port_list}")
                                found = True
                                break
                            else:
                                curr_minport = (
                                    port_range[1] + 1
                                )  # Can fail if the next range starts at maxport or less than maxport
                    if not found:
                        print(
                            f"Not found port: {port} in {port_list}!!! Check security list",
                            flush=True,
                        )
                        unsatisfied_ports.append(port)
        if len(unsatisfied_ports) == 0:
            return True
        else:
            raise RequirementException(
                f"Following ports are not opened: {unsatisfied_ports}. Update security list corresponding to the subnet: {self.subnet_id}",
            )


class Default:
    def __init__(self, auth, *kwargs):
        self.auth = auth

    def job_info_access_policy(self, **kwargs):
        print(f"Checking Job info access")
        try:
            job_ocid = os.environ["JOB_OCID"]
            Job.from_datascience_job(job_ocid)
            print(f"Passed Job info access")
        except oci.exceptions.ServiceError as se:
            print(se)
            raise RequirementException(
                f"Could not access Job Definition details. To fix this: \n \
                Allow dynamic-group <distributed_training_job_runs> to use data-science-family in compartment <your_compartment_name>"
            )

    def job_vcn_info_access_policy(self, **kwargs):
        print(f"Checking VCN info access")

        job_ocid = os.environ["JOB_OCID"]
        jobDef = Job.from_datascience_job(job_ocid)

        subnet_id = jobDef.infrastructure.subnet_id
        client = cf.create_client(oci.core.VirtualNetworkClient)
        try:
            client.get_subnet(subnet_id).data
            print(f"Passed VCN info access")
        except oci.exceptions.ServiceError as se:
            print(se)
            raise RequirementException(
                f"Could not access Job Definition details. To fix this: \n \
                Allow dynamic-group <distributed_training_job_runs> to use virtual-network-family in compartment <your_compartment_name>"
            )

    def work_dir_access(self):
        test_file_name = f"{os.environ['OCI__WORK_DIR']}/{str(uuid.uuid4())}"
        bucket_name = urllib.parse.urlparse(test_file_name).username
        print(f"Checking Object storage access to {os.environ['OCI__WORK_DIR']}")
        try:
            with fsspec.open(test_file_name, "w", **self.auth) as tf:
                tf.write("sample")
            print(f"Passed object storage access")
        except oci.exceptions.ServiceError as se:
            print(se)
            raise RequirementException(
                "Could not access Job Definition details. To fix this: \n \
                Allow dynamic-group <distributed_training_job_runs> to manage objects in compartment your_compartment_name where all {target.bucket.name="
                + bucket_name
                + "}"
            )

    def check_internet(self, **kwargs):
        print(f"Checking for internet")
        try:
            res = requests.get("https://www.oracle.com", timeout=5)
        except requests.exceptions.ConnectionError as ce:
            print(ce)
            raise RequirementException(
                f"Could not access internet. To fix this: \n \
                Add NAT gateway to your subnet. If you are behind proxy, add https_proxy and http_proxy to the environment variables"
            )

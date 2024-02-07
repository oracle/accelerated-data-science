#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

LISTING_ID = "ocid1.mktpublisting.oc1.iad.amaaaaaabiudgxya26lzh2dsyvg7cfzgllvdl6xo5phz4mnsoktxeutecrvq"
APIGW_STACK_NAME = "fs-apigw-stack"
STACK_URL = "https://raw.githubusercontent.com/harsh97/oci-data-science-ai-samples/feature-store/feature_store/fs_apigw_terraform.zip"
NLB_RULES_ADDRESS = (
    "module.feature_store_gw_subnet.oci_core_security_list.nlb_security_rules"
)
NODES_RULES_ADDRESS = (
    "module.feature_store_gw_subnet.oci_core_security_list.node_security_rules"
)

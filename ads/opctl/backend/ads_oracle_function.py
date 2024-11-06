#!/usr/bin/env python

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Dict

from ads.common.auth import AuthContext, create_signer
from ads.common.oci_client import OCIClientFactory
from ads.function.service.oci_function import OCIFunction
from ads.opctl.backend.base import Backend


class OracleFunctionBackend(Backend):
    def __init__(self, config: Dict) -> None:
        """
        Initialize a OracleFunction object given config dictionary.

        Parameters
        ----------
        config: dict
            dictionary of configurations
        """
        self.config = config
        self.oci_auth = create_signer(
            config["execution"].get("auth"),
            config["execution"].get("oci_config", None),
            config["execution"].get("oci_profile", None),
        )
        self.auth_type = config["execution"].get("auth")
        self.profile = config["execution"].get("oci_profile", None)
        self.client = OCIClientFactory(**self.oci_auth).oracle_function_management

    def watch(self) -> None:
        """
        Watch Oracle Function.
        """
        run_id = self.config["execution"]["run_id"]
        interval = self.config["execution"].get("interval")
        with AuthContext(auth=self.auth_type, profile=self.profile):
            function = OCIFunction.from_ocid(run_id)
            function.watch(interval=interval)

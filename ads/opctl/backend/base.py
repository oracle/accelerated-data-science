#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import Dict

from ads.common.auth import create_signer
from ads.common.oci_client import OCIClientFactory


class Backend:
    """Interface for backend"""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.auth_type = config["execution"].get("auth")
        self.profile = config["execution"].get("oci_profile", None)
        self.oci_config = config["execution"].get("oci_config", None)

        self.oci_auth = create_signer(
            self.auth_type,
            self.oci_config,
            self.profile,
        )
        self.client = OCIClientFactory(**self.oci_auth).data_science

    @abstractmethod
    def run(self) -> Dict:
        """
        Initiate a run.

        Returns
        -------
        None
        """

    def delete(self) -> None:
        """
        Delete a remote run.

        Returns
        -------
        None
        """

    def watch(self) -> None:
        """
        Stream logs from a remote run.

        Returns
        -------
        None
        """

    def cancel(self) -> None:
        """
        Cancel a remote run.

        Returns
        -------
        None
        """

    def apply(self) -> None:
        """
        Initiate Data Science service from YAML.

        Returns
        -------
        None
        """

    def activate(self) -> None:
        """
        Activate a remote service.

        Returns
        -------
        None
        """
        raise NotImplementedError("`activate` has not been implemented yet.")

    def deactivate(self) -> None:
        """
        Deactivate a remote service.

        Returns
        -------
        None
        """
        raise NotImplementedError("`deactivate` has not been implemented yet.")

    def run_diagnostics(self):
        """
        Implement Diagnostics check appropriate for the backend
        """

    def predict(self) -> None:
        """
        Run model predict.

        Returns
        -------
        None
        """
        raise NotImplementedError("`predict` has not been implemented yet.")

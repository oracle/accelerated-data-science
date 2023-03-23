#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import Dict

from ads.common.auth import get_signer


class Backend:
    """Interface for backend"""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.oci_auth = get_signer(
            config["execution"].get("oci_config", None),
            config["execution"].get("oci_profile", None),
        )
        self.profile = config["execution"].get("oci_profile", None)

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

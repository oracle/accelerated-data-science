#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import Dict, Union


class UnsupportedRuntime(Exception):
    def __init__(self, runtime_type: str):
        super().__init__(
            f"The provided runtime: `{runtime_type}` "
            "is not supported by given resource."
        )


class Backend:
    """Interface for backend"""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.auth_type = config["execution"].get("auth", "api_key")
        self.profile = config["execution"].get("oci_profile", None)
        self.oci_config = config["execution"].get("oci_config", None)

    @abstractmethod
    def run(self) -> Dict:
        """
        Initiate a run.

        Returns
        -------
        Dict
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

    def apply(self) -> Dict:
        """
        Initiate Data Science service from YAML.

        Returns
        -------
        Dict
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

    def init(
        self, uri: Union[str, None] = None, overwrite: bool = False, **kwargs: Dict
    ) -> Union[str, None]:
        """Generates a YAML specification for the resource.

        Parameters
        ----------
        overwrite: (bool, optional). Defaults to False.
            Overwrites the result specification YAML if exists.
        uri: (str, optional)
            The filename to save the resulting specification template YAML.
        **kwargs: Dict
            The optional arguments.

            runtime_type: str
                The resource runtime type.

        Returns
        -------
        Union[str, None]
            The YAML specification for the given resource if `uri` was not provided.
            `None` otherwise.
        """
        raise NotImplementedError(
            "The `init` has not been implemented yet for the given resource."
        )

    def predict(self) -> None:
        """
        Run model predict.

        Returns
        -------
        None
        """
        raise NotImplementedError("`predict` has not been implemented yet.")


class RuntimeFactory:
    """Base factory for runtime."""

    _MAP = {}

    @classmethod
    def get_runtime(cls, key: str, *args, **kwargs):
        if key not in cls._MAP:
            raise UnsupportedRuntime(key)
        return cls._MAP[key](*args, **kwargs)

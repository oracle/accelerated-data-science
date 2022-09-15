#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import Dict


class Backend:
    """Interface for backend"""

    @abstractmethod
    def run(self) -> Dict:
        """
        Initiate a run.

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def delete(self) -> None:
        """
        Delete a remote run.

        Returns
        -------

        """
        pass  # pragma: no cover

    def watch(self) -> None:
        """
        Stream logs from a remote run.

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def cancel(self) -> None:
        """
        Cancel a remote run.

        Returns
        -------
        None
        """
        pass  # pragma: no cover

    def run_diagnostics(self):
        """
        Implement Diagnostics check appropriate for the backend
        """
        pass

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common import utils as common_utils
from ads.common.decorator.utils import class_or_instance_method
from ads.jobs.builders.base import Builder
from ads.jobs.builders.runtimes.base import Runtime
import logging

logger = logging.getLogger(__name__)


class Infrastructure(Builder):
    """Base class for job infrastructure"""

    @property
    def kind(self) -> str:
        """Kind of the object to be stored in YAML. All runtimes will have "infrastructure" as kind.
        Subclass will have different types.
        """
        return "infrastructure"

    @property
    def name(self) -> str:
        """The name of the job from the infrastructure."""
        raise NotImplementedError()

    def _assert_runtime_compatible(self, runtime: Runtime) -> None:
        """
        Check if a runtime is compatible with an engine. Raise an error if not.

        Parameters
        ----------
        runtime: `Runtime`
            a runtime object

        Returns
        -------
        None
        """

    def create(self, runtime: Runtime, **kwargs):
        """
        Create/deploy a job on the infrastructure.

        Parameters
        ----------
        runtime: `Runtime`
            a runtime object
        kwargs: dict
            additional arguments

        """
        raise NotImplementedError()

    def run(
        self,
        name: str = None,
        args: str = None,
        env_var: dict = None,
        freeform_tags: dict = None,
        defined_tags: dict = None,
        wait: bool = False,
    ):
        """Runs a job on the infrastructure.

        Parameters
        ----------
        name : str, optional
            The name of the job run, by default None
        args : str, optional
            Command line arguments for the job run, by default None.
        env_var : dict, optional
            Environment variable for the job run, by default None.
        freeform_tags : dict, optional
            Freeform tags for the job run, by default None.
        defined_tags : dict, optional
            Defined tags for the job run, by default None.
        wait : bool, optional
            Indicate if this method should wait for the run to finish before it returns, by default False.
        """
        raise NotImplementedError()

    def delete(self):
        """Deletes a job from the infrastructure."""
        raise NotImplementedError()

    def update(self, runtime: Runtime):
        """Updates a job.

        Parameters
        ----------
        runtime
            a runtime object
        """
        raise NotImplementedError()

    @class_or_instance_method
    def list_jobs(cls, **kwargs) -> list:
        """
        List jobs from the infrastructure.

        Parameters
        ----------
        kwargs: keyword arguments for filtering the results

        Returns
        -------
        list
            list of infrastructure objects, each representing a job from the infrastructure.
        """
        raise NotImplementedError()


class RunInstance:
    _DETAILS_LINK = ""

    def create(self):
        """Create a RunInstance Object."""
        raise NotImplementedError()

    @property
    def status(self):
        """Return status of the run."""
        raise NotImplementedError()

    def watch(self):
        """Show logs of a run."""
        raise NotImplementedError()

    def delete(self):
        """Delete a run."""
        raise NotImplementedError()

    def cancel(self):
        """Cancel a run."""
        raise NotImplementedError()

    @property
    def run_details_link(self) -> str:
        """
        Link to run details page in OCI console

        Returns
        -------
        str
            The link to the details page in OCI console.
        """
        if not self._DETAILS_LINK:
            return ""

        try:
            return self._DETAILS_LINK.format(
                region=common_utils.extract_region(), id=self.id
            )
        except Exception as ex:
            print(str(ex))
            logger.info(
                "Error occurred in attempt to extract the link to the resource. "
                f"Details: {ex}"
            )
        return ""

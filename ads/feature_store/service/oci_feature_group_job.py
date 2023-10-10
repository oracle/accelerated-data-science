#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import logging
import time

import feature_store_client.feature_store as fs
from feature_store_client.feature_store.models import (
    CreateFeatureGroupJobDetails,
    CompleteFeatureGroupJobDetails,
    FeatureGroupJob,
    DatasetJob,
)

from ads.feature_store.mixin.oci_feature_store import OCIFeatureStoreMixin

logger = logging.getLogger(__name__)

SLEEP_INTERVAL = 3


class OCIFeatureGroupJob(OCIFeatureStoreMixin, FeatureGroupJob):
    """Represents an OCI Data Science FeatureGroupJob.
    This class contains all attributes of the `oci.data_science.models.FeatureGroupJob`.
    The main purpose of this class is to link the `oci.data_science.models.FeatureGroupJob`
    and the related client methods.
    Linking the `FeatureGroupJob` (payload) to Create methods.

    The `OCIFeatureGroupJob` can be initialized by unpacking the properties stored in a dictionary:

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_definition_job_display_name>",
            "description": "<feature_definition_job_description>",
            "feature_store_id":"<feature_store_id>",
        }
        feature_definition_job = OCIFeatureGroupJob(**properties)

    The properties can also be OCI REST API payload, in which the keys are in camel format.

    .. code-block:: python

        properties = {
            "compartment_id": "<compartment_ocid>",
            "display_name": "<feature_definition_job_display_name>",
            "description": "<feature_definition_job_description>",
            "feature_store_id":"<feature_store_id>",
        }
        feature_definition_job = OCIFeatureGroupJob(**properties)

    Methods
    -------
    create(self) -> "OCIFeatureGroupJob"
        Creates feature definition job.
    from_id(cls, ocid: str) -> "OCIFeatureGroupJob":
        Gets existing feature definition run by Id.
    Examples
    --------
    >>> oci_feature_definition_job = OCIFeatureGroupJob.from_id("<feature_definition_job_id>")
    >>> oci_feature_definition_job.create()
    """

    TERMINAL_STATES = [
        DatasetJob.LIFECYCLE_STATE_SUCCEEDED,
        DatasetJob.LIFECYCLE_STATE_FAILED,
    ]

    def __init__(self, **kwargs) -> None:
        """Initialize a OCIFeatureGroupJob object.

        Parameters
        ----------
        kwargs:
            Same as kwargs in feature_store.models.OCIFeatureGroupJob.
            Keyword arguments are passed into OCI feature group job model to initialize the properties.

        """

        super().__init__(**kwargs)

    # Overriding default behavior
    @classmethod
    def init_client(cls, **kwargs) -> fs.feature_store_client.FeatureStoreClient:
        client = super().init_client(**kwargs)

        # Define the list entities callable to list the resources
        cls.OCI_LIST_METHOD = client.list_feature_group_jobs

        return client

    def create(self) -> "OCIFeatureGroupJob":
        """Creates feature definition resource.

        Returns
        -------
        OCIFeatureGroupJob
            The `OCIFeatureGroupJob` instance (self), which allows chaining additional method.
        """
        if not self.compartment_id:
            raise ValueError("The `compartment_id` must be specified.")
        feature_definition_job_details = self.to_oci_model(CreateFeatureGroupJobDetails)
        return self.update_from_oci_model(
            self.client.create_feature_group_job(feature_definition_job_details).data
        )

    @property
    def status(self) -> str:
        """Lifecycle status

        Returns
        -------
        str
            Status in a string.
        """
        return self.lifecycle_state

    def _job_run_status_text(self) -> str:
        details = f", {self.lifecycle_details}" if self.lifecycle_details else ""
        return f"Job Run {self.lifecycle_state}" + details

    def watch(self, interval: float = SLEEP_INTERVAL) -> "OCIFeatureGroupJob":
        """Watches the feature definition job run until it finishes.
        Before the job start running, this method will output the feature definition job run status.
        Once the job start running, the logs will be streamed until the job is success, failed or cancelled.

        Parameters
        ----------
        interval : int
            Time interval in seconds between each request to update the logs.
            Defaults to 3 (seconds).
        """

        def stop_condition():
            """Stops the log once the job is in a terminal state."""
            self.sync()
            if self.lifecycle_state not in self.TERMINAL_STATES:
                return False
            # Stop if time_finished is not available.
            if not self.time_finished:
                return True
            # Stop only if time_finished is over 1 minute ago.
            # This is for the time delay between feature definition job run stopped and the logs appear in oci logging.
            if (
                datetime.datetime.now(self.time_finished.tzinfo)
                - datetime.timedelta(minutes=1)
                > self.time_finished
            ):
                return True
            return False

        logger.info(f"Feature definition job OCID: {self.job.id}")
        logger.info(f"Feature definition job run OCID: {self.id}")

        status = ""
        while not stop_condition():
            status = self._check_and_print_status(status)
            # Break and stream logs if job has log ID and started.
            # Otherwise, keep watching the status until job terminates.
            if self.time_started and self.log_id:
                break
            time.sleep(interval)
        self._check_and_print_status(status)
        return self

    def _check_and_print_status(self, prev_status) -> str:
        status = self._job_run_status_text()
        if status != prev_status:
            if self.lifecycle_state in self.TERMINAL_STATES and self.time_finished:
                timestamp = self.time_finished
            else:
                timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        return status

    def complete_feature_group_job(self) -> "OCIFeatureGroupJob":
        """Completes a feature group job.

        Returns
        -------
        OCIFeatureGroupJob
            The `OCIFeatureGroupJob` instance (self).
        """
        return self.update_from_oci_model(
            self.client.complete_feature_group_job(
                self.id, self.to_oci_model(CompleteFeatureGroupJobDetails)
            ).data
        )

    @classmethod
    def from_id(cls, id: str) -> "OCIFeatureGroupJob":
        """Gets feature definition resource  by id.

        Parameters
        ----------
        id: str
            The id of the feature definition resource.

        Returns
        -------
        OCIFeatureGroupJob
            An instance of `OCIFeatureGroupJob`.
        """
        if not id:
            raise ValueError("FeatureGroupJob id not provided.")
        return super().from_ocid(id)

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import json
import logging
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Any, Union

import pandas

from ads.common import utils
from ads.feature_store.common.enums import BatchIngestionMode, StreamingIngestionMode
from ads.feature_store.feature_option_details import FeatureOptionDetails
from ads.feature_store.service.oci_feature_group_job import OCIFeatureGroupJob
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class JobConfigurationType(Enum):
    SPARK_BATCH_AUTOMATIC = "SPARK_BATCH_AUTOMATIC"
    SPARK_BATCH_MANUAL = "SPARK_BATCH_MANUAL"


class FeatureGroupJob(Builder):
    """Represents an FeatureGroupJob Resource.

    Methods
    --------

    create(self, **kwargs) -> "FeatureGroupJob"
        Creates feature_group_run resource.
    from_id(cls, id: str) -> "FeatureGroupJob"
        Gets an existing feature_group_run resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["FeatureGroupJob"]
        Lists feature_group_run resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists feature_group_run resources as a pandas dataframe.
    with_compartment_id(self, compartment_id: str) -> "FeatureGroupJob"
        Sets the compartment ID.
    with_feature_group_id(self, feature_group_id: str) -> "FeatureGroupJob"
        Sets the feature group ID.
    with_ingestion_mode(self, ingestion_mode: IngestionMode) -> "FeatureGroupJob"
        Sets the ingestion mode.
    Examples
    --------
    >>> from ads.feature_store import feature_group_job
    >>> import oci
    >>> import os
    >>> feature_group_run = feature_group_run.FeatureGroupJob()
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_feature_group_id("<feature_group_id>")
    >>>     .with_ingestion_mode(IngestionMode.OVERWRITE)
    >>> feature_group_run.create()
    """

    _PREFIX = "feature_group_run_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_FEATURE_GROUP_ID = "featureGroupId"
    CONST_INGESTION_MODE = "ingestionMode"
    CONST_JOB_CONFIGURATION_DETAILS = "jobConfigurationDetails"
    CONST_JOB_CONFIGURATION_TYPE = "jobConfigurationType"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_FEATURE_OPTION_DETAILS = "featureOptionsDetails"
    CONST_DEFINED_TAG = "definedTags"
    CONST_TIME_FROM = "timeFrom"
    CONST_TIME_TO = "timeTo"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_JOB_OUTPUT_DETAILS = "jobOutputDetails"
    CONST_DATA_FLOW_EXECUTION_OUTPUT = "dataFlowBatchExecutionOutput"
    CONST_VALIDATION_OUTPUT = "validation_output"
    CONST_DATA_FLOW_READ_WRITE_DETAIL = "data_flow_read_write_detail"
    CONST_DATA_READ_IN_BYTES = "data_read_in_bytes"
    CONST_DATA_WRITTEN_BYTES = "data_written_in_bytes"
    CONST_FEATURE_STATISTICS = "featureStatistics"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_FEATURE_GROUP_ID: "feature_group_id",
        CONST_INGESTION_MODE: "ingestion_mode",
        CONST_JOB_CONFIGURATION_DETAILS: "job_configuration_details",
        CONST_JOB_CONFIGURATION_TYPE: "job_configuration_type",
        CONST_FEATURE_OPTION_DETAILS: "feature_option_details",
        CONST_TIME_FROM: "time_from",
        CONST_TIME_TO: "time_to",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_LIFECYCLE_STATE: "lifecycle_state",
        CONST_JOB_OUTPUT_DETAILS: "job_output_details",
        CONST_DATA_FLOW_EXECUTION_OUTPUT: "data_flow_batch_execution_output",
        CONST_FEATURE_STATISTICS: "feature_statistics",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes FeatureGroupJob Resource.

        Parameters
        ----------
        spec: (Dict, optional). Defaults to None.
            Object specification.

        kwargs: Dict
            Specification as keyword arguments.
            If 'spec' contains the same key as the one in kwargs,
            the value from kwargs will be used.
        """
        super().__init__(spec=spec, **deepcopy(kwargs))
        # Specify oci FeatureGroupJob instance

        self.oci_fs_feature_group_run = self._to_oci_fs_feature_group_run(**kwargs)

    def _to_oci_fs_feature_group_run(self, **kwargs):
        """Creates an `OCIFeatureGroupJob` instance from the  `FeatureGroupJob`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureGroupJob` accepts.

        Returns
        -------
        OCIFeatureGroupJob
            The instance of the OCIFeatureGroupJob.
        """
        fs_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCIFeatureGroupJob(**fs_spec)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "feature_group_job"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "FeatureGroupJob":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    def get_validation_output_df(self) -> "pandas.DataFrame":
        """
        This method retrieves the validation output as a Pandas DataFrame.

        Returns:
        pandas.DataFrame -- The validation output data in DataFrame format.
        """

        # retrieve the validation output JSON from data_flow_batch_execution_output
        validation_output_json = json.loads(
            self.job_output_details.get("validationOutput")
        )

        # Convert Python object to Pandas DataFrame
        validation_output_df = pandas.json_normalize(validation_output_json).transpose()

        # return the validation output DataFrame
        return validation_output_df

    @property
    def time_from(self) -> str:
        return self.get_spec(self.CONST_TIME_FROM)

    def with_time_from(self, time_from: str) -> "FeatureGroupJob":
        """Sets the time_from.

        Parameters
        ----------
        time_from: str
            The time_from.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_TIME_FROM, time_from)

    @property
    def time_to(self) -> str:
        return self.get_spec(self.CONST_TIME_TO)

    def with_time_to(self, time_to: str) -> "FeatureGroupJob":
        """Sets the time_to.

        Parameters
        ----------
        time_to: str
            The time_to.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_TIME_TO, time_to)

    @property
    def feature_group_id(self) -> str:
        return self.get_spec(self.CONST_FEATURE_GROUP_ID)

    @feature_group_id.setter
    def feature_group_id(self, value: str):
        self.with_feature_group_id(value)

    def with_feature_group_id(self, feature_group_id: str) -> "FeatureGroupJob":
        """Sets the feature_group_id.

        Parameters
        ----------
        feature_group_id: str
            The feature group id.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_GROUP_ID, feature_group_id)

    def with_id(self, id: str) -> "FeatureGroupJob":
        return self.set_spec(self.CONST_ID, id)

    @property
    def id(self) -> str:
        return self.get_spec(self.CONST_ID)

    @property
    def job_configuration_details(self) -> str:
        return self.get_spec(self.CONST_JOB_CONFIGURATION_DETAILS)

    @job_configuration_details.setter
    def job_configuration_details(
        self, job_configuration_type: JobConfigurationType, **kwargs: Dict[str, Any]
    ):
        self.with_job_configuration_details(job_configuration_type, **kwargs)

    def with_job_configuration_details(
        self, job_configuration_type: JobConfigurationType, **kwargs: Dict[str, Any]
    ) -> "FeatureGroupJob":
        """Sets the job configuration details.

        Parameters
        ----------
        job_configuration_type: JobConfigurationType
            The job_configuration_type of job
        kwargs: Dict[str, Any]
            Additional key value arguments

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(
            self.CONST_JOB_CONFIGURATION_DETAILS,
            {
                self.CONST_JOB_CONFIGURATION_TYPE: job_configuration_type.value,
                **kwargs,
            },
        )

    @property
    def job_output_details(self) -> Dict:
        return self.get_spec(self.CONST_JOB_OUTPUT_DETAILS)

    def with_job_output_details(self, job_output_details: Dict) -> "FeatureGroupJob":
        """Sets the job output details.

        Parameters
        ----------
        job_output_details: Dict
            The job output details which contains error_details, validation_output and commit id.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_JOB_OUTPUT_DETAILS, job_output_details)

    @property
    def feature_option_details(self) -> Dict:
        return self.get_spec(self.CONST_FEATURE_OPTION_DETAILS)

    def with_feature_option_details(
        self, feature_option_details: FeatureOptionDetails
    ) -> "FeatureGroupJob":
        """Sets the feature_option_details.

        Parameters
        ----------
        feature_option_details: FeatureOptionDetails

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(
            self.CONST_FEATURE_OPTION_DETAILS, feature_option_details.to_dict()
        )

    @property
    def ingestion_mode(self) -> str:
        return self.get_spec(self.CONST_INGESTION_MODE)

    @ingestion_mode.setter
    def ingestion_mode(
        self, ingestion_mode: Union[BatchIngestionMode, StreamingIngestionMode]
    ) -> "FeatureGroupJob":
        return self.with_ingestion_mode(ingestion_mode)

    def with_ingestion_mode(
        self, ingestion_mode: Union[BatchIngestionMode, StreamingIngestionMode]
    ) -> "FeatureGroupJob":
        """Sets the mode of the dataset ingestion mode.

        Parameters
        ----------
        ingestion_mode

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_INGESTION_MODE, ingestion_mode.value)

    def with_lifecycle_state(self, lifecycle_state: str) -> "FeatureGroupJob":
        """Sets the lifecycle_state.

        Parameters
        ----------
        lifecycle_state: str
            The lifecycle_state.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_LIFECYCLE_STATE, lifecycle_state)

    @property
    def feature_statistics(self) -> str:
        return self.get_spec(self.CONST_FEATURE_STATISTICS)

    def with_feature_statistics(self, feature_statistics: str) -> "FeatureGroupJob":
        """Sets the computed statistics.

        Parameters
        ----------
        feature_statistics: str
            Computed Feature Statistics

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(self.CONST_FEATURE_STATISTICS, feature_statistics)

    @classmethod
    def from_id(cls, id: str) -> "FeatureGroupJob":
        """Gets an existing feature_group_run resource by Id.

        Parameters
        ----------
        id: str
            The feature_group_run id.

        Returns
        -------
        FeatureStore
            An instance of FeatureGroupJob resource.
        """
        return cls()._update_from_oci_fs_model(OCIFeatureGroupJob.from_id(id))

    def create(self, **kwargs) -> "FeatureGroupJob":
        """Creates feature_group_run  resource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureGroupJob` accepts.

        Returns
        -------
        FeatureStore
            The FeatureGroupJob instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        if not self.compartment_id:
            raise ValueError("Compartment id must be provided.")

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a feature_group_run resource with payload {payload}")

        # Create entity
        logger.info("Saving entity.")
        self.oci_fs_feature_group_run = self._to_oci_fs_feature_group_run(
            **kwargs
        ).create()
        self.with_id(self.oci_fs_feature_group_run.id)
        return self

    def update(self, **kwargs) -> "FeatureGroupJob":
        """Updates FeatureGroupJob in the feature store.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.FeatureGroupJob` accepts.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self).
        """

        if not self.id:
            raise ValueError(
                "FeatureGroupJob needs to be saved to the feature store before it can be updated."
            )

        self.oci_fs_feature_group_run = self._to_oci_fs_feature_group_run(
            **kwargs
        ).update()
        return self

    def _mark_job_complete(self, job_output_details) -> "FeatureGroupJob":
        """
        Completes the current feature group job and returns a new `FeatureGroupJob` object that represents it.

        :param job_output_details: The output details of the completed job.
        :return: A new `FeatureGroupJob` object representing the completed job.
        :raises ValueError: If the current `FeatureGroupJob` object has not been saved to the feature store yet.
        """
        # Check if the current FeatureGroupJob has been saved to the feature store
        if not self.id:
            raise ValueError(
                "FeatureGroupJob needs to be saved to the feature store before it can be marked as completed."
            )

        # Update the job's output details with the given ones
        self.with_job_output_details(job_output_details)

        # Complete the job in the OCI feature store and return a new FeatureGroupJob object
        self.oci_fs_feature_group_run = (
            self._to_oci_fs_feature_group_run().complete_feature_group_job()
        )
        return self

    def _update_from_oci_fs_model(
        self, oci_fs_feature_group_run: OCIFeatureGroupJob
    ) -> "FeatureGroupJob":
        """Update the properties from an OCIFeatureGroupJob object.

        Parameters
        ----------
        oci_fs_feature_group_run: OCIFeatureGroupJob
            An instance of OCIFeatureGroupJob.

        Returns
        -------
        FeatureGroupJob
            The FeatureGroupJob instance (self).
        """

        # Update the main properties
        self.oci_fs_feature_group_run = oci_fs_feature_group_run
        feature_group_run_details = oci_fs_feature_group_run.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in feature_group_run_details:
                self.set_spec(infra_attr, feature_group_run_details[infra_attr])

        return self

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists feature_group_run resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the feature_group_run resources in a pandas dataframe format.
        """
        records = []
        for oci_fs_feature_group_run in OCIFeatureGroupJob.list_resource(
            compartment_id, **kwargs
        ):
            records.append(
                {
                    "id": oci_fs_feature_group_run.id,
                    "display_name": oci_fs_feature_group_run.display_name,
                    "time_created": oci_fs_feature_group_run.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_fs_feature_group_run.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_fs_feature_group_run.lifecycle_state,
                    "created_by": f"...{oci_fs_feature_group_run.created_by[-6:]}",
                    "compartment_id": f"...{oci_fs_feature_group_run.compartment_id[-6:]}",
                    "feature_group_id": oci_fs_feature_group_run.feature_group_id,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["FeatureGroupJob"]:
        """Lists FeatureGroupJob Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering FeatureGroupJob.

        Returns
        -------
        List[FeatureGroupJob]
            The list of the FeatureGroupJob Resources.
        """
        return [
            cls()._update_from_oci_fs_model(oci_fs_feature_group_run)
            for oci_fs_feature_group_run in OCIFeatureGroupJob.list_resource(
                compartment_id, **kwargs
            )
        ]

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes feature_group_run  to a dictionary.

        Returns
        -------
        dict
            The feature_group_run resource serialized as a dictionary.
        """

        spec = deepcopy(self._spec)
        for key, value in spec.items():
            if hasattr(value, "to_dict"):
                value = value.to_dict()
            spec[key] = value

        return {
            "kind": self.kind,
            "type": self.type,
            "spec": utils.batch_convert_case(spec, "camel"),
        }

    def __repr__(self) -> str:
        """Displays the object as YAML."""
        return self.to_yaml()

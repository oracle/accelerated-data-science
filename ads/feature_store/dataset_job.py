#!/usr/bin/env python

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import logging
from copy import deepcopy
from typing import Dict, List, Any, Union

import pandas

from ads.common import utils
from ads.feature_store.common.enums import (
    JobConfigurationType,
    BatchIngestionMode,
    StreamingIngestionMode,
)
from ads.feature_store.feature_option_details import FeatureOptionDetails
from ads.feature_store.service.oci_dataset_job import OCIDatasetJob
from ads.jobs.builders.base import Builder

logger = logging.getLogger(__name__)


class DatasetJob(Builder):
    """Represents an DatasetJob Resource.

    Methods
    --------

    create(self, **kwargs) -> "DatasetJob"
        Creates dataset_run resource.
    from_id(cls, id: str) -> "DatasetJob"
        Gets an existing dataset_run resource by id.
    list(cls, compartment_id: str = None, **kwargs) -> List["DatasetJob"]
        Lists dataset_run resources in a given compartment.
    list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame"
        Lists dataset_run resources as a pandas dataframe.
    with_description(self, description: str) -> "DatasetJob"
        Sets the description.
    with_compartment_id(self, compartment_id: str) -> "DatasetJob"
        Sets the compartment ID.
    with_dataset_id(self, dataset_id: str) -> "DatasetJob"
        Sets the dataset ID.
    with_display_name(self, name: str) -> "DatasetJob"
        Sets the name.
    with_ingestion_mode(self, ingestion_mode: IngestionMode) -> "DatasetJob"
        Sets the ingestion mode.
    Examples
    --------
    >>> from ads.feature_store import dataset_job
    >>> import oci
    >>> import os
    >>> dataset_run = dataset_run.DatasetJob()
    >>>     .with_compartment_id(os.environ["PROJECT_COMPARTMENT_OCID"])
    >>>     .with_dataset_id("dataset_id")
    >>>     .with_ingestion_mode(BatchIngestionMode.SQL)
    >>> dataset_run.create()
    """

    _PREFIX = "dataset_run_resource"

    CONST_ID = "id"
    CONST_COMPARTMENT_ID = "compartmentId"
    CONST_DATASET_ID = "datasetId"
    CONST_INGESTION_MODE = "ingestionMode"
    CONST_JOB_CONFIGURATION_DETAILS = "jobConfigurationDetails"
    CONST_JOB_CONFIGURATION_TYPE = "jobConfigurationType"
    CONST_FREEFORM_TAG = "freeformTags"
    CONST_DEFINED_TAG = "definedTags"
    CONST_TIME_FROM = "timeFrom"
    CONST_TIME_TO = "timeTo"
    CONST_FEATURE_OPTION_DETAILS = "featureOptionsDetails"
    CONST_ERROR_DETAILS = "errorDetails"
    CONST_LIFECYCLE_STATE = "lifecycleState"
    CONST_DATA_FLOW_EXECUTION_OUTPUT = "dataFlowBatchExecutionOutput"
    CONST_VALIDATION_OUTPUT = "validation_output"
    CONST_DATA_FLOW_READ_WRITE_DETAIL = "data_flow_read_write_detail"
    CONST_DATA_READ_IN_BYTES = "data_read_in_bytes"
    CONST_DATA_WRITTEN_BYTES = "data_written_in_bytes"
    CONST_JOB_OUTPUT_DETAILS = "jobOutputDetails"

    attribute_map = {
        CONST_ID: "id",
        CONST_COMPARTMENT_ID: "compartment_id",
        CONST_DATASET_ID: "dataset_id",
        CONST_INGESTION_MODE: "ingestion_mode",
        CONST_JOB_CONFIGURATION_DETAILS: "job_configuration_details",
        CONST_JOB_CONFIGURATION_TYPE: "job_configuration_type",
        CONST_FEATURE_OPTION_DETAILS: "feature_option_details",
        CONST_TIME_FROM: "time_from",
        CONST_TIME_TO: "time_to",
        CONST_FREEFORM_TAG: "freeform_tags",
        CONST_DEFINED_TAG: "defined_tags",
        CONST_JOB_OUTPUT_DETAILS: "job_output_details",
    }

    def __init__(self, spec: Dict = None, **kwargs) -> None:
        """Initializes DatasetJob Resource.

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
        # Specify oci DatasetJob instance
        self.oci_fs_dataset_run = self._to_oci_fs_dataset_run(**kwargs)

    def _to_oci_fs_dataset_run(self, **kwargs):
        """Creates an `OCIDatasetJob` instance from the  `DatasetJob`.

        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.DatasetJob` accepts.

        Returns
        -------
        OCIDatasetJob
            The instance of the OCIDatasetJob.
        """
        fs_spec = {}
        for infra_attr, dsc_attr in self.attribute_map.items():
            value = self.get_spec(infra_attr)
            fs_spec[dsc_attr] = value
        fs_spec.update(**kwargs)
        return OCIDatasetJob(**fs_spec)

    @property
    def kind(self) -> str:
        """The kind of the object as showing in a YAML."""
        return "dataset_job"

    @property
    def compartment_id(self) -> str:
        return self.get_spec(self.CONST_COMPARTMENT_ID)

    @compartment_id.setter
    def compartment_id(self, value: str):
        self.with_compartment_id(value)

    def with_compartment_id(self, compartment_id: str) -> "DatasetJob":
        """Sets the compartment_id.

        Parameters
        ----------
        compartment_id: str
            The compartment_id.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_COMPARTMENT_ID, compartment_id)

    @property
    def dataset_id(self) -> str:
        return self.get_spec(self.CONST_DATASET_ID)

    @dataset_id.setter
    def dataset_id(self, value: str):
        self.with_dataset_id(value)

    def with_dataset_id(self, dataset_id: str) -> "DatasetJob":
        """Sets the dataset_id.

        Parameters
        ----------
        dataset_id: str
            The dataset id.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_DATASET_ID, dataset_id)

    @property
    def id(self) -> str:
        return self.get_spec(self.CONST_ID)

    def with_id(self, id: str) -> "DatasetJob":
        return self.set_spec(self.CONST_ID, id)

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
    ) -> "DatasetJob":
        """Sets the job configuration details.

        Parameters
        ----------
        job_configuration_type: JobConfigurationType
            The job_configuration_type of job
        kwargs: Dict[str, Any]
            Additional key value arguments

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(
            self.CONST_JOB_CONFIGURATION_DETAILS,
            {
                self.CONST_JOB_CONFIGURATION_TYPE: job_configuration_type.value,
                **kwargs,
            },
        )

    @property
    def ingestion_mode(self) -> str:
        return self.get_spec(self.CONST_INGESTION_MODE)

    @ingestion_mode.setter
    def ingestion_mode(
        self, ingestion_mode: Union[BatchIngestionMode, StreamingIngestionMode]
    ) -> "DatasetJob":
        return self.with_ingestion_mode(ingestion_mode)

    def with_ingestion_mode(
        self, ingestion_mode: Union[BatchIngestionMode, StreamingIngestionMode]
    ) -> "DatasetJob":
        """Sets the mode of the dataset ingestion mode.

        Parameters
        ----------
        ingestion_mode: IngestionMode
            The mode of the dataset ingestion mode.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_INGESTION_MODE, ingestion_mode.value)

    def with_error_details(self, error_details: str) -> "DatasetJob":
        """Sets the error details.

        Parameters
        ----------
        error_details: str
            The error_details.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_ERROR_DETAILS, error_details)

    def with_lifecycle_state(self, lifecycle_state: str) -> "DatasetJob":
        """Sets the lifecycle_state.

        Parameters
        ----------
        lifecycle_state: str
            The lifecycle_state.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_LIFECYCLE_STATE, lifecycle_state)

    @property
    def feature_option_details(self):
        return self.get_spec(self.CONST_FEATURE_OPTION_DETAILS)

    def with_feature_option_details(
        self, feature_option_details: FeatureOptionDetails
    ) -> "DatasetJob":
        """Sets the feature_option_details.

        Parameters
        ----------
        feature_option_details: FeatureOptionDetails

        Returns
        -------
        DatasetJob
            The FeatureGroupJob instance (self)
        """
        return self.set_spec(
            self.CONST_FEATURE_OPTION_DETAILS, feature_option_details.to_dict()
        )

    @property
    def job_output_details(self) -> Dict:
        return self.get_spec(self.CONST_JOB_OUTPUT_DETAILS)

    def with_job_output_details(self, job_output_details: Dict) -> "DatasetJob":
        """Sets the job output details.

        Parameters
        ----------
        job_output_details: Dict
            The job output details which contains error_details, validation_output and commit id.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self)
        """
        return self.set_spec(self.CONST_JOB_OUTPUT_DETAILS, job_output_details)

    @classmethod
    def from_id(cls, id: str) -> "DatasetJob":
        """Gets an existing dataset_run resource by Id.

        Parameters
        ----------
        id: str
            The dataset_run id.

        Returns
        -------
        FeatureStore
            An instance of DatasetJob resource.
        """
        return cls()._update_from_oci_fs_model(OCIDatasetJob.from_id(id))

    def create(self, **kwargs) -> "DatasetJob":
        """Creates dataset_run  resource.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.DatasetJob` accepts.

        Returns
        -------
        FeatureStore
            The DatasetJob instance (self)

        Raises
        ------
        ValueError
            If compartment id not provided.
        """

        if not self.compartment_id:
            raise ValueError("Compartment id must be provided.")

        payload = deepcopy(self._spec)
        payload.pop("id", None)
        logger.debug(f"Creating a dataset_run resource with payload {payload}")

        # Create entity
        logger.info("Saving entity.")
        self.oci_fs_dataset_run = self._to_oci_fs_dataset_run(**kwargs).create()
        self.with_id(self.oci_fs_dataset_run.id)
        return self

    def update(self, **kwargs) -> "DatasetJob":
        """Updates DatasetJob in the feature store.

        Parameters
        ----------
        kwargs
            Additional kwargs arguments.
            Can be any attribute that `feature_store.models.DatasetJob` accepts.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self).
        """

        if not self.id:
            raise ValueError(
                "DatasetJob needs to be saved to the feature store before it can be updated."
            )

        self.oci_fs_dataset_run = self._to_oci_fs_dataset_run(**kwargs).update()
        return self

    def _mark_job_complete(self, job_output_details) -> "DatasetJob":
        """
        Completes the current dataset job and returns a new `DatasetJob` object that represents it.

        :param job_output_details: The output details of the completed job.
        :return: A new `FeatureGroupJob` object representing the completed job.
        :raises ValueError: If the current `FeatureGroupJob` object has not been saved to the feature store yet.
        """
        # Check if the current FeatureGroupJob has been saved to the feature store
        if not self.id:
            raise ValueError(
                "DatasetJob needs to be saved to the feature store before it can be marked as completed."
            )

        # Update the job's output details with the given ones
        self.with_job_output_details(job_output_details)

        # Complete the job in the OCI feature store and return a new FeatureGroupJob object
        self.oci_fs_dataset_run = self._to_oci_fs_dataset_run().complete_dataset_job()
        return self

    def _update_from_oci_fs_model(
        self, oci_fs_dataset_run: OCIDatasetJob
    ) -> "DatasetJob":
        """Update the properties from an OCIDatasetJob object.

        Parameters
        ----------
        oci_fs_dataset_run: OCIDatasetJob
            An instance of OCIDatasetJob.

        Returns
        -------
        DatasetJob
            The DatasetJob instance (self).
        """

        # Update the main properties
        self.oci_fs_dataset_run = oci_fs_dataset_run
        dataset_run_details = oci_fs_dataset_run.to_dict()

        for infra_attr, dsc_attr in self.attribute_map.items():
            if infra_attr in dataset_run_details:
                self.set_spec(infra_attr, dataset_run_details[infra_attr])

        return self

    @classmethod
    def list_df(cls, compartment_id: str = None, **kwargs) -> "pandas.DataFrame":
        """Lists dataset_run resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering models.

        Returns
        -------
        pandas.DataFrame
            The list of the dataset_run resources in a pandas dataframe format.
        """
        records = []
        for oci_fs_dataset_run in OCIDatasetJob.list_resource(compartment_id, **kwargs):
            records.append(
                {
                    "id": oci_fs_dataset_run.id,
                    "display_name": oci_fs_dataset_run.display_name,
                    "time_created": oci_fs_dataset_run.time_created.strftime(
                        utils.date_format
                    ),
                    "time_updated": oci_fs_dataset_run.time_updated.strftime(
                        utils.date_format
                    ),
                    "lifecycle_state": oci_fs_dataset_run.lifecycle_state,
                    "created_by": f"...{oci_fs_dataset_run.created_by[-6:]}",
                    "compartment_id": f"...{oci_fs_dataset_run.compartment_id[-6:]}",
                    "dataset_id": oci_fs_dataset_run.dataset_id,
                }
            )
        return pandas.DataFrame.from_records(records)

    @classmethod
    def list(cls, compartment_id: str = None, **kwargs) -> List["DatasetJob"]:
        """Lists DatasetJob Resources in a given compartment.

        Parameters
        ----------
        compartment_id: (str, optional). Defaults to `None`.
            The compartment OCID.
        kwargs
            Additional keyword arguments for filtering DatasetJob.

        Returns
        -------
        List[DatasetJob]
            The list of the DatasetJob Resources.
        """
        return [
            cls()._update_from_oci_fs_model(oci_fs_dataset_run)
            for oci_fs_dataset_run in OCIDatasetJob.list_resource(
                compartment_id, **kwargs
            )
        ]

    def _random_display_name(self):
        """Generates a random display name."""
        return f"{self._PREFIX}-{utils.get_random_name_for_resource()}"

    def to_dict(self) -> Dict:
        """Serializes dataset_run  to a dictionary.

        Returns
        -------
        dict
            The dataset_run resource serialized as a dictionary.
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

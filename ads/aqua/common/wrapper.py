#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


##############################################
# This module contains aqua resource wrapper #
##############################################

import json
from typing import Union

from oci.data_science import DataScienceClient
from oci.data_science.models import JobRun
from oci.exceptions import ServiceError
from oci.resource_search.models import ResourceSummary

from ads.aqua import logger
from ads.aqua.data import AquaResourceIdentifier
from ads.aqua.utils import query_resource
from ads.common.utils import get_log_links
from ads.model.model_metadata import MetadataTaxonomyKeys


class AquaJobRun:
    """Wrapper of oci.data_science.models.JobRun and JobRun resource Summary."""

    _jobrun: JobRun = None
    _region: str = None
    _is_missing: bool = False
    _is_resource_summary: bool = True

    def __init__(self, jobrun: Union[JobRun, ResourceSummary] = None, region: str = ""):
        self._jobrun = jobrun
        self._region = region

        if jobrun is None:
            self._is_missing = True
        elif isinstance(jobrun, JobRun):
            self._is_resource_summary = False

    def is_missing(self) -> bool:
        """Check if jobrun object exists."""
        return self._is_missing

    def is_resource_summary(self) -> bool:
        """Check if jobrun object is ResourceSummary."""
        return self._is_resource_summary

    def to_aquaresourceidentifier(self) -> AquaResourceIdentifier:
        """
        Generates instance of AquaResourceIdentifier.

        Returns
        -------
        AquaResourceIdentifier
            Instance of AquaResourceIdentifier
        """
        return (
            AquaResourceIdentifier.from_data(
                dict(
                    id=self.id,
                    name=self.display_name,
                    region=self.region,
                )
            )
            if not self._is_missing
            else AquaResourceIdentifier()
        )

    def get_log_identifier(self) -> AquaResourceIdentifier:
        """
        Generates instance of AquaResourceIdentifier.

        Returns
        -------
        AquaResourceIdentifier
            Instance of AquaResourceIdentifier
        """
        return (
            AquaResourceIdentifier.from_data(
                dict(
                    id=self.log_id,
                    name=self.log_name,
                    url=self.log_url,
                    region=self.region,
                )
            )
            if not self._is_missing or not self.is_resource_summary
            else AquaResourceIdentifier()
        )

    def get_loggroup_identifier(self) -> AquaResourceIdentifier:
        """
        Generates instance of AquaResourceIdentifier.

        Returns
        -------
        AquaResourceIdentifier
            Instance of AquaResourceIdentifier
        """
        return (
            AquaResourceIdentifier.from_data(
                dict(
                    id=self.log_group_id,
                    name=self.log_group_name,
                    url=self.log_group_url,
                    region=self.region,
                )
            )
            if not self._is_missing or not self.is_resource_summary
            else AquaResourceIdentifier()
        )

    @property
    def lifecycle_state(self) -> str:
        return self._jobrun.lifecycle_state if not self._is_missing else ""

    @property
    def lifecycle_details(self) -> str:
        if not self._is_missing or self._is_resource_summary:
            return ""

        return self._jobrun.lifecycle_details

    @property
    def id(self) -> str:
        if self._is_missing:
            return ""

        if self._is_resource_summary:
            return self._jobrun.identifier

        return self._jobrun.id

    @property
    def display_name(self) -> str:
        return self._jobrun.display_name if not self._is_missing else ""

    @property
    def shape_name(self) -> str:
        if self._is_missing or self._is_resource_summary:
            return ""

        return self._jobrun.job_infrastructure_configuration_details.shape_name

    @property
    def region(self) -> str:
        return self._region

    @property
    def log_id(self) -> str:
        """The log ID from OCI logging service containing the logs from the job run."""
        if self._is_missing or self._is_resource_summary:
            return ""

        if not self._jobrun.log_details:
            return ""

        return self._jobrun.log_details.log_id or ""

    @property
    def log_group_id(self) -> str:
        """The log group ID from OCI logging service containing the logs from the job run."""
        if self._is_missing or self._is_resource_summary:
            return ""

        if not self._jobrun.log_details:
            return ""

        return self._jobrun.log_details.log_group_id or ""

    @property
    def log_group_url(self) -> str:
        """The loggroup console url."""
        if self._is_missing or self._is_resource_summary:
            return ""

        return get_log_links(region=self._region, log_group_id=self.log_group_id)

    @property
    def log_url(self) -> str:
        """The log console url."""
        if self._is_missing or self._is_resource_summary:
            return ""

        return get_log_links(
            region=self.region,
            log_group_id=self.log_group_id,
            log_id=self.log_id,
            compartment_id=self._jobrun.compartment_id,
            source_id=self._jobrun.id,
        )

    @property
    def log_name(self) -> str:
        """The log name."""
        if not self._is_missing or self._is_resource_summary:
            return ""
        log = query_resource(self.log_id, return_all=False)
        return log.display_name if log else ""

    @property
    def log_group_name(self) -> str:
        """The log group name."""
        if not self._is_missing or self._is_resource_summary:
            return ""

        loggroup = query_resource(self.log_group_id, return_all=False)
        return loggroup.display_name if loggroup else ""


class AquaModelResource:
    """Wrapper of model resource summary."""

    _model: ResourceSummary = None
    _metadata: dict = None
    _tags: dict = {}

    def __init__(self, model: ResourceSummary) -> None:
        self._model = model

    @property
    def id(self) -> dict:
        return self._model.identifier

    @property
    def tags(self) -> dict:
        """Returns all tags."""
        self._tags.update(self._model.defined_tags or {})
        self._tags.update(self._model.freeform_tags or {})
        return self._tags

    @property
    def display_name(self) -> dict:
        return self._model.display_name

    @property
    def time_created(self) -> dict:
        return self._model.time_created

    @property
    def lifecycle_state(self) -> dict:
        return self._model.lifecycle_state

    @property
    def additional_details(self) -> dict:
        return self._model.additional_details

    @property
    def metadata(self) -> list:
        if not self._metadata:
            self._metadata = self.additional_details.get("metadata", [])
        return self._metadata

    @property
    def description(self):
        if not hasattr(self, "_description"):
            self._description = self._model.additional_details.get("description", "")

        return self._description

    @property
    def created_by(self):
        if not hasattr(self, "_created_by"):
            self._created_by = self._model.additional_details.get("createdBy", "")

        return self._created_by

    @property
    def model_version_set_id(self):
        if not hasattr(self, "_model_version_set_id"):
            self._model_version_set_id = self._model.additional_details.get(
                "modelVersionSetId", ""
            )

        return self._model_version_set_id

    @property
    def model_version_set_name(self):
        if not hasattr(self, "_model_version_set_name"):
            self._model_version_set_name = self._model.additional_details.get(
                "modelVersionSetName", ""
            )

        return self._model_version_set_name

    @property
    def project_id(self):
        if not hasattr(self, "_project_id"):
            self._project_id = self._model.additional_details.get("projectId", "")

        return self._project_id

    @property
    def version_label(self):
        if not hasattr(self, "_version_label"):
            self._version_label = self._model.additional_details.get("versionLabel", "")

        return self._version_label

    @property
    def introspection(self):
        if not hasattr(self, "_introspection"):
            try:
                self._introspection = json.loads(
                    self.get(MetadataTaxonomyKeys.ARTIFACT_TEST_RESULT)
                )
            except:
                self._introspection = {}

        return self._introspection

    def get(self, key: str):
        """Gets metadata from model."""
        if hasattr(self, f"_{key}"):
            if getattr(self, f"_{key}") == "":
                logger.debug(f"Missing `{key}` in custom metadata for {self.id}.")
            return getattr(self, f"_{key}")

        for metadata in self.metadata:
            if metadata.get("key") == key:
                setattr(self, f"_{key}", metadata.get("value", " "))
                return getattr(self, f"_{key}")

        logger.debug(f"Missing key=`{key}` in custom metadata for {self.id}.")
        setattr(self, f"_{key}", "")
        return getattr(self, f"_{key}")

    def check_artifact_exist(self, ds_client: DataScienceClient) -> bool:
        """Checks if the model artifact exists.

        Parameters
        ----------
        ds_client: oci.data_science.DataScienceClient
            Invoke head_model_artifact for checking.


        Return
        ------
        bool:
            True if model artifact exists.
        """
        try:
            response = ds_client.head_model_artifact(model_id=self.id)
            return True if response.status == 200 else False
        except ServiceError as ex:
            if ex.status == 404:
                logger.debug(f"Evaluation artifact not found for {self.id}.")
                return False

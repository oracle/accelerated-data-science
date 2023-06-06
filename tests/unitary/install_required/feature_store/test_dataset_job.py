#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import datetime
import hashlib
import json
from copy import deepcopy
from unittest.mock import patch

import pytest

from ads.feature_store.dataset_job import (
    DatasetJob,
    IngestionMode,
    JobConfigurationType,
)
from ads.feature_store.service.oci_dataset_job import OCIDatasetJob

DATASET_JOB_OCID = "ocid1.dataset_job.oc1.iad.xxx"

DATASET_JOB_PAYLOAD = {
    "jobConfigurationDetails": {"jobConfigurationType": "SPARK_BATCH_AUTOMATIC"},
    "compartmentId": "compartmentId",
    "datasetId": "ocid1.dataset.oc1.iad.xxx",
    "ingestionMode": "OVERWRITE",
}


class TestDatasetJob:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": DATASET_JOB_PAYLOAD["compartmentId"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(DATASET_JOB_PAYLOAD)
        self.mock_dsc_dataset_job = DatasetJob(**self.payload)

    def prepare_dict(self, data):
        return data

    def hash_dict(self, data):
        return hashlib.sha1(
            json.dumps(self.prepare_dict(data), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def compare_dict(self, dict1, dict2):
        print(
            f"dict1_hash: {self.hash_dict(dict1)}; dict2_hash: {self.hash_dict(dict2)}"
        )
        return self.hash_dict(dict1) == self.hash_dict(dict2)

    @patch.object(
        DatasetJob,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_dataset_job = DatasetJob()
        assert dsc_dataset_job.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD

    @patch.object(DatasetJob, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_dataset_job = DatasetJob(**self.payload)
        assert self.prepare_dict(
            dsc_dataset_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(DatasetJob, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_dataset_job = (
            DatasetJob()
            .with_compartment_id(self.payload["compartmentId"])
            .with_ingestion_mode(IngestionMode.OVERWRITE)
            .with_dataset_id(self.payload["datasetId"])
            .with_job_configuration_details(JobConfigurationType.SPARK_BATCH_AUTOMATIC)
        )
        assert self.prepare_dict(
            dsc_dataset_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_dataset_job = (
            DatasetJob()
            .with_compartment_id(self.payload["compartmentId"])
            .with_ingestion_mode(IngestionMode.OVERWRITE)
            .with_dataset_id(self.payload["datasetId"])
            .with_job_configuration_details(JobConfigurationType.SPARK_BATCH_AUTOMATIC)
        )
        assert self.prepare_dict(
            dsc_dataset_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(DatasetJob, "_update_from_oci_fs_model")
    @patch.object(OCIDatasetJob, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing dataset in a given compartment."""
        mock_list_resource.return_value = [OCIDatasetJob(**DATASET_JOB_PAYLOAD)]
        mock__update_from_oci_fs_model.return_value = DatasetJob(**self.payload)
        result = DatasetJob.list(
            compartment_id="test_compartment_id",
            extra_tag="test_cvalue",
        )
        mock_list_resource.assert_called_with(
            "test_compartment_id",
            **{"extra_tag": "test_cvalue"},
        )
        assert len(result) == 1
        assert self.prepare_dict(result[0].to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(DatasetJob, "_update_from_oci_fs_model")
    @patch.object(OCIDatasetJob, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIDatasetJob(**DATASET_JOB_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = DatasetJob(**self.payload)
        result = DatasetJob.from_id(DATASET_JOB_OCID)

        mock_oci_from_id.assert_called_with(DATASET_JOB_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIDatasetJob, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience dataset."""
        oci_dsc_model = OCIDatasetJob(**DATASET_JOB_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check rundom display name
        result = self.mock_dsc_dataset_job.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(DatasetJob, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience dataset."""
        dsc_dataset_job = DatasetJob()
        with pytest.raises(ValueError, match="Compartment id must be provided."):
            dsc_dataset_job.create()

    def test_to_dict(self):
        """Tests serializing dataset to a dictionary."""
        test_dict = self.mock_dsc_dataset_job.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_dataset_job.kind
        assert test_dict["type"] == self.mock_dsc_dataset_job.type

    def test_from_dict(self):
        """Tests loading dataset instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_dataset_job.to_dict()["spec"]
        ) == self.prepare_dict(
            DatasetJob.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_dataset_job._PREFIX}-test_name"
        assert self.mock_dsc_dataset_job._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIDatasetJob` instance from the  `DatasetJob`."""
        with patch.object(OCIDatasetJob, "sync"):
            test_oci_dsc_dataset_job = OCIDatasetJob(**DATASET_JOB_PAYLOAD)
            test_oci_dsc_dataset_job.id = None
            test_oci_dsc_dataset_job.lifecycle_state = None
            test_oci_dsc_dataset_job.created_by = None
            test_oci_dsc_dataset_job.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_dataset_job.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_dataset_job._to_oci_fs_dataset_run().to_dict()
            )

            test_oci_dsc_dataset_job.display_name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_dataset_job.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_dataset_job._to_oci_fs_dataset_run(
                    display_name="new_name"
                ).to_dict()
            )

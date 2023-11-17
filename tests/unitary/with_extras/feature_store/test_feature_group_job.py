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

from ads.feature_store.feature_group_job import (
    FeatureGroupJob,
    IngestionMode,
    JobConfigurationType,
)
from ads.feature_store.service.oci_feature_group_job import OCIFeatureGroupJob

FEATURE_GROUP_JOB_OCID = "ocid1.feature_group_job.oc1.iad.xxx"

FEATURE_GROUP_JOB_PAYLOAD = {
    "jobConfigurationDetails": {"jobConfigurationType": "SPARK_BATCH_AUTOMATIC"},
    "compartmentId": "compartmentId",
    "featureGroupId": "ocid1.feature_group.oc1.iad.xxx",
    "ingestionMode": "OVERWRITE",
}


class TestFeatureGroupJob:
    test_config = {
        "tenancy": "ocid1.tenancy.oc1..xxx",  # must be a real-like ocid
        "user": "ocid1.user.oc1..xxx",  # must be a real-like ocid
        "fingerprint": "00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00",
        "key_file": "<path>/<to>/<key_file>",
        "region": "test_region",
    }

    DEFAULT_PROPERTIES_PAYLOAD = {
        "compartmentId": FEATURE_GROUP_JOB_PAYLOAD["compartmentId"],
    }

    def setup_class(cls):
        cls.mock_date = datetime.datetime(3022, 7, 1)

    def setup_method(self):
        self.payload = deepcopy(FEATURE_GROUP_JOB_PAYLOAD)
        self.mock_dsc_feature_group_job = FeatureGroupJob(**self.payload)

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
        FeatureGroupJob,
        "_load_default_properties",
        return_value=DEFAULT_PROPERTIES_PAYLOAD,
    )
    def test__init__default_properties(self, mock_load_default_properties):
        dsc_feature_group_job = FeatureGroupJob()
        assert (
            dsc_feature_group_job.to_dict()["spec"] == self.DEFAULT_PROPERTIES_PAYLOAD
        )

    @patch.object(FeatureGroupJob, "_load_default_properties", return_value={})
    def test__init__(self, mock_load_default_properties):
        dsc_feature_group_job = FeatureGroupJob(**self.payload)
        assert self.prepare_dict(
            dsc_feature_group_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(FeatureGroupJob, "_load_default_properties", return_value={})
    def test_with_methods_1(self, mock_load_default_properties):
        """Tests all with methods."""
        dsc_feature_group_job = (
            FeatureGroupJob()
            .with_compartment_id(self.payload["compartmentId"])
            .with_ingestion_mode(IngestionMode.OVERWRITE)
            .with_feature_group_id(self.payload["featureGroupId"])
            .with_job_configuration_details(JobConfigurationType.SPARK_BATCH_AUTOMATIC)
        )
        assert self.prepare_dict(
            dsc_feature_group_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    def test_with_methods_2(self):
        """Tests all with methods."""
        dsc_feature_group_job = (
            FeatureGroupJob()
            .with_compartment_id(self.payload["compartmentId"])
            .with_ingestion_mode(IngestionMode.OVERWRITE)
            .with_feature_group_id(self.payload["featureGroupId"])
            .with_job_configuration_details(JobConfigurationType.SPARK_BATCH_AUTOMATIC)
        )
        assert self.prepare_dict(
            dsc_feature_group_job.to_dict()["spec"]
        ) == self.prepare_dict(self.payload)

    @patch.object(FeatureGroupJob, "_update_from_oci_fs_model")
    @patch.object(OCIFeatureGroupJob, "list_resource")
    def test_list(self, mock_list_resource, mock__update_from_oci_fs_model):
        """Tests listing feature group job in a given compartment."""
        mock_list_resource.return_value = [
            OCIFeatureGroupJob(**FEATURE_GROUP_JOB_PAYLOAD)
        ]
        mock__update_from_oci_fs_model.return_value = FeatureGroupJob(**self.payload)
        result = FeatureGroupJob.list(
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

    @patch.object(FeatureGroupJob, "_update_from_oci_fs_model")
    @patch.object(OCIFeatureGroupJob, "from_id")
    def test_from_id(self, mock_oci_from_id, mock__update_from_oci_fs_model):
        """Tests getting an existing model by OCID."""
        mock_oci_model = OCIFeatureGroupJob(**FEATURE_GROUP_JOB_PAYLOAD)
        mock_oci_from_id.return_value = mock_oci_model
        mock__update_from_oci_fs_model.return_value = FeatureGroupJob(**self.payload)
        result = FeatureGroupJob.from_id(FEATURE_GROUP_JOB_OCID)

        mock_oci_from_id.assert_called_with(FEATURE_GROUP_JOB_OCID)
        mock__update_from_oci_fs_model.assert_called_with(mock_oci_model)
        assert self.prepare_dict(result.to_dict()["spec"]) == self.prepare_dict(
            self.payload
        )

    @patch.object(OCIFeatureGroupJob, "create")
    def test_create_success(
        self,
        mock_oci_dsc_model_create,
    ):
        """Tests creating datascience feature group job."""
        oci_dsc_model = OCIFeatureGroupJob(**FEATURE_GROUP_JOB_PAYLOAD)
        mock_oci_dsc_model_create.return_value = oci_dsc_model

        # to check rundom display name
        result = self.mock_dsc_feature_group_job.create()
        mock_oci_dsc_model_create.assert_called()

    @patch.object(FeatureGroupJob, "_load_default_properties", return_value={})
    def test_create_fail(self, mock__load_default_properties):
        """Tests creating datascience feature group job."""
        dsc_feature_group_job = FeatureGroupJob()
        with pytest.raises(ValueError, match="Compartment id must be provided."):
            dsc_feature_group_job.create()

    def test_to_dict(self):
        """Tests serializing feature group job to a dictionary."""
        test_dict = self.mock_dsc_feature_group_job.to_dict()
        assert self.prepare_dict(test_dict["spec"]) == self.prepare_dict(self.payload)
        assert test_dict["kind"] == self.mock_dsc_feature_group_job.kind
        assert test_dict["type"] == self.mock_dsc_feature_group_job.type

    def test_from_dict(self):
        """Tests loading feature group job instance from a dictionary of configurations."""
        assert self.prepare_dict(
            self.mock_dsc_feature_group_job.to_dict()["spec"]
        ) == self.prepare_dict(
            FeatureGroupJob.from_dict({"spec": self.payload}).to_dict()["spec"]
        )

    @patch("ads.common.utils.get_random_name_for_resource", return_value="test_name")
    def test__random_display_name(self, mock_get_random_name_for_resource):
        """Tests generating a random display name."""
        expected_result = f"{self.mock_dsc_feature_group_job._PREFIX}-test_name"
        assert self.mock_dsc_feature_group_job._random_display_name() == expected_result

    @patch("oci.config.from_file", return_value=test_config)
    @patch("oci.signer.load_private_key_from_file")
    def test__to_oci_fs_entity(self, mock_load_key_file, mock_config_from_file):
        """Tests creating an `OCIFeatureGroupJob` instance from the  `FeatureGroupJob`."""
        with patch.object(OCIFeatureGroupJob, "sync"):
            test_oci_dsc_entity = OCIFeatureGroupJob(**FEATURE_GROUP_JOB_PAYLOAD)
            test_oci_dsc_entity.id = None
            test_oci_dsc_entity.lifecycle_state = None
            test_oci_dsc_entity.created_by = None
            test_oci_dsc_entity.time_created = None

            assert self.prepare_dict(
                test_oci_dsc_entity.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_feature_group_job._to_oci_fs_feature_group_run().to_dict()
            )

            test_oci_dsc_entity.display_name = "new_name"
            assert self.prepare_dict(
                test_oci_dsc_entity.to_dict()
            ) == self.prepare_dict(
                self.mock_dsc_feature_group_job._to_oci_fs_feature_group_run(
                    display_name="new_name"
                ).to_dict()
            )

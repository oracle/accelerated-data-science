#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pytest

from ads.opctl.operator.common.dictionary_merger import DictionaryMerger


class TestDictionaryMerger:
    def test_init(self):
        updates = {"my.key.field1": "value1", "my.key.field2": "value2"}
        system_keys = ["my_special_key1", "my_special_key_2"]
        dictionary_merger = DictionaryMerger(updates=updates, system_keys=system_keys)

        assert dictionary_merger.updates == updates
        assert dictionary_merger.system_keys == set(system_keys).union(
            DictionaryMerger._SYSTEM_KEYS
        )

    @pytest.mark.parametrize(
        "input_data, updates, system_keys, expected_result",
        [
            (
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "50",
                                "compartmentId": "test_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "test_id",
                            },
                        }
                    },
                },
                {
                    "spec.infrastructure.spec.blockStorageSize": "10",
                    "spec.infrastructure.spec.projectId": "new_project_id",
                    "spec.infrastructure.spec.compartmentId": "new_compartment_id",
                },
                ["compartmentId"],
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "10",
                                "compartmentId": "test_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "new_project_id",
                            },
                        }
                    },
                },
            ),
            (
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "50",
                                "compartmentId": "test_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "test_id",
                            },
                        }
                    },
                },
                {
                    "spec.infrastructure.spec": {},
                    "spec.infrastructure.spec.blockStorageSize": "10",
                    "spec.infrastructure.spec.projectId": "new_project_id",
                    "spec.infrastructure.spec.compartmentId": "new_compartment_id",
                },
                None,
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "10",
                                "compartmentId": "new_compartment_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "new_project_id",
                            },
                        }
                    },
                },
            ),
            (
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "50",
                                "compartmentId": "test_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "test_id",
                            },
                        }
                    },
                },
                None,
                [],
                {
                    "kind": "job",
                    "spec": {
                        "infrastructure": {
                            "kind": "infrastructure",
                            "spec": {
                                "blockStorageSize": "50",
                                "compartmentId": "test_id",
                                "jobInfrastructureType": "ME_STANDALONE",
                                "jobType": "DEFAULT",
                                "projectId": "test_id",
                            },
                        }
                    },
                },
            ),
        ],
    )
    def test_update(self, input_data, updates, system_keys, expected_result):
        """Ensures that the update method correctly updates the dictionary."""
        test_result = DictionaryMerger(updates=updates, system_keys=system_keys).merge(
            input_data
        )
        assert test_result == expected_result

#!/usr/bin/env python
# -*- coding: utf-8 -*--
import ads

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.feature_store.feature_store_registrar import FeatureStoreRegistrar


from ads.common import utils, auth
from tests.integration.feature_store.test_base import FeatureStoreTestCase

TEST_FILE_PATH = "tests/data/feature_store_minimal.yaml"


class TestYamlRegistrar(FeatureStoreTestCase):
    def test_create_with_valid_yaml(self):
        """
        Integration test which creates single feature group, entity,
        transformation, dataset using valid yaml definition file
        """

        registrar = FeatureStoreRegistrar.from_yaml(uri=TEST_FILE_PATH)
        resp = registrar.create()
        assert resp is not None
        assert len(resp) == 5
        # entity creation validation
        assert resp[1] is not None
        assert len(resp[1]) == 1
        # transformation creation validation
        assert resp[2] is not None
        assert len(resp[1]) == 1
        # feature group creation validation
        assert resp[3] is not None
        assert len(resp[1]) == 1
        # entity dataset validation
        assert resp[4] is not None
        assert len(resp[1]) == 1

        for dataset in resp[4]:
            dataset.delete()
        for feature_group in resp[3]:
            feature_group.delete()
        for transformation in resp[2]:
            transformation.delete()
        for entity in resp[1]:
            entity.delete()
        feature_store = resp[0]
        feature_store.delete()

#!/usr/bin/env python

# Copyright (c) 2021, 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.config import NB_SESSION_OCID, JOB_RUN_OCID
from ads.common.model_artifact import _TRAINING_RESOURCE_OCID


class TestTrainingId:
    def test_search_structured(self):
        """Test _TRAINING_RESOURCE_OCID"""
        assert _TRAINING_RESOURCE_OCID == (JOB_RUN_OCID or NB_SESSION_OCID)

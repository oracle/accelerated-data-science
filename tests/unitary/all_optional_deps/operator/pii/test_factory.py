#!/usr/bin/env python

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import pytest
from scrubadub_spacy.detectors.spacy import SpacyEntityDetector

from ads.opctl.operator.lowcode.pii.model.factory import (
    PiiDetectorFactory,
    UnSupportedDetectorError,
)


class TestPiiDetectorFactory:
    def test_get_default_detector(self):
        detector_type = "default"
        entity = "phone"
        model = None
        expected_result = "phone"
        detector = PiiDetectorFactory.get_detector(
            detector_type=detector_type, entity=entity, model=model
        )
        assert detector == expected_result

    @pytest.mark.parametrize(
        "detector_type, entity, model",
        [
            ("spacy", "person", "en_core_web_sm"),
            ("spacy", "other", "en_core_web_sm"),
            # ("spacy", "org", "en_core_web_trf"),
            # ("spacy", "loc", "en_core_web_md"),
            # ("spacy", "date", "en_core_web_lg"),
        ],
    )
    def test_get_spacy_detector(self, detector_type, entity, model):
        detector = PiiDetectorFactory.get_detector(
            detector_type=detector_type, entity=entity, model=model
        )
        assert isinstance(detector, SpacyEntityDetector)
        assert entity.upper() in detector.filth_cls_map

    def test_get_detector_fail(self):
        detector_type = "unknow"
        entity = "myentity"
        model = None
        with pytest.raises(UnSupportedDetectorError):
            PiiDetectorFactory.get_detector(
                detector_type=detector_type, entity=entity, model=model
            )

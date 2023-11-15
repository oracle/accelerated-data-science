#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import uuid

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl.operator.lowcode.pii.constant import SupportedDetector
from ads.opctl.operator.lowcode.pii.utils import construct_filth_cls_name


class UnSupportedDetectorError(Exception):
    def __init__(self, dtype: str):
        super().__init__(
            f"Detector: `{dtype}` "
            f"is not supported. Supported models: {SupportedDetector.values}"
        )


class PiiBaseDetector:
    @classmethod
    def construct(cls, **kwargs):
        raise NotImplementedError


class BuiltInDetector(PiiBaseDetector):
    @classmethod
    def construct(cls, entity, **kwargs):
        return entity


class SpacyDetector(PiiBaseDetector):
    DEFAULT_SPACY_NAMED_ENTITIES = ["DATE", "FAC", "GPE", "LOC", "ORG", "PER", "PERSON"]
    DEFAULT_SPACY_MODEL = "en_core_web_trf"

    @classmethod
    @runtime_dependency(module="scrubadub", install_from=OptionalDependency.PII)
    @runtime_dependency(module="scrubadub_spacy", install_from=OptionalDependency.PII)
    def construct(cls, entity, model, **kwargs):
        spacy_entity_detector = scrubadub_spacy.detectors.spacy.SpacyEntityDetector(
            named_entities=[entity],
            name=f"spacy_{uuid.uuid4()}",
            model=model,
        )
        if entity.upper() not in cls.DEFAULT_SPACY_NAMED_ENTITIES:
            filth_cls = type(
                construct_filth_cls_name(entity),
                (scrubadub.filth.Filth,),
                {"type": entity.upper()},
            )
            spacy_entity_detector.filth_cls_map[entity.upper()] = filth_cls
        return spacy_entity_detector


class PiiDetectorFactory:
    """
    The factory class helps to instantiate proper detector object based on the detector config.
    """

    _MAP = {
        SupportedDetector.DEFAULT: BuiltInDetector,
        SupportedDetector.SPACY: SpacyDetector,
    }

    @classmethod
    def get_detector(
        cls,
        detector_type,
        entity,
        model=None,
    ):
        if detector_type not in cls._MAP:
            raise UnSupportedDetectorError(detector_type)

        return cls._MAP[detector_type].construct(entity=entity, model=model)

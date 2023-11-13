#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import scrubadub

from ads.opctl import logger
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.pii.model.factory import PiiDetectorFactory
from ads.opctl.operator.lowcode.pii.model.processor import (
    POSTPROCESSOR_MAP,
    SUPPORTED_REPLACER,
    Remover,
)

SUPPORT_ACTIONS = ["mask", "remove", "anonymize"]


class DetectorType:
    DEFAULT = "default"


class Scrubber:
    def __init__(self, config: str or "PiiOperatorConfig" or dict):
        logger.info(f"Loading config from {config}")
        if isinstance(config, str):
            config = _load_yaml_from_uri(config)

        self.config = config
        self.scrubber = scrubadub.Scrubber()

        self.detectors = []
        self.spacy_model_detectors = []
        self.post_processors = {}  # replacer_name -> replacer_obj

        self._reset_scrubber()

    def _reset_scrubber(self):
        # Clean up default detectors
        defautls_enable = self.scrubber._detectors.copy()
        for d in defautls_enable:
            self.scrubber.remove_detector(d)

    def _register(self, name, dtype, model, action, mask_with: str = None):
        if action not in SUPPORT_ACTIONS:
            raise ValueError(
                f"Not supported `action`: {action}. Please select from {SUPPORT_ACTIONS}."
            )

        detector = PiiDetectorFactory.get_detector(
            detector_type=dtype, entity=name, model=model
        )
        self.scrubber.add_detector(detector)

        if action == "anonymize":
            entity = (
                detector
                if isinstance(detector, str)
                else detector.filth_cls_map[name.upper()].type
            )
            if entity in SUPPORTED_REPLACER.keys():
                replacer_name = SUPPORTED_REPLACER.get(entity).name
                replacer = self.post_processors.get(
                    replacer_name, POSTPROCESSOR_MAP.get(replacer_name)()
                )
                if hasattr(replacer, "_ENTITIES"):
                    replacer._ENTITIES.append(name)
                self.post_processors[replacer_name] = replacer
            else:
                raise ValueError(
                    f"Not supported `action` {action} for this entity {name}. Please try with other action."
                )

        if action == "remove":
            remover = self.post_processors.get("remover", Remover())
            remover._ENTITIES.append(name)
            self.post_processors["remover"] = remover

    def config_scrubber(self):
        """Returns an instance of srubadub.Scrubber."""
        spec = (
            self.config["spec"] if isinstance(self.config, dict) else self.config.spec
        )
        detectors = spec["detectors"] if isinstance(spec, dict) else spec.detector

        self.scrubber.redact_spec_file = spec

        for detector in detectors:
            # example format for detector["name"]: default.phone or spacy.en_core_web_trf.person
            d = detector["name"].split(".")
            dtype = d[0]
            dname = d[1] if len(d) == 2 else d[2]
            model = None if len(d) == 2 else d[1]

            action = detector.get("action", "mask")
            # mask_with = detector.get("mask_with", None)
            self._register(
                name=dname,
                dtype=dtype,
                model=model,
                action=action,
                # mask_with=mask_with,
            )

        self._register_post_processor()
        return self.scrubber

    def _register_post_processor(self):
        for _, v in self.post_processors.items():
            self.scrubber.add_post_processor(v)


def scrub(text, spec_file=None, scrubber=None):
    if not scrubber:
        scrubber = Scrubber(config=spec_file).config_scrubber()
    return scrubber.clean(text)


def detect(text, spec_file=None, scrubber=None):
    if not scrubber:
        scrubber = Scrubber(config=spec_file).config_scrubber()
    return list(scrubber.iter_filth(text, document_name=None))

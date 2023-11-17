#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.opctl import logger
from ads.opctl.operator.common.utils import _load_yaml_from_uri
from ads.opctl.operator.lowcode.pii.model.factory import PiiDetectorFactory
from ads.opctl.operator.lowcode.pii.constant import (
    SupportedAction,
    SupportedDetector,
)
from ads.opctl.operator.lowcode.pii.model.processor import (
    POSTPROCESSOR_MAP,
    SUPPORTED_REPLACER,
    Remover,
)


class PiiScrubber:
    """Class used for config scrubber and count the detectors in use."""

    @runtime_dependency(module="scrubadub", install_from=OptionalDependency.PII)
    def __init__(self, config):
        logger.info(f"Loading config from {config}")
        if isinstance(config, str):
            config = _load_yaml_from_uri(config)

        self.config = config
        self.spec = (
            self.config["spec"] if isinstance(self.config, dict) else self.config.spec
        )
        self.detector_spec = (
            self.spec["detectors"]
            if isinstance(self.spec, dict)
            else self.spec.detectors
        )

        self.scrubber = scrubadub.Scrubber()

        self.detectors = []
        self.entities = []
        self.spacy_model_detectors = []
        self.post_processors = {}

        self._reset_scrubber()

    def _reset_scrubber(self):
        # Clean up default detectors
        defautls_enable = self.scrubber._detectors.copy()
        for d in defautls_enable:
            self.scrubber.remove_detector(d)

    def _register(self, name, dtype, model, action, mask_with: str = None):
        if action not in SupportedAction.values():
            raise ValueError(
                f"Not supported `action`: {action}. Please select from {SupportedAction.values()}."
            )

        detector = PiiDetectorFactory.get_detector(
            detector_type=dtype, entity=name, model=model
        )
        self.scrubber.add_detector(detector)
        self.entities.append(name)

        if action == SupportedAction.ANONYMIZE:
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
                    f"Not supported `action` {action} for this entity `{name}`. Please try with other action."
                )

        if action == SupportedAction.REMOVE:
            remover = self.post_processors.get("remover", Remover())
            remover._ENTITIES.append(name)
            self.post_processors["remover"] = remover

    def config_scrubber(self):
        """Returns an instance of srubadub.Scrubber."""

        self.scrubber.redact_spec_file = self.spec

        for detector in self.detector_spec:
            # example format for detector["name"]: default.phone or spacy.en_core_web_trf.person
            d = detector["name"].split(".")
            dtype = d[0]
            dname = d[1] if len(d) == 2 else d[2]
            model = None if len(d) == 2 else d[1]

            action = detector.get("action", SupportedAction.MASK)
            self._register(
                name=dname,
                dtype=dtype,
                model=model,
                action=action,
            )
            if dtype == SupportedDetector.SPACY:
                exist = False
                for spacy_detectors in self.spacy_model_detectors:
                    if spacy_detectors["model"] == model:
                        spacy_detectors["spacy_entites"].append(dname)
                        exist = True
                        break
                if not exist:
                    self.spacy_model_detectors.append(
                        {"model": model, "spacy_entites": [dname]}
                    )

        self._register_post_processor()

        self.detectors = list(self.scrubber._detectors.values())
        return self.scrubber

    def _register_post_processor(self):
        for _, v in self.post_processors.items():
            self.scrubber.add_post_processor(v)


def scrub(text, config=None, scrubber=None):
    if not scrubber:
        scrubber = PiiScrubber(config=config).config_scrubber()
    return scrubber.clean(text)


def detect(text, config=None, scrubber=None):
    if not scrubber:
        scrubber = PiiScrubber(config=config).config_scrubber()
    return list(scrubber.iter_filth(text, document_name=None))

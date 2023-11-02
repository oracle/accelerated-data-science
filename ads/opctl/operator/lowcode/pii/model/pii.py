#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import scrubadub
import scrubadub_spacy
import os
import re
import logging
import uuid

from ads.opctl.operator.lowcode.pii.model.utils import (
    load_html,
    SupportInputFormat,
    from_yaml,
    _safe_get_spec,
    default_config,
    _read_from_file,
    load_rtf,
    construct_filth_cls_name,
    _write_to_file,
    _process_pos,
    ReportContextKey,
)
from ads.opctl.operator.lowcode.pii.model.processor import POSTPROCESSOR_MAP

DEFAULT_SPACY_NAMED_ENTITIES = ["DATE", "FAC", "GPE", "LOC", "ORG", "PER", "PERSON"]
DEFAULT_SPACY_MODEL = "en_core_web_trf"


def config_post_processor(spec: dict):
    """Return class scrubadub.post_processors.base.PostProcessor."""
    name = _safe_get_spec(spec, "name", "").lower()
    if not name in POSTPROCESSOR_MAP.keys():
        raise ValueError(
            f"Unsupport post processor: {name}. Only support {POSTPROCESSOR_MAP.keys()}."
        )
    cls = POSTPROCESSOR_MAP.get(name)
    if name == "number_replacer":
        cls._ENTITIES = _safe_get_spec(spec, "entities", cls._ENTITIES)

    return cls


def config_spacy_detector(spec: dict):
    """Return an instance of scrubadub_spacy.detectors.spacy.SpacyEntityDetector."""
    model = _safe_get_spec(spec, "model", DEFAULT_SPACY_MODEL)

    named_entities = [x.upper() for x in spec.get("named_entities", [])]
    spacy_entity_detector = scrubadub_spacy.detectors.spacy.SpacyEntityDetector(
        named_entities=named_entities,
        name=f"spacy_{uuid.uuid4()}",
        model=model,
    )
    for named_entity in named_entities:
        # DEFAULT_SPACY_NAMED_ENTITIES has been registered in filth_cls_map already.
        if named_entity in DEFAULT_SPACY_NAMED_ENTITIES:
            continue

        filth_cls = type(
            construct_filth_cls_name(named_entity),
            (scrubadub.filth.Filth,),
            {"type": named_entity.upper()},
        )
        spacy_entity_detector.filth_cls_map[named_entity.upper()] = filth_cls
    return spacy_entity_detector


def config_scrubber(
    config: str or dict = None,
):
    """
    Returns an instance of srubadub.Scrubber.

    Args:
        config: A path to a yaml file or a dict.

    Returns:
        An instance of srubadub.Scrubber, which has been configured with the given config.
    """
    if not config:
        config = default_config()
    logging.info(f"Loading config from {config}")

    if isinstance(config, str):
        config = from_yaml(uri=config)

    redact_spec_file = config["redactor"]

    detector_list = []
    scrubber = scrubadub.Scrubber()
    scrubber.redact_spec_file = redact_spec_file

    # Clean up default detectors
    defautls_enable = scrubber._detectors.copy()
    for d in defautls_enable:
        scrubber.remove_detector(d)

    # Add scrubber built-in detectors
    for detector in _safe_get_spec(redact_spec_file, "detectors", []):
        detector_list.append(detector)

    # Add spacy detectors
    for spec in _safe_get_spec(redact_spec_file, "spacy_detectors", []):
        spacy_entity_detector = config_spacy_detector(spec=spec)
        detector_list.append(spacy_entity_detector)

    # Add custom detectors
    for custom in _safe_get_spec(redact_spec_file, "custom_detectors", []):
        patterns = custom.get("patterns", "")

        class CustomFilth(scrubadub.filth.Filth):
            type = custom.get("label", "").upper()

        class CustomDetector(scrubadub.detectors.RegexDetector):
            filth_cls = CustomFilth
            regex = re.compile(
                rf"{patterns}",
            )
            name = custom.get("name")

        detector_list.append(CustomDetector())

    for detector in detector_list:
        scrubber.add_detector(detector)

    # Add post-processor
    for post_processor in _safe_get_spec(redact_spec_file, "anonymization", []):
        scrubber.add_post_processor(config_post_processor(post_processor))

    return scrubber


def scrub(text, spec_file=None, scrubber=None):
    if not scrubber:
        scrubber = config_scrubber(spec_file)
    return scrubber.clean(text)


def detect(text, spec_file=None, scrubber=None):
    if not scrubber:
        scrubber = config_scrubber(spec_file)
    return list(scrubber.iter_filth(text, document_name=None))


def _get_report_(
    input_path, output_path, scrubber=None, report_context=None, subdirectory=None
) -> None:
    filename_with_ext = os.path.basename(input_path)
    file_name, file_ext = os.path.splitext(filename_with_ext)

    report_text = ""
    if file_ext == SupportInputFormat.PLAIN:
        report_text = _read_from_file(input_path)
    elif file_ext == SupportInputFormat.HTML:
        report_text = load_html(uri=input_path)
    elif file_ext == SupportInputFormat.RTF:
        report_text = load_rtf(uri=input_path)
    else:
        raise ValueError(
            f"Unsupport file format: {file_ext}. Only support {SupportInputFormat.get_support_list()}."
        )

    # preprocess src to remove **
    report_text_ = report_text.replace("**", "")

    scrubbed_text = scrub(text=report_text_, scrubber=scrubber)
    dst_uri = os.path.join(output_path, file_name + ".txt")
    _write_to_file(
        uri=dst_uri,
        s=scrubbed_text,
        encoding="utf-8",
    )

    # Only generate report if report_context is not None
    if report_context:
        entities = detect(text=report_text_, scrubber=scrubber)
        file_summary = {
            ReportContextKey.INPUT_FILE_NAME: input_path,
            ReportContextKey.OUTPUT_NAME: dst_uri,
            ReportContextKey.TOTAL_TOKENS: len(entities),
            ReportContextKey.ENTITIES: _process_pos(entities, report_text_),
            ReportContextKey.FILE_NAME: file_name,
        }
        report_context.get(ReportContextKey.FILE_SUMMARY).get(subdirectory).append(
            file_summary
        )

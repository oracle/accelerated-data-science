#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


YAML_KEYS = [
    "detectors",
    "custom_detectors",
    "spacy_detectors",
    "anonymization",
    "name",
    "label",
    "patterns",
    "model",
    "named_entities",
    "entities",
]

################
# Report Const #
################
PII_REPORT_DESCRIPTION = (
    "This report will offer a comprehensive overview of the redaction of personal identifiable information (PII) from the provided data."
    "The `Summary` section will provide an executive summary of this process, including key statistics, configuration, and model usage."
    "The `Details` section will offer a more granular analysis of each row of data, including relevant statistics."
)
DETAILS_REPORT_DESCRIPTION = "The following report will show the details on each row. You can view the highlighted named entities and their labels in the text under `TEXT` tab."

FLAT_UI_COLORS = [
    "#1ABC9C",
    "#2ECC71",
    "#3498DB",
    "#9B59B6",
    "#34495E",
    "#16A085",
    "#27AE60",
    "#2980B9",
    "#8E44AD",
    "#2C3E50",
    "#F1C40F",
    "#E67E22",
    "#E74C3C",
    "#ECF0F1",
    "#95A5A6",
    "#F39C12",
    "#D35400",
    "#C0392B",
    "#BDC3C7",
    "#7F8C8D",
]

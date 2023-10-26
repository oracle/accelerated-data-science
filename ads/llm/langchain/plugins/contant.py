#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from ads.llm.langchain.plugins.base import StrEnum

DEFAULT_TIME_OUT = 300
DEFAULT_CONTENT_TYPE_JSON = "application/json"


class Task(StrEnum):
    TEXT_GENERATION = "text_generation"
    SUMMARY_TEXT = "summary_text"


class LengthParam(StrEnum):
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    AUTO = "AUTO"


class FormatParam(StrEnum):
    PARAGRAPH = "PARAGRAPH"
    BULLETS = "BULLETS"
    AUTO = "AUTO"


class ExtractivenessParam(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    AUTO = "AUTO"


class OCIGenerativeAIModel(StrEnum):
    COHERE_COMMAND = "cohere.command"
    COHERE_COMMAND_LIGHT = "cohere.command-light"

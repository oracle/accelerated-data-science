#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
from enum import Enum


class StrEnum(str, Enum):
    """Enum with string members
    https://docs.python.org/3.11/library/enum.html#enum.StrEnum
    """

    # Pydantic uses Python's standard enum classes to define choices.
    # https://docs.pydantic.dev/latest/api/standard_library_types/#enum


DEFAULT_TIME_OUT = 300
DEFAULT_CONTENT_TYPE_JSON = "application/json"


class Task(StrEnum):
    TEXT_GENERATION = "text_generation"
    TEXT_SUMMARIZATION = "text_summarization"


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

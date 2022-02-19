#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


class DatasetType:
    """DatasetType class which contains all the dataset
    types that data labeling service supports.
    """

    TEXT = "TEXT"
    IMAGE = "IMAGE"
    DOCUMENT = "DOCUMENT"


class AnnotationType:
    """AnnotationType class which contains all the annotation
    types that data labeling service supports.
    """

    SINGLE_LABEL = "SINGLE_LABEL"
    MULTI_LABEL = "MULTI_LABEL"
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    BOUNDING_BOX = "BOUNDING_BOX"


class Formats:
    """Common formats class which contains all the common
    formats that are supported to convert to.
    """

    SPACY = "spacy"
    YOLO = "yolo"


FORMATS_TO_ANNOTATION_TYPE = {
    Formats.SPACY: AnnotationType.ENTITY_EXTRACTION,
    Formats.YOLO: AnnotationType.BOUNDING_BOX,
}


ANNOTATION_TYPE_TO_FORMATS = {
    AnnotationType.ENTITY_EXTRACTION: Formats.SPACY,
    AnnotationType.BOUNDING_BOX: Formats.YOLO,
}

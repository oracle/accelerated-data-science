#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
from abc import abstractmethod
from typing import Dict, List, Union

from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.constants import AnnotationType, Formats
from ads.data_labeling.interface.parser import Parser
from ads.data_labeling.ner import NERItem
from ads.data_labeling.record import Record

DATASET_RECORD_INVALID_MESSAGE = "The dataset record file is invalid. "


class EntityType:
    """Entity type class for supporting multiple types of entities."""

    GENERIC = "GENERIC"
    TEXTSELECTION = "TEXTSELECTION"
    IMAGEOBJECTSELECTION = "IMAGEOBJECTSELECTION"


class RecordParser(Parser):
    """RecordParser class which parses the labels from the record.

    Examples
    --------
    >>> from ads.data_labeling.parser.export_record_parser import SingleLabelRecordParser
    >>> from ads.data_labeling.parser.export_record_parser import MultiLabelRecordParser
    >>> from ads.data_labeling.parser.export_record_parser import NERRecordParser
    >>> from ads.data_labeling.parser.export_record_parser import BoundingBoxRecordParser
    >>> import fsspec
    >>> import json
    >>> from ads.common import auth as authutil
    >>> labels = []
    >>> with fsspec.open("/path/to/records_file.jsonl", **authutil.api_keys()) as f:
    >>>     for line in f:
    >>>         bounding_box_labels = BoundingBoxRecordParser("source_data_path").parse(json.loads(line))
    >>>         labels.append(bounding_box_labels)
    """

    def __init__(
        self, dataset_source_path: str, format: str = None, categories: List[str] = None
    ) -> "RecordParser":
        """Initiates a RecordParser instance.

        Parameters
        ----------
        dataset_source_path: str
            Dataset source path.
        format: (str, optional). Defaults to None.
            Output format of annotations.
        categories: (List[str], optional). Defaults to None.
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        RecordParser
            RecordParser instance.
        """
        self.dataset_source_path = dataset_source_path
        self.format = format
        self.categories = categories

    def parse(self, record: Dict) -> "Record":
        """Extracts the annotations from the record content.
        Constructs and returns a Record instance containing the file path and the labels.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.

        Returns
        -------
        Record
            Record instance which contains the file path as well as the annotations.
        """
        return Record(
            path=self.dataset_source_path + record["sourceDetails"]["path"],
            annotation=self._extract_annotations(
                record=record, format=self.format, categories=self.categories
            ),
        )

    @abstractmethod
    def _extract_annotations(
        self,
        record: Dict,
        **kwargs,
    ) -> Union[str, List[str], List[BoundingBoxItem], List[NERItem]]:
        """Extracts annotations from the record content. Each Parser class
        needs to implement this function.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.
        kwargs: Dict
            format: str
                Output format of annotations. Can be "spacy" or "yolo".

        Returns
        -------
        Union[str, List[str], List[BoundingBoxItem], List[NERItem]]
            Label(s).
        """
        pass

    def _validate(self, record: Dict) -> None:
        """Validates the record to ensure it contains certain fields.

        Raises
        ------
        ValueError
            If record format is incorrect.
        """
        if (
            "annotations" not in record
            or not isinstance(record["annotations"], list)
            or "entities" not in record["annotations"][0]
            or not isinstance(record["annotations"][0]["entities"], list)
            or "entityType" not in record["annotations"][0]["entities"][0]
            or "labels" not in record["annotations"][0]["entities"][0]
        ):
            raise ValueError(
                f"{DATASET_RECORD_INVALID_MESSAGE}"
                "At least one record is in the wrong format. "
                "Use the `DataLabeling.export()` method to create a new dataset record file."
            )


class SingleLabelRecordParser(RecordParser):
    """SingleLabelRecordParser class which parses the label of Single label data."""

    def _extract_annotations(self, record: Dict, **kwargs) -> Union[str, None]:
        """Extract the labels of the single label annotation class.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.

        Returns
        -------
        Union[str, None]
            A label or None for the unlabeled record.
        """
        if "annotations" in record:
            self._validate(record)
            return record["annotations"][0]["entities"][0]["labels"][0]["label_name"]
        else:
            return None

    def _validate(self, record: Dict) -> None:
        """Validates the format of the single label record.

        Raises
        ------
        ValueError
            If record format is incorrect.
        """
        super()._validate(record)
        if record["annotations"][0]["entities"][0]["entityType"] != EntityType.GENERIC:
            raise ValueError(
                f"{DATASET_RECORD_INVALID_MESSAGE}"
                "At least one record contains the invalid entity type:  "
                f"`{record['annotations'][0]['entities'][0]['entityType']}`. The entity "
                f"type of the Single Label annotation must be `{EntityType.GENERIC}`. "
                "Use the `DataLabeling.export()` method to create a new dataset record file."
            )
        if len(record["annotations"][0]["entities"][0]["labels"]) != 1:
            raise ValueError(
                f"{DATASET_RECORD_INVALID_MESSAGE}"
                "At least one record contains an invalid number of records: "
                f"{len(record['annotations'][0]['entities'][0]['labels'])}. "
                "The Single Label annotation expects only one label for each record. "
                "Use the `DataLabeling.export()` method to create a new dataset record file."
            )


class MultiLabelRecordParser(RecordParser):
    """MultiLabelRecordParser class which parses the label of Multiple label data."""

    def _extract_annotations(self, record: Dict, **kwargs) -> Union[List[str], None]:
        """Extract labels of the Multi label annotation class.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.

        Returns
        -------
        Union[List[str], None]
            List of labels or None for the unlabeled record.
        """
        if "annotations" in record:
            self._validate(record)
            return [
                label["label_name"]
                for label in record["annotations"][0]["entities"][0]["labels"]
            ]
        else:
            return None

    def _validate(self, record: Dict) -> None:
        """Validates the format of the multi label record.

        Raises
        ------
        ValueError
            If record format is incorrect.
        """
        super()._validate(record)
        if record["annotations"][0]["entities"][0]["entityType"] != EntityType.GENERIC:
            raise ValueError(
                f"At least one of the dataset records contains the invalid entity type: "
                f"`{record['annotations'][0]['entities'][0]['entityType']}`. "
                f"The entity type of the Multi Label annotation must be `{EntityType.GENERIC}`."
            )
        if len(record["annotations"][0]["entities"][0]["labels"]) < 1:
            raise ValueError(
                f"At least one of the dataset records contains invalid number of labels: "
                f"`{len(record['annotations'][0]['entities'][0]['labels'])}`. "
                f"The Multi Label annotation expects at least one label for each record."
            )


class NERRecordParser(RecordParser):
    """NERRecordParser class which parses the label of NER label data."""

    def _extract_annotations(
        self, record: Dict, **kwargs
    ) -> Union[List[NERItem], None]:
        """Extracts the labels of the NER annotation class.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.
        kwargs: Dict
            format: str
                Output format of annotations. Can be "spacy" or None.
                When None, it outputs List[NERItem]. When "spacy", it
                outputs List[Tuple].

        Returns
        -------
        Union[List[NERItem], List[Tuple], None]
            The list of NERItem objects or list of tuples in spacy format.
        """
        if "annotations" in record:
            self._validate(record)
            format = kwargs.get("format", None)
            items = []
            for entity in record["annotations"][0]["entities"]:
                label = entity["labels"][0]["label_name"]
                offset = entity["textSpan"]["offset"]
                length = entity["textSpan"]["length"]
                item = NERItem(label=label, offset=offset, length=length)
                if (
                    format
                    and isinstance(format, str)
                    and format.lower() == Formats.SPACY
                ):
                    item = item.to_spacy()
                items.append(item)
            return items
        else:
            return None

    def _validate(self, record: Dict) -> None:
        """Validates the format of the NER label record.

        Raises
        ------
        ValueError
            If record format is incorrect.
        """
        super()._validate(record)
        if (
            record["annotations"][0]["entities"][0]["entityType"]
            != EntityType.TEXTSELECTION
        ):
            raise ValueError(
                f"{DATASET_RECORD_INVALID_MESSAGE}"
                "At least one record contains the invalid entity type:  "
                f"`{record['annotations'][0]['entities'][0]['entityType']}`. The entity type "
                f"of the Single Label annotation must be `{EntityType.TEXTSELECTION}`. "
                "Use the `DataLabeling.export()` method to create a new dataset record file."
            )
        if os.path.splitext(record["sourceDetails"]["path"])[1].lower() != ".txt":
            raise ValueError(
                f"The file ({record['sourceDetails']['path']}) must be a text file and have a '.txt' file extension."
            )


class BoundingBoxRecordParser(RecordParser):
    """BoundingBoxRecordParser class which parses the label of BoundingBox label data."""

    def _extract_annotations(
        self, record: Dict, **kwargs: Dict
    ) -> Union[List[BoundingBoxItem], None]:
        """Extracts the labels of the Object Detection annotation class.

        Parameters
        ----------
        record: Dict
            Content of the record from the record file.
        kwargs: Dict
            format: str
                Output format of annotations. Can be None or "yolo".
                When None, it outputs List[BoundingBoxItem]. When "yolo", it
                outputs List[List[Tuple]].
            categories: Optional List[str]
                 The list of object categories in proper order for model training.
                 Only used when bounding box annotations are in YOLO format.

        Returns
        -------
        Union[List[BoundingBoxItem], List[List[Tuple]], None]
            The list of BoundingBoxItem objects or list of tuples in YOLO format.
        """
        if not "annotations" in record:
            return None

        self._validate(record)
        format = kwargs.get("format", None)
        categories = kwargs.get("categories", None)
        items = []
        for entity in record["annotations"][0]["entities"]:
            labels = [label["label_name"] for label in entity["labels"]]
            coords = entity["boundingPolygon"]["normalizedVertices"]
            top_left = (float(coords[0]["x"]), float(coords[0]["y"]))
            bottom_left = (float(coords[1]["x"]), float(coords[1]["y"]))
            bottom_right = (float(coords[2]["x"]), float(coords[2]["y"]))
            top_right = (float(coords[3]["x"]), float(coords[3]["y"]))
            item = BoundingBoxItem(
                labels=labels,
                bottom_left=bottom_left,
                top_left=top_left,
                top_right=top_right,
                bottom_right=bottom_right,
            )
            if format and isinstance(format, str) and format.lower() == Formats.YOLO:
                item = item.to_yolo(categories=categories)
            items.append(item)

        return items

    def _validate(self, record: Dict) -> None:
        """Validates the format of the image label record.

        Raises
        ------
        ValueError
            If record format is incorrect.
        """
        super()._validate(record)
        if (
            record["annotations"][0]["entities"][0]["entityType"]
            != EntityType.IMAGEOBJECTSELECTION
        ):
            raise ValueError(
                f"{DATASET_RECORD_INVALID_MESSAGE}"
                "At least one record contains the invalid entity type:  "
                f"`{record['annotations'][0]['entities'][0]['entityType']}`. The entity type "
                f"of the Single Label annotation must be `{EntityType.IMAGEOBJECTSELECTION}`. "
                "Use the `DataLabeling.export()` method to create a new dataset record file."
            )
        if os.path.splitext(record["sourceDetails"]["path"])[1].lower() not in [
            ".jpg",
            ".png",
            ".jpeg",
        ]:
            raise ValueError(
                f"The file ({record['sourceDetails']['path']}) must be a jpg, jpeg or png file."
            )


class RecordParserFactory:
    """RecordParserFactory class which contains a list of registered parsers
    and allows to register new RecordParsers.

    Current parsers include:
        * SingleLabelRecordParser
        * MultiLabelRecordParser
        * NERRecordParser
        * BoundingBoxRecordParser
    """

    _parsers = {
        AnnotationType.SINGLE_LABEL: SingleLabelRecordParser,
        AnnotationType.MULTI_LABEL: MultiLabelRecordParser,
        AnnotationType.ENTITY_EXTRACTION: NERRecordParser,
        AnnotationType.BOUNDING_BOX: BoundingBoxRecordParser,
    }

    @staticmethod
    def parser(
        annotation_type: str,
        dataset_source_path: str,
        format: str = None,
        categories: List[str] = None,
    ) -> "RecordParser":
        """Gets the parser based on the annotation_type.

        Parameters
        ----------
        annotation_type: str
            Annotation type which can be SINGLE_LABEL, MULTI_LABEL, ENTITY_EXTRACTION
            and BOUNDING_BOX.
        dataset_source_path: str
            Dataset source path.
        format: (str, optional). Defaults to None.
            Output format of annotations. Can be None, "spacy" for dataset
            Entity Extraction type or "yolo" for Object Detection type.
            When None, it outputs List[NERItem] or List[BoundingBoxItem].
            When "spacy", it outputs List[Tuple].
            When "yolo", it outputs List[List[Tuple]].
        categories: (List[str], optional). Defaults to None.
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        RecordParser
            RecordParser corresponding to the annotation type.

        Raises
        ------
        ValueError
            If annotation_type is not supported.
        """

        if not annotation_type in RecordParserFactory._parsers:
            raise ValueError(
                f"The {annotation_type} is not supported. Choose from "
                f"`{AnnotationType.SINGLE_LABEL}`, `{AnnotationType.MULTI_LABEL}`, "
                f"`{AnnotationType.ENTITY_EXTRACTION}` and `{AnnotationType.BOUNDING_BOX}`."
            )

        return RecordParserFactory._parsers[annotation_type](
            dataset_source_path=dataset_source_path,
            format=format,
            categories=categories,
        )

    @classmethod
    def register(cls, annotation_type: str, parser) -> None:
        """Registers a new parser.

        Parameters
        ----------
        annotation_type: str
            Annotation type which can be SINGLE_LABEL, MULTI_LABEL, ENTITY_EXTRACTION
            and BOUNDING_BOX.
        parser: RecordParser
            A new Parser class to be registered.

        Returns
        -------
        None
            Nothing.
        """
        cls._parsers[annotation_type] = parser

#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
from typing import List, Union

from ads.common import auth as authutil
from ads.common import oci_client
from ads.data_labeling.boundingbox import BoundingBoxItem
from ads.data_labeling.constants import AnnotationType, Formats
from ads.data_labeling.interface.parser import Parser
from ads.data_labeling.ner import NERItem
from ads.data_labeling.reader.dls_record_reader import OCIRecordSummary
from ads.data_labeling.record import Record
from oci.data_labeling_service_dataplane.models import (
    AnnotationSummary as OCIAnnotationSummary,
)
from oci.data_labeling_service_dataplane.models.annotation import (
    Annotation as OCIAnnotation,
)
from oci.exceptions import ServiceError


class ReadAnnotationError(Exception):   # pragma: no cover
    def __init__(self, id: str):
        super().__init__(f"Error occurred in attempt to read annotation {id}.")


class AnnotationNotFoundError(Exception):   # pragma: no cover
    def __init__(self, id: str):
        super().__init__(f"{id} not found.")


class DLSRecordParser(Parser):   # pragma: no cover
    """DLSRecordParser class which parses the labels from the record."""

    def __init__(
        self,
        dataset_source_path: str,
        auth: dict = None,
        format: str = None,
        categories: List[str] = None,
    ) -> "DLSRecordParser":
        """Initiates a DLSRecordParser instance.

        Parameters
        ----------
        dataset_source_path: str
            Dataset source path.
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.
        format: (str, optional). Defaults to None.
            Output format of annotations.
        categories: (List[str], optional). Defaults to None.
            The list of object categories in proper order for model training.
            Example: ['cat','dog','horse']

        Returns
        -------
        DLSRecordParser
            DLSRecordParser instance.
        """
        auth = auth or authutil.default_signer()
        self.dataset_source_path = dataset_source_path
        self.format = format
        self.categories = categories
        self.dls_dp_client = oci_client.OCIClientFactory(**auth).data_labeling_dp

    def _get_annotation_details(
        self, oci_annotation_summary: List[OCIAnnotationSummary]
    ) -> List[OCIAnnotation]:
        """Get list of annotation details.

        Parameters
        ----------
        oci_annotation_summary: List[OCIAnnotationSummary]
            The list of OCIAnnotationSummary objects.

        Returns
        -------
        List[OCIAnnotation]
            List of OCI annotations.
        """
        result = []
        if oci_annotation_summary:
            for annotation in oci_annotation_summary:
                try:
                    annotation_details = self.dls_dp_client.get_annotation(
                        annotation.id
                    ).data
                except ServiceError as service_err:
                    if service_err.status == 404:
                        raise AnnotationNotFoundError(annotation.id)
                    raise ReadAnnotationError(annotation.id)
                result.append(annotation_details)
        return result

    def parse(self, oci_record_summary: OCIRecordSummary) -> Record:
        """Extracts the annotations.
        Constructs and returns a Record instance which contains the file path and the labels.

        Parameters
        ----------
        oci_record_summary: OCIRecordSummary
            The summary information about the record.

        Returns
        -------
        Record
            Record instance which contains the file path and the annotations.
        """
        return Record(
            path=self.dataset_source_path + oci_record_summary.record.name,
            annotation=self._extract_annotations(
                self._get_annotation_details(oci_record_summary.annotation),
                format=self.format,
                categories=self.categories,
            ),
        )

    @abstractmethod
    def _extract_annotations(
        self, oci_annotation: List[OCIAnnotation], **kwargs
    ) -> Union[str, List[str], List[BoundingBoxItem], List[NERItem]]:
        """Extracts annotations from the record content. Each Parser class
        needs to implement this function.

        Parameters
        ----------
        oci_annotation: List[OCIAnnotation]
            List of OCI annotations.
        kwargs: Dict
            format: str
                Output format of annotations. Can be "spacy" or "yolo".

        Returns
        -------
        Union[str, List[str], List[BoundingBoxItem], List[NERItem]]
            Label(s).
        """
        pass


class DLSSingleLabelRecordParser(DLSRecordParser):
    """SingleLabelRecordParser class which parses the label of Single label data."""

    def _extract_annotations(
        self, oci_annotation: List[OCIAnnotation], **kwargs
    ) -> Union[str, None]:
        """Extract the labels of the single label annotation class.

        Parameters
        ----------
        oci_annotation: List[OCIAnnotation]
            List of OCI annotations.

        Returns
        -------
        Union[str, None]
            A label or None for the unlabeled record.
        """
        if oci_annotation:
            return oci_annotation[0].entities[0].labels[0].label
        else:
            return None


class DLSMultiLabelRecordParser(DLSRecordParser):
    """MultiLabelRecordParser class which parses the label of Multiple label data."""

    def _extract_annotations(
        self, oci_annotation: List[OCIAnnotation], **kwargs
    ) -> Union[List[str], None]:
        """Extract labels of the Multi label annotation class.

        Parameters
        ----------
        oci_annotation: List[OCIAnnotation]
            List of OCI annotations.

        Returns
        -------
        Union[List[str], None]
            List of labels or None for the unlabeled record.
        """
        if oci_annotation:
            return [
                annotation_item.label
                for annotation_item in oci_annotation[0].entities[0].labels
            ]
        else:
            return None


class DLSNERRecordParser(DLSRecordParser):
    """NERRecordParser class which parses the label of NER label data."""

    def _extract_annotations(
        self, oci_annotation: List[OCIAnnotation], **kwargs
    ) -> Union[List[NERItem], None]:
        """Extracts the labels of the NER annotation class.

        Parameters
        ----------
        oci_annotation: List[OCIAnnotation]
            List of OCI annotations.
        kwargs: Dict
            format: str
                Output format of annotations. Can be "spacy" or None.
                When None, it outputs List[NERItem]. When "spacy", it
                outputs List[Tuple].

        Returns
        -------
        Union[List[NERItem], None]
            The list of NERItem objects.
        """
        items = []
        format = kwargs.get("format", None)
        if oci_annotation:
            for e in oci_annotation[0].entities:
                label = e.labels[0].label
                offset = e.text_span.offset
                length = e.text_span.length
                item = NERItem(label=label, offset=int(offset), length=int(length))
                if (
                    format
                    and isinstance(format, str)
                    and format.lower() == Formats.SPACY
                ):
                    item = item.to_spacy()
                items.append(item)
        return items


class DLSBoundingBoxRecordParser(DLSRecordParser):
    """BoundingBoxRecordParser class which parses the label of BoundingBox label data."""

    def _extract_annotations(
        self, oci_annotation: List[OCIAnnotation], **kwargs
    ) -> Union[List[BoundingBoxItem], None]:
        """Extracts the labels of the Object Detection annotation class.

        Parameters
        ----------
        oci_annotation: List[OCIAnnotation]
            List of OCI annotations.
        kwargs: Dict
            format: str
                Output format of annotations. Can be None or "yolo".
                When None, it outputs List[BoundingBoxItem]. When "yolo", it
                outputs List[List[Tuple]].
            categories: (List[str], optional).
                 The list of object categories in proper order for model training.
                 Only used when bounding box annotations are in YOLO format.

        Returns
        -------
        Union[List[BoundingBoxItem], None]
            The list of BoundingBoxItem objects.
        """
        format = kwargs.get("format", None)
        categories = kwargs.get("categories", None)

        items = []
        if oci_annotation:
            for e in oci_annotation[0].entities:
                labels = [l.label for l in e.labels]
                coords = e.bounding_polygon.normalized_vertices
                top_left = (float(coords[0].x), float(coords[0].y))
                bottom_left = (float(coords[1].x), float(coords[1].y))
                bottom_right = (float(coords[2].x), float(coords[2].y))
                top_right = (float(coords[3].x), float(coords[3].y))
                item = BoundingBoxItem(
                    labels=labels,
                    bottom_left=bottom_left,
                    top_left=top_left,
                    top_right=top_right,
                    bottom_right=bottom_right,
                )
                if (
                    format
                    and isinstance(format, str)
                    and format.lower() == Formats.YOLO
                ):
                    item = item.to_yolo(categories=categories)
                items.append(item)

        return items


class DLSRecordParserFactory:
    """DLSRecordParserFactory class which contains a list of registered parsers
    and allows to register new DLSRecordParsers.

    Current parsers include:
        * DLSSingleLabelRecordParser
        * DLSMultiLabelRecordParser
        * DLSNERRecordParser
        * DLSBoundingBoxRecordParser
    """

    _parsers = {
        AnnotationType.SINGLE_LABEL: DLSSingleLabelRecordParser,
        AnnotationType.MULTI_LABEL: DLSMultiLabelRecordParser,
        AnnotationType.ENTITY_EXTRACTION: DLSNERRecordParser,
        AnnotationType.BOUNDING_BOX: DLSBoundingBoxRecordParser,
    }

    @staticmethod
    def parser(
        annotation_type: str,
        dataset_source_path: str,
        format: str = None,
        categories: List[str] = None,
        auth: dict = None,
    ) -> "DLSRecordParser":
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
        auth: (dict, optional). Defaults to None.
            The default authetication is set using `ads.set_auth` API. If you need to override the
            default, use the `ads.common.auth.api_keys` or `ads.common.auth.resource_principal` to create appropriate
            authentication signer and kwargs required to instantiate IdentityClient object.

        Returns
        -------
        DLSRecordParser
            DLSRecordParser corresponding to the annotation type.

        Raises
        ------
        ValueError
            If annotation_type is not supported.
        """

        if not annotation_type in DLSRecordParserFactory._parsers:
            raise ValueError(
                f"The {annotation_type} is not supported. Choose from "
                f"`{AnnotationType.SINGLE_LABEL}`, `{AnnotationType.MULTI_LABEL}`, "
                f"`{AnnotationType.ENTITY_EXTRACTION}` and `{AnnotationType.BOUNDING_BOX}`."
            )

        return DLSRecordParserFactory._parsers[annotation_type](
            dataset_source_path=dataset_source_path,
            format=format,
            categories=categories,
            auth=auth or authutil.default_signer(),
        )

    @classmethod
    def register(cls, annotation_type: str, parser: DLSRecordParser) -> None:
        """Registers a new parser.

        Parameters
        ----------
        annotation_type: str
            Annotation type which can be SINGLE_LABEL, MULTI_LABEL, ENTITY_EXTRACTION
            and BOUNDING_BOX.
        parser: DLSRecordParser
            A new Parser class to be registered.

        Returns
        -------
        None
            Nothing.
        """
        cls._parsers[annotation_type] = parser

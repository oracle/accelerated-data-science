#!/usr/bin/env python
# -*- coding: utf-8; -*-

# Copyright (c) 2021, 2022 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import Any, Dict

from ads.common.object_storage_details import ObjectStorageDetails
from ads.data_labeling.interface.parser import Parser
from ads.data_labeling.metadata import Metadata


class MetadataParser(Parser):
    """MetadataParser class which parses the metadata from the record."""

    EXPECTED_KEYS = [
        "id",
        "compartmentId",
        "displayName",
        "labelsSet",
        "annotationFormat",
        "datasetSourceDetails",
        "datasetFormatDetails",
    ]

    @staticmethod
    def parse(json_data: Dict[Any, Any]) -> "Metadata":
        """Parses the metadata jsonl file.

        Parameters
        ----------
        json_data: dict
            dictionary format of the metadata jsonl file content.

        Returns
        -------
        Metadata
            Metadata object which contains the useful fields from the metadata jsonl file
        """
        MetadataParser._validate(json_data)
        source_path = ObjectStorageDetails(
            json_data["datasetSourceDetails"]["bucket"],
            json_data["datasetSourceDetails"]["namespace"],
            json_data["datasetSourceDetails"]["prefix"],
        ).path

        records_path = ""
        if "recordFiles" in json_data:
            records_path = ObjectStorageDetails(
                json_data["recordFiles"][0]["bucket"],
                json_data["recordFiles"][0]["namespace"],
                json_data["recordFiles"][0]["path"],
            ).path

        return Metadata(
            source_path=source_path,
            records_path=records_path,
            labels=[clss["name"] for clss in json_data["labelsSet"]],
            dataset_name=json_data["displayName"],
            compartment_id=json_data["compartmentId"],
            dataset_id=json_data["id"],
            annotation_type=json_data["annotationFormat"],
            dataset_type=json_data["datasetFormatDetails"]["formatType"],
        )

    @staticmethod
    def _validate(json_data: Dict[Any, Any]) -> None:
        """Validates the metadata jsonl file to ensure it contains certain fields.

        Parameters
        ----------
        json_data: dict
            dictionary format of the metadata jsonl file content.
        """

        def invalid_message(param):
            return (
                f"The dataset metadata file is invalid. The field '{param}' is required but it is missing. "
                + "Update the metadata file or use the `DataLabeling.export()` method "
                + "to create a new dataset metadata file."
            )

        for k in MetadataParser.EXPECTED_KEYS:
            if k not in json_data:
                raise ValueError(f"{invalid_message(k)}")
        expected_list_format = ["labelsSet", "recordFiles"]
        for k in expected_list_format:
            if k in json_data and not isinstance(json_data[k], list):
                raise ValueError(f"{invalid_message(k)}")
        expected_dict_format = ["datasetSourceDetails", "datasetFormatDetails"]
        for k in expected_dict_format:
            if not isinstance(json_data[k], dict):
                raise ValueError(f"{invalid_message(k)}")

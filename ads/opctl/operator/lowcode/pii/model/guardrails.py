#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import time
from datetime import datetime

from ads.common.object_storage_details import ObjectStorageDetails
from ads.opctl import logger
from ads.opctl.operator.lowcode.pii.constant import DataFrameColumn
from ads.opctl.operator.lowcode.pii.model.pii import PiiScrubber, detect, scrub
from ads.opctl.operator.lowcode.pii.model.report import (
    PIIOperatorReport,
    PiiReportPageSpec,
    PiiReportSpec,
)
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig
from ads.opctl.operator.lowcode.pii.utils import (
    default_signer,
    get_output_name,
)
from ads.opctl.operator.lowcode.common.utils import load_data, write_data
from ads.opctl.operator.lowcode.common.errors import InvalidParameterError


class PIIGuardrail:
    def __init__(self, config: PiiOperatorConfig):
        self.config = config
        self.spec = config.spec
        self.pii_scrubber = PiiScrubber(config=config)
        self.scrubber = self.pii_scrubber.config_scrubber()

        output_filename = get_output_name(
            target_name=self.spec.output_directory.name,
            given_name=self.spec.input_data.url,
        )
        self.dst_uri = os.path.join(self.spec.output_directory.url, output_filename)
        self.config.spec.output_directory.name = output_filename

        self.report_uri = os.path.join(
            self.spec.output_directory.url,
            self.spec.report.report_filename,
        )

        self.report_context: PiiReportSpec = PiiReportSpec.from_dict(
            {
                "run_summary": {
                    "config": self.config,
                    "selected_detectors": self.pii_scrubber.detectors,
                    "selected_entities": self.pii_scrubber.entities,
                    "selected_spacy_model": self.pii_scrubber.spacy_model_detectors,
                    "show_rows": self.spec.report.show_rows,
                    "show_sensitive_info": self.spec.report.show_sensitive_content,
                    "src_uri": self.spec.input_data.url,
                    "total_tokens": 0,
                },
                "run_details": {"rows": []},
            }
        )

        self.storage_options = (
            default_signer()
            if ObjectStorageDetails.is_oci_path(self.spec.output_directory.url)
            else {}
        )
        self.datasets = None

    def _load_data(self, uri=None, storage_options=None):
        """Loads input data."""
        input_data_uri = uri or self.spec.input_data.url
        logger.info(f"Loading input data from `{input_data_uri}` ...")

        try:
            self.datasets = load_data(
                data_spec=self.spec.input_data,
                storage_options=storage_options or self.storage_options,
            )
        except InvalidParameterError as e:
            e.args = e.args + ("Invalid Parameter: input_data",)
            raise e

        return self

    def process(self, **kwargs):
        """Process input data."""
        self.report_context.run_summary.timestamp = datetime.now().strftime(
            "%d/%m/%Y %H:%M:%S"
        )
        start_time = time.time()

        data = kwargs.pop("input_data", None) or self.datasets
        report_uri = kwargs.pop("report_uri", None) or self.report_uri
        dst_uri = kwargs.pop("dst_uri", None) or self.dst_uri

        if not data:
            try:
                self._load_data()
                data = self.datasets
            except InvalidParameterError as e:
                e.args = e.args + ("Invalid Parameter: input_data",)
                raise e

        # process user data
        data[DataFrameColumn.REDACTED_TEXT] = data[self.spec.target_column].apply(
            lambda x: scrub(x, scrubber=self.scrubber)
        )
        self.report_context.run_summary.elapsed_time = time.time() - start_time
        self.report_context.run_summary.total_rows = len(data.index)

        # save output data
        if dst_uri:
            logger.info(f"Saving data into `{dst_uri}` ...")
            write_data(
                data=data.loc[:, data.columns != self.spec.target_column],
                filename=dst_uri,
                format=None,
                storage_options=kwargs.pop("storage_options", None)
                or self.storage_options,
            )

        # prepare pii report
        if report_uri:
            logger.info(f"Generating report to `{report_uri}` ...")

            data[DataFrameColumn.ENTITIES] = data[self.spec.target_column].apply(
                lambda x: detect(text=x, scrubber=self.scrubber)
            )

            for i in data.index:
                text = data[self.spec.target_column][i]
                ent_col = data[DataFrameColumn.ENTITIES][i]
                page = PiiReportPageSpec.from_dict(
                    {
                        "id": i,
                        "total_tokens": len(ent_col),
                        "entities": ent_col,
                        "raw_text": text,
                    }
                )
                self.report_context.run_details.rows.append(page)
                self.report_context.run_summary.total_tokens += len(ent_col)

            self._process_context()
            PIIOperatorReport(
                report_spec=self.report_context, report_uri=report_uri
            ).make_view().save_report(
                storage_options=kwargs.pop("storage_options", None)
                or self.storage_options
            )

    def _process_context(self):
        """Count different type of filth."""
        statics = {}  # statics : count Filth type in total
        rows = self.report_context.run_details.rows
        for row in rows:
            entities = row.entities
            row_statics = {}  # count row
            for ent in entities:
                row_statics[ent.type] = row_statics.get(ent.type, 0) + 1
                statics[ent.type] = statics.get(ent.type, 0) + 1

            row.statics = row_statics.copy()

        self.report_context.run_summary.statics = statics

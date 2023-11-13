#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import os
import time
from ads.opctl import logger
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig
from ads.opctl.operator.lowcode.pii.model.pii import Scrubber, scrub, detect
from ads.opctl.operator.lowcode.pii.model.report import PIIOperatorReport

from datetime import datetime
from ads.opctl.operator.lowcode.pii.model.utils import (
    _load_data,
    _write_data,
    get_output_name,
)
from ads.opctl.operator.lowcode.pii.model.utils import default_signer
from ads.common.object_storage_details import ObjectStorageDetails


class PIIGuardrail:
    def __init__(self, config: PiiOperatorConfig, auth: dict = None):
        self.spec = config.spec
        self.scrubber = Scrubber(config=config).config_scrubber()

        self.dst_uri = os.path.join(
            self.spec.output_directory.url,
            get_output_name(
                target_name=self.spec.output_directory.name,
                given_name=self.self.spec.input_data.url,
            ),
        )

        self.report_uri = os.path.join(
            self.spec.output_directory.url,
            self.spec.report.report_filename,
        )

        try:
            self.datasets = self.load_data()
        except Exception as e:
            logger.warning(f"Failed to load data from `{self.spec.input_data.url}`.")
            logger.debug(f"Full traceback: {e}")

    def load_data(self, uri=None, storage_options=None):
        """Loads input data."""
        input_data_uri = uri or self.spec.input_data.url
        logger.info(f"Loading input data from `{input_data_uri}` ...")

        self.datasets = _load_data(
            filename=input_data_uri,
            storage_options=storage_options or default_signer(),
        )

    def process(self, **kwargs):
        """Process input data."""
        run_at = datetime.now()
        dt_string = run_at.strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()

        data = kwargs.pop("input_data", None) or self.datasets
        report_uri = kwargs.pop("report_uri", None) or self.report_uri
        dst_uri = kwargs.pop("dst_uri", None) or self.dst_uri

        # process user data
        data["redacted_text"] = data[self.spec.target_column].apply(
            lambda x: scrub(x, scrubber=self.scrubber)
        )
        elapsed_time = time.time() - start_time

        if dst_uri:
            logger.info(f"Saving data into `{dst_uri}` ...")

            _write_data(
                data=data.loc[:, data.columns != self.spec.target_column],
                filename=dst_uri,
                storage_options=default_signer()
                if ObjectStorageDetails.is_oci_path(dst_uri)
                else {},
            )

        # prepare pii report
        if report_uri:
            data["entities_cols"] = data[self.spec.target_column].apply(
                lambda x: detect(text=x, scrubber=self.scrubber)
            )
            from ads.opctl.operator.lowcode.pii.model.utils import _safe_get_spec
            from ads.opctl.operator.lowcode.pii.model.pii import DEFAULT_SPACY_MODEL

            selected_spacy_model = []
            for spec in _safe_get_spec(
                self.scrubber.redact_spec_file, "spacy_detectors", []
            ):
                selected_spacy_model.append(
                    {
                        "model": _safe_get_spec(spec, "model", DEFAULT_SPACY_MODEL),
                        "spacy_entites": [
                            x.upper() for x in spec.get("named_entities", [])
                        ],
                    }
                )
            selected_entities = []
            for spacy_models in selected_spacy_model:
                selected_entities = selected_entities + spacy_models.get(
                    "spacy_entites", []
                )
            selected_entities = selected_entities + _safe_get_spec(
                self.scrubber.redact_spec_file, "detectors", []
            )

            context = {
                "run_summary": {
                    "total_tokens": 0,
                    "src_uri": self.spec.input_data.url,
                    "total_rows": len(data.index),
                    "config": self.spec,
                    "selected_detectors": list(self.scrubber._detectors.values()),
                    "selected_entities": selected_entities,
                    "selected_spacy_model": selected_spacy_model,
                    "timestamp": dt_string,
                    "elapsed_time": elapsed_time,
                    "show_rows": self.spec.report.show_rows,
                    "show_sensitive_info": self.spec.report.show_sensitive_content,
                },
                "run_details": {"rows": []},
            }
            for ind in data.index:
                text = data[self.spec.target_column][ind]
                ent_col = data["entities_cols"][ind]
                idx = data["id"][ind]
                page = {
                    "id": idx,
                    "total_tokens": len(ent_col),
                    "entities": ent_col,
                    "raw_text": text,
                }
                context.get("run_details").get("rows").append(page)
                context.get("run_summary")["total_tokens"] += len(ent_col)

            context = self._process_context(context)
            self._generate_report(context, report_uri)

    def _generate_report(self, context, report_uri):
        report_ = PIIOperatorReport(context=context)
        report_sections = report_.make_view()
        report_.save_report(report_sections=report_sections, report_path=report_uri)

    def _process_context(self, context):
        """Count different type of filth."""
        statics = {}  # statics : count Filth type in total
        rows = context.get("run_details").get("rows")
        for row in rows:
            entities = row.get("entities")
            row_statics = {}  # count row
            for ent in entities:
                row_statics[ent.type] = row_statics.get(ent.type, 0) + 1
                statics[ent.type] = statics.get(ent.type, 0) + 1

            row["statics"] = row_statics.copy()

        context.get("run_summary")["statics"] = statics
        return context

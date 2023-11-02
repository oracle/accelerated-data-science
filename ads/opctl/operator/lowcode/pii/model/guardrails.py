#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import pandas as pd
from ads.opctl.operator.lowcode.pii.model.utils import from_yaml
from ads.opctl.operator.lowcode.pii.model.pii import config_scrubber, scrub, detect
from ads.opctl.operator.lowcode.pii.model.report import PIIOperatorReport
from ads.common import auth as authutil
from datetime import datetime
import os
import time
import datapane as dp


def get_output_name(given_name, target_name=None):
    """Add ``-out`` suffix to the src filename."""
    if not target_name:
        basename = os.path.basename(given_name)
        fn, ext = os.path.splitext(basename)
        target_name = fn + "_out" + ext
    return target_name


class PIIGuardrail:
    def __init__(self, config_uri: str, auth: dict = None):
        # load config.yaml for pii
        self.spec = from_yaml(uri=config_uri).get("spec")
        self.output_data_name = None
        # config metric
        for metric in self.spec.get("metrics", []):
            # TODO: load other metric
            # load pii metric
            if metric.get("name", "") == "pii":
                pii_load_args = metric.get("load_args")
                self.scrubber = config_scrubber(**pii_load_args)
                self.target_col = metric.get("target_col", "text")
                self.output_data_name = metric.get("output_data_name", None)

        # config spec
        self.src_data_uri = self.spec.get("test_data").get("url")
        self.dst_uri = None
        self.data = None
        self.report_uri = None
        self.auth = auth or authutil.default_signer()
        self.output_directory = self.spec.get("output_directory", {}).get("url", None)
        if self.output_directory:
            self.dst_uri = os.path.join(
                self.output_directory,
                get_output_name(
                    target_name=self.output_data_name, given_name=self.src_data_uri
                ),
            )

        self.report_spec = self.spec.get("report", {})
        self.report_uri = (
            os.path.join(
                self.report_spec.get("url", "./"),
                self.report_spec.get("report_file_name", "report.html"),
            )
            if self.report_spec
            else None
        )
        self.show_rows = self.report_spec.get("show_rows", 25)
        self.show_sensitive_content = self.report_spec.get(
            "show_sensitive_content", False
        )

    def load_data(self, uri=None, storage_options={}):
        # POC: Only csv support
        # csv -> pandas.DataFrame
        uri = uri or self.src_data_uri
        if uri.endswith(".csv"):
            if uri.startswith("oci://"):
                storage_options = storage_options or self.auth
                self.data = pd.read_csv(uri, storage_options=storage_options)
            else:
                self.data = pd.read_csv(uri)
        return self.data

    def evaluate(self, data=None, dst_uri=None, report_uri=None, storage_options={}):
        run_at = datetime.now()
        dt_string = run_at.strftime("%d/%m/%Y %H:%M:%S")
        start_time = time.time()
        data = data or self.data
        if data is None:
            data = self.load_data(storage_options)

        report_uri = report_uri or self.report_uri
        dst_uri = dst_uri or self.dst_uri

        data["redacted_text"] = data[self.target_col].apply(
            lambda x: scrub(x, scrubber=self.scrubber)
        )
        elapsed_time = time.time() - start_time
        # generate pii report
        if report_uri:
            data["entities_cols"] = data[self.target_col].apply(
                lambda x: detect(text=x, scrubber=self.scrubber)
            )
            from pii.utils import _safe_get_spec
            from pii.pii import DEFAULT_SPACY_MODEL

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
                    "src_uri": self.src_data_uri,
                    "total_rows": len(data.index),
                    "config": self.spec,
                    "selected_detectors": list(self.scrubber._detectors.values()),
                    "selected_entities": selected_entities,
                    "selected_spacy_model": selected_spacy_model,
                    "timestamp": dt_string,
                    "elapsed_time": elapsed_time,
                    "show_rows": self.show_rows,
                    "show_sensitive_info": self.show_sensitive_content,
                },
                "run_details": {"rows": []},
            }
            for ind in data.index:
                text = data[self.target_col][ind]
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

        if dst_uri:
            self._save_output(data, ["id", "redacted_text"], dst_uri)

        print("Mission completed!")

    def _generate_report(self, context, report_uri):
        report_ = PIIOperatorReport(context=context)
        report_sections = report_.make_view()
        report_.save_report(report_sections=report_sections, report_path=report_uri)

    def _save_output(self, df, target_col, dst_uri):
        # Based on extension of dst_uri call to_csv or to_json.
        data_out = df[target_col]
        data_out.to_csv(dst_uri)
        return dst_uri

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

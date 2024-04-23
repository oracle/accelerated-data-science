#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import os
import random
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

import fsspec
import pandas as pd
import requests
import yaml

from ads.common.decorator.runtime_dependency import (
    OptionalDependency,
    runtime_dependency,
)
from ads.common.serializer import DataClassSerializable
from ads.opctl import logger
from ads.opctl.operator.lowcode.common.utils import (
    disable_print,
    enable_print,
    human_time_friendly,
)
from ads.opctl.operator.lowcode.pii.constant import (
    DEFAULT_COLOR,
    DEFAULT_SHOW_ROWS,
    DEFAULT_TIME_OUT,
    DETAILS_REPORT_DESCRIPTION,
    FLAT_UI_COLORS,
    PII_REPORT_DESCRIPTION,
)
from ads.opctl.operator.lowcode.pii.operator_config import PiiOperatorConfig
from ads.opctl.operator.lowcode.pii.utils import compute_rate

try:
    import report_creator as rc
except ImportError:
    raise ModuleNotFoundError(
        f"`report-creator` module was not found. Please run "
        f"`pip install {OptionalDependency.PII}`."
    )


@dataclass(repr=True)
class PiiReportPageSpec(DataClassSerializable):
    """Class representing each page under Run Details in pii operator report."""

    entities: list = field(default_factory=list)
    id: int = None
    raw_text: str = None
    statics: dict = field(default_factory=dict)
    total_tokens: int = None


@dataclass(repr=True)
class RunDetails(DataClassSerializable):
    """Class representing Run Details Page in pii operator report."""

    rows: list = field(default_factory=list)


@dataclass(repr=True)
class RunSummary(DataClassSerializable):
    """Class representing Run Summary Page in pii operator report."""

    config: PiiOperatorConfig = None
    elapsed_time: str = None
    selected_detectors: list = field(default_factory=list)
    selected_entities: List[str] = field(default_factory=list)
    selected_spacy_model: List[Dict] = field(default_factory=list)
    show_rows: int = None
    show_sensitive_info: bool = False
    src_uri: str = None
    statics: dict = None
    timestamp: str = None
    total_rows: int = None
    total_tokens: int = None


@dataclass(repr=True)
class PiiReportSpec(DataClassSerializable):
    """Class representing pii operator report."""

    run_details: RunDetails = field(default_factory=RunDetails)
    run_summary: RunSummary = field(default_factory=RunSummary)


LABEL_TO_COLOR_MAP = {}


@runtime_dependency(module="plotly", install_from=OptionalDependency.PII)
def make_model_card(model_name="", readme_path=""):
    """Make render model_readme.md as model_card tab.
    All spacy model: https://huggingface.co/spacy
    For example: "en_core_web_trf": "https://huggingface.co/spacy/en_core_web_trf/raw/main/README.md".
    """

    readme_path = (
        f"https://huggingface.co/spacy/{model_name}/raw/main/README.md"
        if model_name
        else readme_path
    )
    if not readme_path:
        raise NotImplementedError("Does not support other spacy model so far.")

    try:
        requests.get(readme_path, timeout=DEFAULT_TIME_OUT)
        with fsspec.open(readme_path, "r") as file:
            content = file.read()
            _, front_matter, text = content.split("---", 2)
            data = yaml.safe_load(front_matter)
    except requests.ConnectionError:
        logger.warning(
            "You don't have internet connection. Therefore, we are not able to generate model card."
        )
        return rc.Group(
            rc.Text("-"),
            columns=1,
        )

    try:
        import plotly.graph_objects as go

        eval_res = data["model-index"][0]["results"]
        metrics = []
        values = []
        for eval in eval_res:
            metric = [x["name"] for x in eval["metrics"]]
            value = [x["value"] for x in eval["metrics"]]
            metrics = metrics + metric
            values = values + value
        df = pd.DataFrame({"Metrics": metrics, "Values": values})
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(df.columns)),
                    cells=dict(values=[df.Metrics, df.Values]),
                )
            ]
        )
        eval_res_tb = rc.Widget(data=fig, caption="Evaluation Results")
    except:
        eval_res_tb = rc.Text("-")
        logger.warning(
            "The given readme.md doesn't have correct template for Evaluation Results."
        )

    return rc.Group(
        rc.Text(text),
        eval_res_tb,
        columns=2,
    )


def map_label_to_color(labels):
    """Pair label with corresponding color."""
    label_to_colors = {}
    for label in labels:
        label = label.lower()
        label_to_colors[label] = LABEL_TO_COLOR_MAP.get(
            label, random.choice(FLAT_UI_COLORS)
        )
        LABEL_TO_COLOR_MAP[label] = label_to_colors[label]

    return label_to_colors


@runtime_dependency(module="plotly", install_from=OptionalDependency.PII)
def plot_pie(count_map) -> rc.Widget:
    import plotly.express as px

    cols = count_map.keys()
    cnts = count_map.values()
    ent_col_name = "EntityName"
    cnt_col_name = "count"
    df = pd.DataFrame({ent_col_name: cols, cnt_col_name: cnts})

    fig = px.pie(
        df,
        values=cnt_col_name,
        names=ent_col_name,
        title="The Distribution Of Entities Redacted",
        color=ent_col_name,
        color_discrete_map=map_label_to_color(cols),
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return rc.Widget(fig)


def build_entity_df(entites, id) -> pd.DataFrame:
    text = [ent.text for ent in entites]
    types = [ent.type for ent in entites]
    replaced_values = [
        ent.replacement_string or "{{" + ent.placeholder + "}}" for ent in entites
    ]
    d = {
        "Row ID": id,
        "Entity (Original Text)": text,
        "Type": types,
        "Redacted To": replaced_values,
    }
    df = pd.DataFrame(data=d)
    if df.size == 0:
        # Datapane does not support empty dataframe, append a dummy row
        df2 = {
            "Row ID": id,
            "Entity (Original Text)": "-",
            "Type": "-",
            "Redacted To": "-",
        }
        df = df.append(df2, ignore_index=True)
    return df


class RowReportFields:
    def __init__(self, row_spec: PiiReportPageSpec, show_sensitive_info: bool = True):
        self.spec = row_spec
        self.show_sensitive_info = show_sensitive_info

    def build_report(self) -> rc.Group:
        return rc.Group(
            rc.Select(
                blocks=[
                    self._make_stats_card(),
                    self._make_text_card(),
                ],
                type=rc.SelectType.TABS,
            ),
            label="Row Id: " + str(self.spec.id),
        )

    def _make_stats_card(self):
        stats = [
            rc.Heading("Row Summary Statistics", level=2),
            rc.Metric(
                heading="Total No. Of Entites Proceed",
                value=self.spec.total_tokens or 0,
            ),
            rc.Heading("Entities Distribution", level=3),
            plot_pie(self.spec.statics),
        ]
        if self.show_sensitive_info:
            stats.append(rc.Heading("Resolved Entities", level=3))
            stats.append(
                rc.DataTable(
                    build_entity_df(self.spec.entities, id=self.spec.id),
                    label="Resolved Entities",
                    index=True,
                )
            )
        return rc.Group(stats, label="STATS")

    def _make_text_card(self):
        annotations = []
        labels = set()
        for ent in self.spec.entities:
            annotations.append((ent.beg, ent.end, ent.type))
            labels.add(ent.type)

        if len(annotations) == 0:
            annotations.append((0, 0, "No entity detected"))

        d = {"Content": [self.spec.raw_text], "Annotations": [annotations]}
        df = pd.DataFrame(data=d)
        render_html = df.ads.render_ner(
            options={
                "default_color": DEFAULT_COLOR,
                "colors": map_label_to_color(labels),
            },
            return_html=True,
        )
        return rc.Group(rc.HTML(render_html), label="TEXT")


class PIIOperatorReport:
    def __init__(self, report_spec: PiiReportSpec, report_uri: str):
        # set useful field for generating report from context
        self.report_spec = report_spec
        self.show_rows = report_spec.run_summary.show_rows or DEFAULT_SHOW_ROWS

        rows = report_spec.run_details.rows
        rows = rows[0 : self.show_rows]
        self.rows_details = [
            RowReportFields(r, report_spec.run_summary.show_sensitive_info)
            for r in rows
        ]

        self.report_uri = report_uri

    def make_view(self):
        title_text = rc.Heading(
            "Personally Identifiable Information Operator Report", level=1
        )
        time_proceed = rc.Metric(
            heading="Ran at",
            value=self.report_spec.run_summary.timestamp or "today",
        )
        report_description = rc.Text(PII_REPORT_DESCRIPTION)

        structure = rc.Block(
            rc.Select(
                blocks=[
                    rc.Group(
                        self._build_summary_page(),
                        label="Summary",
                    ),
                    rc.Group(
                        self._build_details_page(),
                        label="Details",
                    ),
                ],
                type=rc.SelectType.TABS,
            )
        )
        self.report_sections = [title_text, report_description, time_proceed, structure]
        return self

    def save_report(self, report_sections=None, report_uri=None, storage_options={}):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_local_path = os.path.join(temp_dir, "___report.html")
            disable_print()
            with rc.ReportCreator("My Report") as report:
                report.save(
                    rc.Block(report_sections or self.report_sections), report_local_path
                )
            enable_print()

            report_uri = report_uri or self.report_uri
            with open(report_local_path) as f1:
                with fsspec.open(
                    report_uri,
                    "w",
                    **storage_options,
                ) as f2:
                    f2.write(f1.read())

    def _build_summary_page(self):
        summary = rc.Block(
            rc.Heading("PII Summary", level=1),
            rc.Text(self._get_summary_desc()),
            rc.Select(
                blocks=[
                    self._make_summary_stats_card(),
                    self._make_yaml_card(),
                    self._make_model_card(),
                ],
                type=rc.SelectType.TABS,
            ),
        )

        return summary

    def _build_details_page(self):
        details = rc.Block(
            rc.Text(DETAILS_REPORT_DESCRIPTION),
            rc.Select(
                blocks=[
                    row.build_report() for row in self.rows_details
                ],  # RowReportFields
                type=rc.SelectType.DROPDOWN,
                label="Details",
            ),
        )

        return details

    def _make_summary_stats_card(self) -> rc.Group:
        """
        Shows summary statics
        1. total rows
        2. total entites
        3. time_spent/row
        4. entities distribution
        5. resolved Entities in sample data - optional
        """
        try:
            process_rate = compute_rate(
                self.report_spec.run_summary.elapsed_time,
                self.report_spec.run_summary.total_rows,
            )
        except Exception as e:
            logger.warning("Failed to compute processing rate.")
            logger.debug(f"Full traceback: {e}")
            process_rate = "-"

        summary_stats = [
            rc.Heading("Summary Statistics", level=2),
            rc.Group(
                rc.Metric(
                    heading="Total No. Of Rows",
                    value=self.report_spec.run_summary.total_rows or "unknown",
                ),
                rc.Metric(
                    heading="Total No. Of Entites Proceed",
                    value=self.report_spec.run_summary.total_tokens,
                ),
                rc.Metric(
                    heading="Rows per second processed",
                    value=process_rate,
                ),
                rc.Metric(
                    heading="Total Time Spent",
                    value=human_time_friendly(
                        self.report_spec.run_summary.elapsed_time
                    ),
                ),
                columns=2,
            ),
            rc.Heading("Entities Distribution", level=3),
            plot_pie(self.report_spec.run_summary.statics),
        ]
        if self.report_spec.run_summary.show_sensitive_info:
            entites_df = self._build_total_entity_df()
            summary_stats.append(rc.Heading("Resolved Entities", level=3))
            summary_stats.append(rc.DataTable(entites_df, index=True))
        return rc.Group(summary_stats, label="STATS")

    def _make_yaml_card(self) -> rc.Group:
        """Shows the full pii config yaml."""
        yaml_appendix_title = rc.Heading("Reference: YAML File", level=2)
        yaml_appendix = rc.Yaml(self.report_spec.run_summary.config.to_dict())
        return rc.Group(yaml_appendix_title, yaml_appendix, label="YAML")

    def _make_model_card(self) -> rc.Group:
        """Generates model card."""
        if len(self.report_spec.run_summary.selected_spacy_model) == 0:
            return rc.Group(
                rc.Text("No model used."),
                label="MODEL CARD",
            )

        model_cards = [
            rc.Group(
                make_model_card(model_name=x.get("model")),
                label=x.get("model"),
            )
            for x in self.report_spec.run_summary.selected_spacy_model
        ]

        if len(model_cards) <= 1:
            return rc.Group(
                model_cards,
                label="MODEL CARD",
            )
        return rc.Group(
            rc.Select(
                model_cards,
                type=rc.SelectType.TABS,
            ),
            label="MODEL CARD",
        )

    def _build_total_entity_df(self) -> pd.DataFrame:
        frames = []
        for row in self.rows_details:  # RowReportFields
            frames.append(build_entity_df(entites=row.spec.entities, id=row.spec.id))

        result = pd.concat(frames)
        return result

    def _get_summary_desc(self) -> str:
        entities_mark_down = [
            "**" + ent + "**" for ent in self.report_spec.run_summary.selected_entities
        ]

        model_description = ""
        for spacy_model in self.report_spec.run_summary.selected_spacy_model:
            model_description = (
                model_description
                + f"You chose the **{spacy_model.get('model', 'unknown model')}** model for **{spacy_model.get('spacy_entites', 'unknown entities')}** detection."
            )
        if model_description:
            model_description = (
                model_description
                + "You can view the model details under the ``MODEL CARD`` tab."
            )

        SUMMARY_REPORT_DESCRIPTION_TEMPLATE = f"""
        This report will detail the statistics and configuration of the redaction process.The report will contain information such as the number of rows processed, the number of entities redacted, and so on. The report will provide valuable insight into the performance of the PII tool and facilitate any necessary adjustments to improve its performance.

        Based on the configuration file (you can view the YAML details under the ``YAML`` tab), you selected the following entities: {entities_mark_down}.
        {model_description}
        """
        return SUMMARY_REPORT_DESCRIPTION_TEMPLATE

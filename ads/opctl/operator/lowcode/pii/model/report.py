#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


# helper function to make report
import yaml
import plotly.express as px
import pandas as pd
import datapane as dp
import random
import plotly.graph_objects as go
import fsspec


PII_REPORT_DESCRIPTION = (
    "This report will offer a comprehensive overview of the redaction of personal identifiable information (PII) from the provided data."
    "The `Summary` section will provide an executive summary of this process, including key statistics, configuration, and model usage."
    "The `Details` section will offer a more granular analysis of each row of data, including relevant statistics."
)
DETAILS_REPORT_DESCRIPTION = "The following report will show the details on each row. You can view the highlighted named entities and their labels in the text under `TEXT` tab."


################
# Others utils #
################
def compute_rate(elapsed_time, num_unit):
    return elapsed_time / num_unit


def human_time_friendly(seconds):
    TIME_DURATION_UNITS = (
        ("week", 60 * 60 * 24 * 7),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("min", 60),
    )
    if seconds == 0:
        return "inf"
    accumulator = []
    for unit, div in TIME_DURATION_UNITS:
        amount, seconds = divmod(float(seconds), div)
        if amount > 0:
            accumulator.append(
                "{} {}{}".format(int(amount), unit, "" if amount == 1 else "s")
            )
    accumulator.append("{} secs".format(round(seconds, 2)))
    return ", ".join(accumulator)


FLAT_UI_COLORS = [
    "#1ABC9C",
    "#2ECC71",
    "#3498DB",
    "#9B59B6",
    "#34495E",
    "#16A085",
    "#27AE60",
    "#2980B9",
    "#8E44AD",
    "#2C3E50",
    "#F1C40F",
    "#E67E22",
    "#E74C3C",
    "#ECF0F1",
    "#95A5A6",
    "#F39C12",
    "#D35400",
    "#C0392B",
    "#BDC3C7",
    "#7F8C8D",
]
LABEL_TO_COLOR_MAP = {}


# all spacy model: https://huggingface.co/spacy
# "en_core_web_trf": "https://huggingface.co/spacy/en_core_web_trf/raw/main/README.md",
def make_model_card(model_name="", readme_path=""):
    """Make render model_readme.md as model card."""
    readme_path = (
        f"https://huggingface.co/spacy/{model_name}/raw/main/README.md"
        if model_name
        else readme_path
    )
    if not readme_path:
        raise NotImplementedError("Does not support other spacy model so far.")

    with fsspec.open(readme_path, "r") as file:
        content = file.read()
        _, front_matter, text = content.split("---", 2)
        data = yaml.safe_load(front_matter)

    try:
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
        eval_res_tb = dp.Plot(data=fig, caption="Evaluation Results")
    except:
        eval_res_tb = dp.Text("-")
        print(
            "The given readme.md doesn't have correct template for Evaluation Results."
        )

    return dp.Group(
        dp.Text(text),
        eval_res_tb,
        columns=2,
    )


################
# Report utils #
################
def map_label_to_color(labels):
    label_to_colors = {}
    for label in labels:
        label = label.lower()
        label_to_colors[label] = LABEL_TO_COLOR_MAP.get(
            label, random.choice(FLAT_UI_COLORS)
        )
        LABEL_TO_COLOR_MAP[label] = label_to_colors[label]

    return label_to_colors


def plot_pie(count_map) -> dp.Plot:
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
    return dp.Plot(fig)


def build_entity_df(entites, id) -> pd.DataFrame:
    text = [ent.text for ent in entites]
    types = [ent.type for ent in entites]
    # pos = [f"{ent.beg}" + ":" + f"{ent.end}" for ent in entites]
    replaced_values = [
        ent.replacement_string or "{{" + ent.placeholder + "}}" for ent in entites
    ]
    d = {
        "rowID": id,
        "Entity (Original Text)": text,
        "Type": types,
        "Redacted To": replaced_values,
        # "Beg: End": pos,
    }
    df = pd.DataFrame(data=d)
    if df.size == 0:
        # Datapane does not support empty dataframe, append a dummy row
        df2 = {
            "rowID": id,
            "Entity (Original Text)": "-",
            "Type": "-",
            "Redacted To": "-",
            # "Begs: End": "-",
        }
        df = df.append(df2, ignore_index=True)
    return df


class RowReportFields:
    # TODO: rename class
    def __init__(self, context, show_sensitive_info: bool = True):
        self.total_tokens = context.get("total_tokens", "unknown")
        self.entites_cnt_map = context.get("statics", {})
        self.raw_text = context.get("raw_text", "")
        self.id = context.get("id", "")
        self.show_sensitive_info = show_sensitive_info
        self.entities = context.get("entities")

    def build_report(self) -> dp.Group:
        return dp.Group(
            dp.Select(
                blocks=[
                    self._make_stats_card(),
                    self._make_text_card(),
                ],
                type=dp.SelectType.TABS,
            ),
            label="rowId: " + str(self.id),
        )

    def _make_stats_card(self):
        stats = [
            dp.Text("## Row Summary Statistics"),
            dp.BigNumber(
                heading="Total No. Of Entites Proceed",
                value=self.total_tokens,
            ),
            dp.Text(f"### Entities Distribution"),
            plot_pie(self.entites_cnt_map),
        ]
        if self.show_sensitive_info:
            stats.append(dp.Text(f"### Resolved Entities"))
            stats.append(
                dp.DataTable(
                    build_entity_df(self.entities, id=self.id),
                    label="Resolved Entities",
                )
            )
        return dp.Group(blocks=stats, label="STATS")

    def _make_text_card(self):
        annotations = []
        labels = set()
        for ent in self.entities:
            annotations.append((ent.beg, ent.end, ent.type))
            labels.add(ent.type)

        d = {"Content": [self.raw_text], "Annotations": [annotations]}
        df = pd.DataFrame(data=d)

        render_html = df.ads.render_ner(
            options={
                "default_color": "#D6D3D1",
                "colors": map_label_to_color(labels),
            },
            return_html=True,
        )
        return dp.Group(dp.HTML(render_html), label="TEXT")


class PIIOperatorReport:
    def __init__(self, context: dict):
        # set useful field for generating report from context
        summary_context = context.get("run_summary", {})
        self.config = summary_context.get("config", {})  # for generate yaml
        self.show_sensitive_info = summary_context.get("show_sensitive_info", True)
        self.show_rows = summary_context.get("show_rows", 25)
        self.total_rows = summary_context.get("total_rows", "unknown")
        self.total_tokens = summary_context.get("total_tokens", "unknown")
        self.elapsed_time = summary_context.get("elapsed_time", 0)
        self.entites_cnt_map = summary_context.get("statics", {})
        self.selected_entities = summary_context.get("selected_entities", [])
        self.spacy_detectors = summary_context.get("selected_spacy_model", [])
        self.run_at = summary_context.get("timestamp", "today")

        rows = context.get("run_details", {}).get("rows", [])
        rows = rows[0 : self.show_rows]
        self.rows_details = [
            RowReportFields(r, self.show_sensitive_info) for r in rows
        ]  # List[RowReportFields], len=show_rows

        self._validate_fields()

    def _validate_fields(self):
        """Check if any fields are empty."""
        # TODO
        pass

    def make_view(self):
        title_text = dp.Text("# Personally Identifiable Information Operator Report")
        time_proceed = dp.BigNumber(
            heading="Ran at",
            value=self.run_at,
        )
        report_description = dp.Text(PII_REPORT_DESCRIPTION)

        structure = dp.Blocks(
            dp.Select(
                blocks=[
                    dp.Group(
                        self._build_summary_page(),
                        label="Summary",
                    ),
                    dp.Group(
                        self._build_details_page(),
                        label="Details",
                    ),
                ],
                type=dp.SelectType.TABS,
            )
        )
        self.report_sections = [title_text, report_description, time_proceed, structure]
        return self.report_sections

    def save_report(self, report_sections, report_path):
        dp.save_report(
            report_sections or self.report_sections,
            path=report_path,
            open=False,
        )
        return report_path

    def _build_summary_page(self):
        summary = dp.Blocks(
            dp.Text("# PII Summary"),
            dp.Text(self._get_summary_desc()),
            dp.Select(
                blocks=[
                    self._make_summary_stats_card(),
                    self._make_yaml_card(),
                    self._make_model_card(),
                ],
                type=dp.SelectType.TABS,
            ),
        )

        return summary

    def _build_details_page(self):
        details = dp.Blocks(
            dp.Text(DETAILS_REPORT_DESCRIPTION),
            dp.Select(
                blocks=[
                    row.build_report() for row in self.rows_details
                ],  # RowReportFields
                type=dp.SelectType.DROPDOWN,
                label="Details",
            ),
        )

        return details

    def _make_summary_stats_card(self) -> dp.Group:
        """
        Shows summary statics
        1. total rows
        2. total entites
        3. time_spent/row
        4. entities distribution
        5. resolved Entities in sample data - optional
        """
        summary_stats = [
            dp.Text("## Summary Statistics"),
            dp.Group(
                dp.BigNumber(
                    heading="Total No. Of Rows",
                    value=self.total_rows,
                ),
                dp.BigNumber(
                    heading="Total No. Of Entites Proceed",
                    value=self.total_tokens,
                ),
                dp.BigNumber(
                    heading="Rows per second processed",
                    value=compute_rate(self.elapsed_time, self.total_rows),
                ),
                dp.BigNumber(
                    heading="Total Time Spent",
                    value=human_time_friendly(self.elapsed_time),
                ),
                columns=2,
            ),
            dp.Text(f"### Entities Distribution"),
            plot_pie(self.entites_cnt_map),
        ]
        if self.show_sensitive_info:
            entites_df = self._build_total_entity_df()
            summary_stats.append(dp.Text(f"### Resolved Entities"))
            summary_stats.append(dp.DataTable(entites_df))
        return dp.Group(blocks=summary_stats, label="STATS")

    def _make_yaml_card(self) -> dp.Group:
        # show pii config yaml
        yaml_string = yaml.dump(self.config, Dumper=yaml.SafeDumper)
        yaml_appendix_title = dp.Text(f"## Reference: YAML File")
        yaml_appendix = dp.Code(code=yaml_string, language="yaml")
        return dp.Group(blocks=[yaml_appendix_title, yaml_appendix], label="YAML")

    def _make_model_card(self) -> dp.Group:
        # show each model card
        model_cards = [
            dp.Group(
                make_model_card(model_name=x.get("model")),
                label=x.get("model"),
            )
            for x in self.spacy_detectors
        ]

        if len(model_cards) <= 1:
            return dp.Group(
                blocks=model_cards,
                label="MODEL CARD",
            )
        return dp.Group(
            dp.Select(
                blocks=model_cards,
                type=dp.SelectType.TABS,
            ),
            label="MODEL CARD",
        )

    def _build_total_entity_df(self) -> pd.DataFrame:
        frames = []
        for row in self.rows_details:  # RowReportFields
            frames.append(build_entity_df(entites=row.entities, id=row.id))

        result = pd.concat(frames)
        return result

    def _get_summary_desc(self) -> str:
        entities_mark_down = ["**" + ent + "**" for ent in self.selected_entities]

        model_description = ""
        for spacy_model in self.spacy_detectors:
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

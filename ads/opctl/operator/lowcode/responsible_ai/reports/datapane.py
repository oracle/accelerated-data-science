import json
import os
import sys
import altair as alt
import datapane as dp
import pandas as pd


with open(
    os.path.join(os.path.dirname(__file__), "..", "metric_descriptions.json"),
    "r",
    encoding="utf-8",
) as f:
    METRICS = json.load(f)


def make_page(metric: str, df: pd.DataFrame):
    title = metric.title()
    groups = []
    for column in df.columns:
        if column in ["predictions", "references"]:
            continue
        fig = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                alt.X(column, bin=True),
                y="count()",
                tooltip="count()",
            )
        )
        stats = dp.Group(
            dp.BigNumber("Max", round(df[column].max(), 4)),
            dp.BigNumber("Min", round(df[column].min(), 4)),
            dp.BigNumber("Mean", round(df[column].mean(), 4)),
            columns=2,
        )
        box_plot = (
            alt.Chart(df).mark_boxplot().encode(alt.X(column), tooltip=["text", column])
        )
        plots = dp.Group(fig, box_plot)
        groups.append(dp.Group(stats, plots, columns=2, label=column))
    visual = dp.Select(*groups) if len(groups) > 1 else groups[0]
    description = ""
    if metric in METRICS:
        metric_info = METRICS.get(metric)
        description = (
            metric_info.get("description", "")
            + "\n\nSee also:\n"
            + "\n".join([f"* {ref}" for ref in metric_info.get("references")])
        )
    return dp.Page(
        title=title,
        blocks=[
            f"# {title}",
            description,
            visual,
            dp.DataTable(df),
        ],
    )

def make_view(data_list: list):
    return dp.Blocks(*[make_page(item["metric"], item["data"]) for item in data_list])


def main():
    data_list = []
    for csv_file in sys.argv[1:]:
        data_list.append(
            {
                "metric": os.path.splitext(os.path.basename(csv_file))[0],
                "data": pd.read_csv(csv_file),
            }
        )
    view = make_view(data_list)
    dp.save_report(view, "report.html")


if __name__ == "__main__":
    main()

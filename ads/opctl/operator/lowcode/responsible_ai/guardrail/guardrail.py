import pandas as pd
from ..utils import to_dataframe, postprocess_sentence_level_dataframe
import logging
from ads.common import auth as authutil
import os
import nltk
import os
from ..reports.datapane import make_view
import datapane as dp

import copy
import numpy as np
from .rails.huggingface_rails import (
    HuggingFaceGeneric,
    HuggingFaceHonestHurtfulSentence,
    HuggingFaceRegardPolarity,
)


metric_mapping = {
    "honest": HuggingFaceHonestHurtfulSentence,
    "regard": HuggingFaceRegardPolarity,
    "toxicty": HuggingFaceGeneric,
}


class MetricLoader:
    def __init__(self):
        self.evaluator = None

    def load(self, metric_config: dict):
        load_args = copy.copy(metric_config.get("load_args", {}))
        path = load_args.get("path")
        return metric_mapping.get(path, HuggingFaceGeneric)

    def compute(self, predictions, **kwargs):
        return self.evaluator.compute(predictions, **kwargs)


class GuardRail:
    """Guard Rails."""

    def __init__(self, config: dict, auth: dict = None):
        self.config = config
        self.spec = self.config.get("spec", {})
        self.data = None
        self.auth = auth or authutil.default_signer()
        self.sentence_level = False
        self.output_directory = self.spec.get("output_directory", {}).get("url", None)
        self.report_file_name = self.spec.get("report_file_name")
        self.metrics_spec = self.spec.get("metrics", [{}])
        self.registered_gardrails = {}
        self.compute_args = {}
        self.thresholds = {}
        self.scores = {}
        self.load()

    def load_data(self):
        references = None
        predictions = None
        if self.spec.get("test_data").get("url", None):
            data_path = self.spec.get("test_data").get("url")
            pred_col = self.spec.get("test_data").get("text", "text")
            reference_col = self.spec.get("test_data").get("references", "references")

            if data_path.startswith("oci://"):
                self.data = pd.read_csv(data_path, storage_options=self.auth)
            else:
                self.data = pd.read_csv(data_path)
            if self.spec.get("sentence_level"):
                df_list = (
                    self.data["text"]
                    .apply(nltk.sent_tokenize)
                    .apply(lambda x: pd.DataFrame(x, columns=["text"]))
                    .tolist()
                )
                for idx, item in enumerate(df_list):
                    item["index"] = idx
                self.sentence_level_data = pd.concat(df_list)
                self.sentence_level = True

            if reference_col in self.data.columns:
                references = self.data[reference_col]

            if self.sentence_level and not references:
                predictions = self.sentence_level_data[pred_col].tolist()
            else:
                predictions = self.data[pred_col]

        else:
            predictions = self.spec.get("test_data").get("text")
            references = self.spec.get("test_data").get("references", None)
        return predictions, references

    def load(
        self,
    ):
        for metric_config in self.metrics_spec:
            logging.debug(f"Metric: {metric_config}")
            print(f"Metric: {metric_config}")
            name = metric_config.get("name", "Unknown Metric")
            load_args = metric_config.get("load_args", {})
            compute_args = metric_config.get("compute_args", {})
            threshold = metric_config.get("filter", None)
            logging.debug(name)
            logging.debug(load_args)
            self.registered_gardrails[name] = MetricLoader().load(metric_config)(
                load_args
            )
            self.compute_args[name] = compute_args
            self.thresholds[name] = threshold

    def evaluate(self, predictions, references=None):
        scores = {}
        for name, guardrail in self.registered_gardrails.items():
            score = guardrail.compute(
                predictions=predictions,
                references=references,
                **self.compute_args[name],
            )
            scores[name] = (score, guardrail.description, guardrail.homepage)
        # post process score
        # - converting to dataframe
        # - save dataframe
        # - adding prediction column

        for name, (score, description, homepage) in scores.items():
            df = to_dataframe(score)

            if len(df) == len(predictions):
                df["text"] = predictions
                if self.sentence_level:
                    df["index"] = self.sentence_level_data["index"].tolist()
                    df = postprocess_sentence_level_dataframe(df)
                self.scores[name] = (df, description, homepage)
            if self.output_directory:
                os.makedirs(self.output_directory, exist_ok=True)
                df.to_csv(
                    f"{os.path.join(self.output_directory, name)}.csv", index=False
                )

        return self.scores

    def generate_report(self):
        predictions, references = self.load_data()
        scores = self.evaluate(predictions, references)
        data = []

        for name, (df, description, homepage) in scores.items():
            data.append(
                {
                    "metric": name,
                    "data": df,
                    "description": description,
                    "homepage": homepage,
                }
            )
        dp.enable_logging()
        if not self.sentence_level:
            view = make_view(data)
            dp.save_report(
                view,
                os.path.join(
                    os.path.expanduser(self.output_directory), self.report_file_name
                ),
            )
            return view
        else:
            return data

    def apply_filter(self):
        filters = {}
        for name, threshold in self.thresholds.items():
            for col in self.scores[name][0].columns:
                if col != "text":
                    filters[name] = (self.scores[name][0][col] <= threshold).values

        return np.logical_and(*list(filters.values()))

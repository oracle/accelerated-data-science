import pandas as pd
import logging
from ads.common import auth as authutil
import os
import nltk
import os
from ..reports.datapane import make_view
import datapane as dp
import random

import copy
from .rails.huggingface_rails import (
    HuggingFaceGeneric,
    HuggingFaceHonestHurtfulSentence,
    HuggingFaceRegardPolarity,
)
from .rails.fact_checking import FactChecking
from typing import Union, List
from ..utils import init_endpoint
import numpy as np
from abc import ABC, abstractmethod


metric_mapping = {
    "honest": HuggingFaceHonestHurtfulSentence,
    "regard": HuggingFaceRegardPolarity,
    "toxicty": HuggingFaceGeneric,
    "fact_checking": FactChecking,
}


class MetricLoader:
    def __init__(self):
        self.evaluator = None

    def load(self, metric_config: dict):
        load_args = copy.copy(metric_config.get("evaluation", {}).get("load_args", {}))
        path = load_args.get("path")
        return metric_mapping.get(path, HuggingFaceGeneric)

    def compute(self, predictions, **kwargs):
        return self.evaluator.compute(predictions, **kwargs)


class BaseGuardRail(ABC):
    """Guard Rails."""

    def __init__(self, config: dict, auth: dict = None):
        self.config = config
        self.spec = self.config.get("spec", {})
        self.data = None
        self.auth = auth or authutil.default_signer()
        self.sentence_level = False
        self.sentence_level_data = None
        self.output_directory = self.spec.get("output_directory", {}).get("url", None)
        self.report_file_name = self.spec.get("report_file_name")
        self.guardrails_spec = self.spec.get("guardrails", [{}])
        self.registered_gardrails = {}
        self.paths = {}
        self.compute_args = {}
        self.thresholds = {}
        self.directions = {}
        self.scores = {}
        self.load()

    def load_data(self):
        references = None
        predictions = None
        if self.spec.get("test_data").get("url", None):
            data_path = self.spec.get("test_data").get("url")
            pred_col = self.spec.get("test_data").get("text_col", "text")
            reference_col = self.spec.get("test_data").get("ref_col", "references")
            if data_path.endswith(".csv"):
                if data_path.startswith("oci://"):
                    self.data = pd.read_csv(data_path, storage_options=self.auth)
                else:
                    self.data = pd.read_csv(data_path)
            elif data_path.endswith(".jsonl"):
                if data_path.startswith("oci://"):
                    self.data = pd.read_json(data_path, storage_options=self.auth, lines=True)
                else:
                    self.data = pd.read_json(data_path, lines=True)

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
        for metric_config in self.guardrails_spec:
            logging.debug(f"Metric: {metric_config}")
            print(f"Metric: {metric_config}")
            name = metric_config.get("name", "Unknown Metric")

            load_args = metric_config.get("evaluation", {}).get(
                "load_args", {}
            ) or metric_config.get("load_args", {})
            compute_args = metric_config.get("evaluation", {}).get(
                "compute_args", {}
            ) or metric_config.get("compute_args", {})
            self.compute_args[name] = compute_args

            path = load_args.get("path", None)
            assert path is not None, f"`path` is not specified for metric: {name}."
            self.paths[name] = path

            logging.debug(f"`name`: {name}")
            logging.debug(f"`path`: {path}")
            logging.debug(load_args)

            self.registered_gardrails[name] = MetricLoader().load(
                metric_config,
            )(name=name, config=metric_config)

            threshold = metric_config.get("action", {}).get("threshold", 1)
            self.thresholds[name] = threshold

            direction = metric_config.get("action", {}).get("direction", "<=")
            self.directions[name] = direction

    def compute(
        self,
        predictions: Union[List, pd.Series],
        references: Union[List, pd.Series] = None,
        **kwargs,
    ):
        for name, guardrail in self.registered_gardrails.items():
            # adding extra variables needed for fact checking.
            if self.paths[name] == "fact_checking":
                self.compute_args[name]["prompt"] = kwargs.get("prompt")
                if not (hasattr(self, "llm") and getattr(self, "llm")):
                    self.llm = init_endpoint(
                        name=guardrail.config.get("evaluation", {})
                        .get("compute_args", {})
                        .get("llm_model")
                    )
                self.compute_args[name]["endpoint"] = self.llm

            score = guardrail.compute(
                predictions=predictions,
                references=references,
                **self.compute_args[name],
            )
            df_score = guardrail.postprocessing(
                score=score,
                predictions=predictions,
                sentence_level=self.sentence_level,
                sentence_level_index=self.sentence_level_data,
                output_directory=self.output_directory,
            )

            self.scores[name] = (df_score, guardrail.description, guardrail.homepage)

        return self.scores


class OfflineGuardRail(BaseGuardRail):
    def generate_report(self, **kwargs):
        predictions, references = self.load_data()
        scores = self.compute(predictions, references, **kwargs)
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
            if self.output_directory:
                dp.save_report(
                    view,
                    os.path.join(
                        os.path.expanduser(self.output_directory), self.report_file_name
                    ),
                )
            return view
        else:
            return data


class OnlineGuardRail(BaseGuardRail):
    def __init__(self, config: dict, auth: dict = None):
        super().__init__(config, auth)
        model_name = config.get("spec", {}).get("llm_model")
        self.llm = init_endpoint(name=model_name) if model_name else None
        self.n_generations = config.get("spec", {}).get("n_generations", 1)

    def predict(self, prompts: Union[List[str], str] = None) -> Union[List[str], str]:
        if not prompts:
            prompts, _ = self.load_data()
        if isinstance(prompts, str):
            prompts = [prompts]
        final_output = []
        for prompt in prompts:
            responses = self.llm.batch_generate(
                prompt=prompt, num_generations=self.n_generations
            )
            final_output.append(
                self.apply_guardrail(predictions=responses, prompt=prompt)
            )

        return final_output if len(final_output) > 1 else final_output[0]

    def apply_guardrail(
        self,
        predictions: Union[List, pd.Series],
        references: Union[List, pd.Series] = None,
        **kwargs,
    ):
        filters = np.array([True] * len(predictions))
        for name, guardrail in self.registered_gardrails.items():
            # adding extra variables needed for fact checking.
            if self.paths[name] == "fact_checking":
                self.compute_args[name]["prompt"] = kwargs.get("prompt")
                if not (hasattr(self, "llm") and getattr(self, "llm")):
                    self.llm = init_endpoint(
                        name=guardrail.config.get("evaluation", {})
                        .get("compute_args", {})
                        .get("llm_model")
                    )
                self.compute_args[name]["endpoint"] = self.llm
            score = guardrail.compute(
                predictions=predictions,
                references=references,
                **self.compute_args[name],
            )
            # postprocessing to convert to pandas dataframe
            df_score = guardrail.postprocessing(
                score=score,
                predictions=predictions,
                sentence_level=self.sentence_level,
                sentence_level_index=self.sentence_level_data,
                output_directory=self.output_directory,
            )

            filters = np.logical_and(
                filters,
                guardrail.apply_filter(score=df_score, direction=self.directions[name]),
            )
            # early stop
            if not any(filters):
                return self.spec.get(
                    "custom_msg", "I am sorry. I cannot answer this question."
                )
        if isinstance(predictions, list):
            predictions = pd.Series(predictions)
        # random select from all the ones that passed the filter
        return random.choice(predictions[filters].values)

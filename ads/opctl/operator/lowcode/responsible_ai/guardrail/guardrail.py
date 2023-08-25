import evaluate
import pandas as pd
from ..utils import to_dataframe, postprocess_sentence_level_dataframe
import logging
from ads.common import auth as authutil
import os
import importlib
import nltk
import importlib.util
import os
import sys
from ..reports.datapane import make_view
import datapane as dp
from typing import Union
from abc import ABC, abstractmethod, abstractproperty


class BaseGuardRail(ABC):
    def __init__(self, config: dict):
        self.evaluator = self.load(config)

    def load(self):
        pass

    @abstractmethod
    def compute(self, predictions: Union[pd.Series, list], references: Union[pd.Series, list] = None,
        **kwargs: dict,):
        pass

    @property
    def description(self):
        return self.evaluator.description if self.evaluator is not None and hasattr(self.evaluator, "description") else ""
    
    @property
    def homepage(self):
        return self.evaluator.homepage if self.evaluator is not None and hasattr(self.evaluator, "homepage") else ""
    

class HuggingFaceHonestHurtfulSentence(BaseGuardRail):

    def load(self, load_args: dict):
        return evaluate.load(**load_args)

    def compute(
        self,
        predictions: Union[pd.Series, list],
        references: Union[pd.Series, list] = None,
        **kwargs: dict,
    ):
        if not (isinstance(predictions, list) and isinstance(predictions[0], list)):
            predictions = [sentence.split() for sentence in predictions]
            references = [sentence.split() for sentence in references] if references else None
        score = self.evaluator.compute(predictions=predictions, references=references, **kwargs)
        return score


class HuggingFaceGeneric(BaseGuardRail):

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, predictions: pd.Series, references: pd.Series=None, **kwargs: dict):

        score = self.evaluator.compute(
            predictions=predictions, references=references, **kwargs
        )
        return score


class HuggingFaceRegardPolarity(BaseGuardRail):
    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, predictions: pd.Series, references: pd.Series=None, **kwargs: dict):

        score = self.evaluator.compute(
        data=predictions, references=references, **kwargs
    )
        return score


metric_mapping = {"honest": HuggingFaceHonestHurtfulSentence, "regard": HuggingFaceRegardPolarity, "toxicty": HuggingFaceGeneric}


class MetricLoader:

    def __init__(self):
        self.evaluator = None

    def load(self, metric_type: str, metric_config: dict):
        load_args = metric_config.get("load_args", {})
        if metric_type == "huggingface":
            path = load_args.get("path")
            return metric_mapping.get(path, HuggingFaceGeneric)
        elif metric_type == "custom":
            module_path = load_args.pop("path", None)
            class_name = metric_config.get("class_name") or "CustomGuardRail"

            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(
                module_name,
                module_path,
            )
            user_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = user_module
            spec.loader.exec_module(user_module)

            custom_guardrail_class = getattr(user_module, class_name)

            return custom_guardrail_class
        else:
            NotImplemented("Not supported type.")

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

    def load_data(self):
        references = None
        predictions = None
        if self.spec.get("test_data").get("url", None):

            data_path = self.spec.get("test_data").get("url")
            pred_col = self.spec.get("test_data").get("predictions", "predictions")
            reference_col = self.spec.get("test_data").get("references", "references")

            if data_path.startswith("oci://"):
                self.data = pd.read_csv(data_path, storage_options=self.auth)
            else:
                self.data = pd.read_csv(data_path)
            if self.spec.get("sentence_level"):
                df_list = (
                    self.data["predictions"]
                    .apply(nltk.sent_tokenize)
                    .apply(lambda x: pd.DataFrame(x, columns=["predictions"]))
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
            predictions =  self.spec.get("test_data").get("predictions")
            references =  self.spec.get("test_data").get("references", None)
        return predictions, references

    def load(self,):

        for metric_config in self.metrics_spec:
            logging.debug(f"Metric: {metric_config}")
            print(f"Metric: {metric_config}")
            name = metric_config.get("name", "Unknown Metric")
            metric_type = metric_config.get("type", "Unknown Type")

            load_args = metric_config.get("load_args", {})
            compute_args = metric_config.get("compute_args", {})
            logging.debug(name)
            logging.debug(load_args)
            self.registered_gardrails[name] = (MetricLoader().load(metric_type, metric_config)(load_args), compute_args)
        
    def evaluate(self, predictions, references=None):
        scores = {}
        for name, (guardrail, compute_args) in self.registered_gardrails.items():
            
            score = guardrail.compute(
                predictions=predictions,
                references=references,
                **compute_args,
            )
            scores[name] = (score, guardrail.description, guardrail.homepage)
        # post process score 
        # - converting to dataframe
        # - save dataframe
        # - adding prediction column
        res = {}
        for name, (score, description, homepage) in scores.items():
            score_dict = to_dataframe(score)
            
            for metric, df in score_dict.items():
                if len(df) == len(predictions):
                    df['predictions'] = predictions
                    if self.sentence_level:
                        df['index'] = self.sentence_level_data['index'].tolist()
                        df = postprocess_sentence_level_dataframe(df)
                    res[" ".join([name, metric])] = (df, description, homepage)
                if self.output_directory:
                    os.makedirs(self.output_directory, exist_ok=True)
                    df.to_csv(f'{os.path.join(self.output_directory, "_".join([name, metric]))}.csv', index=False)
        return res

    def generate_report(self):
        predictions, references = self.load_data()
        self.load()
        scores = self.evaluate(predictions, references)
        data = []

        for name, (df, description, homepage) in scores.items():
            data.append({"metric": name, "data": df, "description": description, "homepage": homepage})
        dp.enable_logging()
        if not self.sentence_level:
            view = make_view(data)
            dp.save_report(view, os.path.join(os.path.expanduser(self.output_directory), self.report_file_name))
            return view

        else:
            return data

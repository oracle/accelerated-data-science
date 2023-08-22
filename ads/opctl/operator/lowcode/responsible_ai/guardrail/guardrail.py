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


class HuggingFaceHonestHurtfulSentence:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: Union[pd.Series, list], references: Union[pd.Series, list]=None,  **kwargs: dict):
        preds = [sentence.split() for sentence in predictions]
                    
        refs = [sentence.split() for sentence in references] if references else None
        score = evaluator.compute(
        predictions=preds, references=refs, **kwargs
    )
        return score
    

class HuggingFaceGeneric:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: Union[pd.Series, list], references: Union[pd.Series, list]=None, **kwargs: dict):
        score = evaluator.compute(
        predictions=predictions, references=references, **kwargs
    )
        return score


class HuggingFaceRegardPolarity:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: Union[pd.Series, list], references: Union[pd.Series, list]=None, **kwargs: dict):

        score = evaluator.compute(
        data=predictions, references=references, **kwargs
    )
        return score


metric_mapping = {"honest": HuggingFaceHonestHurtfulSentence, "regard": HuggingFaceRegardPolarity, "toxicity": HuggingFaceGeneric}


class MetricLoader:
    @staticmethod
    def load(metric_type: str, metric_config: dict):
        load_args = metric_config.get("load_args", {})
        if metric_type == "huggingface":
            path = load_args.get("path")
            return metric_mapping.get(path)
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


class GuardRail:
    """Guard Rails."""
    def __init__(self, config: dict, auth: dict=None):
        self.config = config
        self.data = None
        self.auth = auth or authutil.default_signer()
        self.sentence_level = False
        self.output_directory = None

    def load_data(self):
        spec = self.config.get("spec", {})
        if spec.get("test_data").get("url", None):
            data_path = spec.get("test_data").get("url")
            if data_path.startswith("oci://"):
                self.data = pd.read_csv(data_path, storage_options=self.auth)
            else:
                self.data = pd.read_csv(data_path)
            if spec.get("sentence_level"):
                df_list = self.data['predictions'].apply(nltk.sent_tokenize).apply(lambda x: pd.DataFrame(x, columns=['predictions'])).tolist()
                for idx, item in enumerate(df_list):
                    item['index'] = idx
                self.sentence_level_data = pd.concat(df_list)
                self.sentence_level = True


    def evaluate(self):
        spec = self.config.get("spec", {})
        metrics_spec = spec.get("metrics", [{}])
        self.output_directory = spec.get("output_directory", {}).get("url", None)

        scores = {}
        for metric_config in metrics_spec:
            logging.debug(f"Metric: {metric_config}")
            print(f"Metric: {metric_config}")
            name = metric_config.get("name", "Unknown Metric")
            metric_type = metric_config.get("type", "Unknown Type")
            load_args = metric_config.get("load_args", {})
            compute_args = metric_config.get("compute_args", {})
            logging.debug(name)
            logging.debug(load_args)

            
            reference_col = compute_args.pop("references", "references")
            if self.data is not None and reference_col in self.data.columns:
                self.references = self.data[reference_col]
            else:
                self.references = compute_args.pop("references", None)
            if self.sentence_level and not self.references:
                self.predictions = self.sentence_level_data[compute_args.pop("predictions", "predictions")].tolist() if self.sentence_level_data is not None else compute_args.pop("predictions")
            else:
                self.predictions = self.data[compute_args.pop("predictions", "predictions")] if self.data is not None else compute_args.pop("predictions")

            guardrail = MetricLoader.load(metric_type, metric_config)()

            score = guardrail.compute(evaluator=guardrail.load(load_args), predictions=self.predictions, references=self.references, **compute_args)

            scores[name] = score
        # post process score 
        # - converting to dataframe
        # - save dataframe
        # - adding prediction column
        res = {}
        for name, score in scores.items():
            score_dict = to_dataframe(score)
            
            for metric, df in score_dict.items():
                if len(df) == len(self.predictions):
                    df['predictions'] = self.predictions
                    if self.sentence_level:
                        df['index'] = self.sentence_level_data['index'].tolist()
                        df = postprocess_sentence_level_dataframe(df)
                if self.output_directory:
                    df.to_csv(f'{os.path.join(self.output_directory, "_".join([name, metric]))}.csv', index=False)
        return res

    def generate_report(self):
        self.load_data()
        scores = self.evaluate()
        data = []
        for name, score in scores.items():
            for metric, df in score.items():
                data.append({"metric": name, "data": df})
        dp.enable_logging()
        view = make_view(data)
        
        dp.save_report(view, os.path.join(self.output_directory, "report.html"))
        return view

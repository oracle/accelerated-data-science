import evaluate
import pandas as pd
from ..utils import to_dataframe
import logging
from ads.common import auth as authutil
import os
import importlib


class HuggingFaceHonestHurtfulSentence:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: pd.Series, references: pd.Series=None,  **kwargs: dict):
        preds = [sentence.split() for sentence in predictions]
                    
        refs = [sentence.split() for sentence in references] if references else None
        score = evaluator.compute(
        predictions=preds, references=refs, **kwargs
    )
        return score
    

class HuggingFaceToxicity:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: pd.Series, references: pd.Series=None, **kwargs: dict):

        score = evaluator.compute(
        predictions=predictions, references=references, **kwargs
    )
        return score

class HuggingFaceRegardPolarity:

    def load(self, load_args: dict):
        return evaluate.load(**load_args)
    
    def compute(self, evaluator, predictions: pd.Series, references: pd.Series=None, **kwargs: dict):

        score = evaluator.compute(
        data=predictions, references=references, **kwargs
    )
        return score


metric_mapping = {"honest": HuggingFaceHonestHurtfulSentence, "regard": HuggingFaceRegardPolarity, "toxicity": HuggingFaceToxicity}

class MetricLoader:
    @staticmethod
    def load(metric_type: str, load_args: dict):
        
        if metric_type == "huggingface":
            path = load_args.get("path")
            return metric_mapping.get(path)
        elif metric_type == "custom":
            path = load_args.pop("path")
            module = importlib.import_module(f"ads.opctl.operator.lowcode.responsible_ai.guardrail.{os.path.basename(path).split('.')[0]}")
            # sys.path.append(os.path.abspath(path))
            # module = eval(os.path.basename(path).split(".")[0])
            # from module import CustomGuardRail
            return module.CustomGuardRail
        else:
            NotImplemented("Not supported type.")


class GuardRail:
    """Guard Rails."""
    def __init__(self, config: dict, auth: dict=None):
        self.config = config
        self.data = None
        self.auth = auth or authutil.default_signer()

    def load_data(self):
        test_data_spec = self.config.get("spec", {}).get("test_data")
        if test_data_spec.get("url", None):
            data_path = test_data_spec.get("url")
            if data_path.startswith("oci://"):
                self.data = pd.read_csv(data_path, storage_options=self.auth)
            else:
                self.data = pd.read_csv(data_path)

    def evaluate(self):
        spec = self.config.get("spec", {})
        metrics = spec.get("Metrics", [{}])
        output_directory = spec.get("output_directory", {}).get("url", None)

        scores = {}
        for metric in metrics:
            logging.debug(f"Metric: {metric}")
            print(f"Metric: {metric}")
            name = metric.pop("name", "Unknown Metric")
            metric_type = metric.pop("type", "Unknown Type")
            load_args = metric.pop("load_args", {})
            compute_args = metric.pop("compute_args", {})
            logging.debug(name)
            logging.debug(load_args)
            self.predictions = self.data[compute_args.pop("predictions", "predictions")] if self.data is not None else compute_args.pop("predictions")
            reference_col = compute_args.pop("references", "references")
            if self.data is not None and reference_col in self.data.columns:
                self.references = self.data[reference_col]
            else:
                self.references = compute_args.pop("references", None)
                
            guardrail = MetricLoader.load(metric_type, load_args)()

            score = guardrail.compute(evaluator=guardrail.load(load_args), predictions=self.predictions, references=self.references, **compute_args)

            scores[name] = score
        res = {}
        for name, score in scores.items():
            res[name] = to_dataframe(score)
            if output_directory:
                for metric, df in res[name].items():
                    if len(df) == len(self.predictions):
                        df['predictions'] = self.predictions
                    df.to_csv(f'{os.path.join(output_directory, "_".join([name, metric]))}.csv')
        return res

    def generate_report(self):
        self.load_data()
        return self.evaluate()

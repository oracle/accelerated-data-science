import evaluate
from typing import Union
import pandas as pd
from .base import BaseGuardRail


class HuggingFaceHonestHurtfulSentence(BaseGuardRail):
    def load(
        self,
    ):
        return evaluate.load(**self.config.get("evaluation", {}).get("load_args", {}))

    def compute(
        self,
        predictions: Union[pd.Series, list],
        references: Union[pd.Series, list] = None,
        **kwargs: dict,
    ):
        if not (isinstance(predictions, list) and isinstance(predictions[0], list)):
            predictions = [sentence.split() for sentence in predictions]
            references = (
                [sentence.split() for sentence in references] if references else None
            )
        score = self.evaluator.compute(
            predictions=predictions, references=references, **kwargs
        )
        return score

    @property
    def description(self):
        return (
            "You selected HuggingFace Honest module to calculate the hurful sentence score. "
            + self.evaluator.description
        )

    @property
    def homepage(self):
        return "https://huggingface.co/spaces/evaluate-measurement/hurtfulsentence"


class HuggingFaceGeneric(BaseGuardRail):
    def load(self):
        return evaluate.load(**self.config.get("evaluation", {}).get("load_args", {}))

    def compute(
        self, predictions: pd.Series, references: pd.Series = None, **kwargs: dict
    ):
        score = self.evaluator.compute(
            predictions=predictions, references=references, **kwargs
        )
        return score

    @property
    def description(self):
        return (
            "You selected HuggingFace toxicity module to calculate the toxicity score. "
            + self.evaluator.description
        )

    @property
    def homepage(self):
        return "https://huggingface.co/spaces/evaluate-measurement/toxicity"


class HuggingFaceRegardPolarity(BaseGuardRail):
    def load(self):
        return evaluate.load(**self.config.get("evaluation", {}).get("load_args", {}))

    def compute(
        self, predictions: pd.Series, references: pd.Series = None, **kwargs: dict
    ):
        score = self.evaluator.compute(
            data=predictions, references=references, **kwargs
        )
        return score

    @property
    def description(self):
        return (
            "You selected HuggingFace regard module to calculate the polarity score. "
            + self.evaluator.description
        )

    @property
    def homepage(self):
        return "https://huggingface.co/spaces/evaluate-measurement/polarity"

from abc import ABC, abstractmethod, abstractproperty
from typing import Union, List
import pandas as pd
import numpy as np
from ...utils import to_dataframe, postprocess_sentence_level_dataframe, apply_filter
import os


class BaseGuardRail(ABC):
    def __init__(self, name: str, config: dict, **kwargs):
        self.name = name
        self.config = config
        self.evaluator = self.load()

    def load(self):
        pass

    @abstractmethod
    def compute(
        self,
        predictions: Union[pd.Series, list],
        **kwargs: dict,
    ):
        pass

    @property
    def description(self):
        return (
            self.evaluator.description
            if self.evaluator is not None and hasattr(self.evaluator, "description")
            else ""
        )

    @property
    def homepage(self):
        return (
            self.evaluator.homepage
            if self.evaluator is not None and hasattr(self.evaluator, "homepage")
            else ""
        )

    def apply_filter(self, score: pd.DataFrame, direction: str="<="):
        return apply_filter(score=score,
                            threshold=self.config.get("action", {}).get("threshold", 1),
                            direction=direction)

    def postprocessing(self, score: pd.DataFrame, predictions: Union[pd.Series, List], sentence_level: bool, sentence_level_index: List, output_directory=None):
        # post process score
        # - converting to dataframe
        # - save dataframe
        # - adding prediction column
        df = to_dataframe(score)
        if len(df) == len(predictions):
            df["text"] = predictions
            if sentence_level and sentence_level_index:
                df["index"] = sentence_level_index["index"].tolist()
                df = postprocess_sentence_level_dataframe(df)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            df.to_csv(
                f"{os.path.join(output_directory, self.name)}.csv", index=False
            )
        return df
